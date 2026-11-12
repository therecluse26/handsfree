"""Build one synthetic person, camera and gaze in Blender. Runs inside Blender's Python.

World coordinates
-----------------
The camera is at the world origin. It looks along world +Y, and world +Z is
up. So the camera's local axes are the handsfree camera space: x to the right
of the image, y up, and the person sits at negative z. World (x, y, z) is
camera space (x, z, -y). Lengths in Blender are metres; all values that leave
this module are millimetres in camera space.

The screen is in the camera plane (world y = 0), like a laptop screen.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import bpy
from mathutils import Matrix, Quaternion, Vector

import eye as eyelib

MM = 0.001
ASSETS = os.path.expanduser(os.environ.get("HANDSFREE_SYNTH_ASSETS", "~/.cache/handsfree/assets/makehuman/system"))

# MakeHuman draws its eyeball as a 16.4 mm sphere around the eye joint. A real
# eyeball has a 12 mm radius. The anatomical eye is put where its cornea apex
# is at the front of the MakeHuman eye.
MAKEHUMAN_EYE_FRONT_MM = 16.4


def cam_from_world(p: Vector) -> tuple[float, float, float]:
    """World metres to camera-space millimetres."""
    return (p.x / MM, p.z / MM, -p.y / MM)


def world_from_cam(x: float, y: float, z: float) -> Vector:
    """Camera-space millimetres to world metres."""
    return Vector((x * MM, -z * MM, y * MM))


@dataclass
class PersonParams:
    macro: dict = field(default_factory=dict)
    """MakeHuman macro values: gender, age, muscle, weight, race = {asian, caucasian, african}."""
    skin: str = "young_caucasian_male"
    eyebrows: str = "eyebrow001"
    eyelashes: str = "eyelashes01"
    eyes: tuple[eyelib.EyeParams, eyelib.EyeParams] = (eyelib.EyeParams(), eyelib.EyeParams())
    """(the person's right eye, the person's left eye). The right eye is on the image's left side."""
    kappa_h_deg: float = 5.0
    """Angle between the optical axis and the line of sight, horizontal. The line of sight is nasal of the optical axis."""
    kappa_v_deg: float = 1.5
    """The same angle, vertical. Positive: the line of sight is above the optical axis."""


@dataclass
class Person:
    params: PersonParams
    rig: bpy.types.Object
    body: bpy.types.Object
    eyes: tuple[bpy.types.Object, bpy.types.Object]
    eye_rest: tuple[Vector, Vector]
    """Eyeball centres in the rig's own coordinates."""
    pivot: Vector
    """Centre of head rotation in the rig's own coordinates."""


def build_person(p: PersonParams) -> Person:
    from bl_ext.blender_org.mpfb.services.humanservice import HumanService
    from bl_ext.blender_org.mpfb.services.targetservice import TargetService

    macro = TargetService.get_default_macro_info_dict()
    macro.update({k: v for k, v in p.macro.items() if k != "race"})
    if "race" in p.macro:
        macro["race"] = dict(p.macro["race"])
    body = HumanService.create_human(scale=0.1, macro_detail_dict=macro)
    HumanService.add_builtin_rig(body, "default")
    rig = body.parent
    HumanService.set_character_skin(os.path.join(ASSETS, "skins", p.skin, p.skin + ".mhmat"), body)
    for kind, name in (("Eyebrows", p.eyebrows), ("Eyelashes", p.eyelashes)):
        folder = kind.lower()
        HumanService.add_mhclo_asset(os.path.join(ASSETS, folder, name, name + ".mhclo"), body, asset_type=kind)

    eyes, rest = [], []
    for side, bone, params in (("R", "eye.R", p.eyes[0]), ("L", "eye.L", p.eyes[1])):
        joint = rig.data.bones[bone].head_local.copy()
        centre = joint + Vector((0.0, -(MAKEHUMAN_EYE_FRONT_MM - params.cornea_apex) * MM, 0.0))
        obj = eyelib.create_eye(params, f"eye.{side}")
        eyes.append(obj)
        rest.append(centre)
    mid = (rest[0] + rest[1]) / 2
    pivot = mid + Vector((0.0, 0.09, -0.04))  # about the top of the neck
    return Person(p, rig, body, (eyes[0], eyes[1]), (rest[0], rest[1]), pivot)


@dataclass
class Pose:
    head_yaw_deg: float = 0.0
    """Positive: the person turns the face to the image's right."""
    head_pitch_deg: float = 0.0
    """Positive: the face tilts up."""
    head_roll_deg: float = 0.0
    eye_mid_cam_mm: tuple[float, float, float] = (0.0, -100.0, -550.0)
    """Midpoint between the eyeball centres, camera space."""
    target_cam_mm: tuple[float, float, float] = (0.0, -120.0, 0.0)
    """The point the person looks at, camera space."""
    upper_lid_follow: float = 0.75
    """How much of the vertical eye rotation the upper lid follows."""
    lower_lid_follow: float = 0.35
    lid_open: float = 0.0
    """Extra opening (+) or closing (-) of the lids, degrees of lid rotation."""


def _head_matrix(person: Person, pose: Pose) -> Matrix:
    """World transform of the rig for this pose."""
    rot = (
        Matrix.Rotation(math.radians(pose.head_roll_deg), 4, "Y")
        @ Matrix.Rotation(math.radians(pose.head_yaw_deg), 4, "Z")
        @ Matrix.Rotation(math.radians(-pose.head_pitch_deg), 4, "X")
    )
    about_pivot = Matrix.Translation(person.pivot) @ rot @ Matrix.Translation(-person.pivot)
    mid = about_pivot @ ((person.eye_rest[0] + person.eye_rest[1]) / 2)
    return Matrix.Translation(world_from_cam(*pose.eye_mid_cam_mm) - mid) @ about_pivot


def _kappa(person: Person, side: int) -> Quaternion:
    """Rotation, in the eye's own frame, from the optical axis to the line of sight."""
    nasal = 1.0 if side == 0 else -1.0  # the right eye (index 0) is at -x; nasal is towards +x
    h = Quaternion((0.0, 0.0, 1.0), math.radians(person.params.kappa_h_deg) * nasal)
    v = Quaternion((1.0, 0.0, 0.0), math.radians(-person.params.kappa_v_deg))
    return h @ v


def apply_pose(person: Person, pose: Pose) -> dict:
    """Move the head, turn each eye to the target, move the lids. Returns the labels (camera space, mm)."""
    head = _head_matrix(person, pose)
    person.rig.matrix_world = head
    head_rot = head.to_quaternion()
    target = world_from_cam(*pose.target_cam_mm)
    labels = {"eyes": [], "target_cam_mm": list(pose.target_cam_mm)}
    pitches = []
    for side, (obj, rest) in enumerate(zip(person.eyes, person.eye_rest)):
        centre = head @ rest
        sight = (target - centre).normalized()
        # No torsion: the eye's local Z stays as near to the head's up axis as it can.
        up = head_rot @ Vector((0.0, 0.0, 1.0))
        y = -sight
        x = y.cross(up).normalized()
        z = x.cross(y).normalized()
        track = Matrix((x, y, z)).transposed().to_quaternion()
        rot = track @ _kappa(person, side).inverted()
        obj.matrix_world = Matrix.LocRotScale(centre, rot, None)
        optical = rot @ Vector((0.0, -1.0, 0.0))
        p_eye = person.params.eyes[side]
        pupil = centre + optical * (p_eye.iris_depth * MM)
        apex = centre + optical * (p_eye.cornea_apex * MM)
        in_head = head_rot.inverted() @ optical
        pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, in_head.z)))))
        labels["eyes"].append(
            {
                "side": "right" if side == 0 else "left",
                "centre_cam_mm": cam_from_world(centre),
                "sight_cam": cam_from_world(sight * MM),
                "optical_cam": cam_from_world(optical * MM),
                "pupil_cam_mm": cam_from_world(pupil),
                "cornea_apex_cam_mm": cam_from_world(apex),
                "pitch_in_head_deg": pitches[-1],
                "yaw_in_head_deg": math.degrees(math.atan2(in_head.x, -in_head.y)),
            }
        )
    _pose_lids(person, pose, pitches)
    labels["head_matrix_cam"] = [list(cam_from_world(head.to_3x3().col[i] * MM)) for i in range(3)]
    labels["head_forward_cam"] = list(cam_from_world((head_rot @ Vector((0.0, -1.0, 0.0))) * MM))
    labels["head_yaw_pitch_roll_deg"] = [pose.head_yaw_deg, pose.head_pitch_deg, pose.head_roll_deg]
    return labels


def _pose_lids(person: Person, pose: Pose, eye_pitch_deg: list[float]) -> None:
    for side, suffix in enumerate(("R", "L")):
        for bone_name, follow, sign in (
            (f"orbicularis03.{suffix}", pose.upper_lid_follow, 1.0),
            (f"orbicularis04.{suffix}", pose.lower_lid_follow, -1.0),
        ):
            bone = person.rig.pose.bones[bone_name]
            angle = eye_pitch_deg[side] * follow + sign * pose.lid_open
            bone.rotation_mode = "QUATERNION"
            # The lid bones start at the eye joint and point forwards. Turn them about the head's left-right axis.
            axis = bone.bone.matrix_local.to_3x3().inverted() @ Vector((1.0, 0.0, 0.0))
            bone.rotation_quaternion = Quaternion(axis, math.radians(-angle))  # a negative turn about +X lifts the lid


@dataclass
class CameraParams:
    width: int = 1920
    height: int = 1080
    vertical_fov_deg: float = 44.7

    @property
    def focal_px(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.vertical_fov_deg) / 2)

    def project(self, cam_mm) -> tuple[float, float]:
        """Camera-space point to image pixels (x right, y down)."""
        x, y, z = cam_mm
        f = self.focal_px
        return (self.width / 2 + f * x / -z, self.height / 2 - f * y / -z)


def make_camera(c: CameraParams) -> bpy.types.Object:
    scene = bpy.context.scene
    data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)  # look along world +Y, world +Z is up
    data.sensor_fit = "VERTICAL"
    data.angle_y = math.radians(c.vertical_fov_deg)
    data.clip_start, data.clip_end = 0.05, 10.0
    scene.render.resolution_x, scene.render.resolution_y = c.width, c.height
    scene.render.resolution_percentage = 100
    return cam


def set_border(c: CameraParams, x0: int, y0: int, width: int, height: int) -> None:
    """Render only the pixel rectangle with top-left (x0, y0)."""
    r = bpy.context.scene.render
    r.use_border, r.use_crop_to_border = True, True
    r.border_min_x, r.border_max_x = x0 / c.width, (x0 + width) / c.width
    r.border_max_y, r.border_min_y = 1.0 - y0 / c.height, 1.0 - (y0 + height) / c.height
