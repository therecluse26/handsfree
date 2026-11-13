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
import glasses as glasseslib

MM = 0.001
ASSETS = os.path.expanduser(os.environ.get("HANDSFREE_SYNTH_ASSETS", "~/.cache/handsfree/assets/makehuman/system"))

# MakeHuman draws its eyeball as a 16.4 mm sphere around the eye joint. A real
# eyeball has a 12 mm radius. The anatomical eye is put where its cornea apex
# is at the front of the MakeHuman eye.
MAKEHUMAN_EYE_FRONT_MM = 16.4
LOWER_LASHES_IN_LOWER_HALF = ("eyelashes01", "eyelashes02", "eyelashes03")
"""Lash assets whose texture has the lower lashes in its lower half (measured on the meshes)."""
LID_REGION_MM = 19.0
"""The lids are within this distance of the character's eye centre."""


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
    lash_line: float = 0.5
    """Darkness of the lash line and the lid crease: 0 is the character's thin lashes only, 1 is a strong dark line."""
    lower_lashes: float = 0.5
    """Opacity of the lower lashes against the character's own: real lower lashes are few and thin."""
    eyeball_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the eyeballs in their sockets: (towards the nose, forwards, up). People differ in this."""
    eye_shape: dict = field(default_factory=dict)
    """MakeHuman eye targets and their weights, for example {"eye-height2-incr": 0.5}. Applied to both eyes."""
    glasses: glasseslib.GlassesParams | None = None
    scale: float = 1.08
    """Size of the head. 1.0 is the MakeHuman size (58.6 mm between the eyes); 1.08 gives the adult mean of 63 mm."""


@dataclass
class Person:
    params: PersonParams
    rig: bpy.types.Object
    body: bpy.types.Object
    eyes: tuple[bpy.types.Object, bpy.types.Object]
    eye_rest: tuple[Vector, Vector]
    """Eyeball centres in the rig's own coordinates."""
    socket_rest: tuple[Vector, Vector]
    """The same points without ``eyeball_offset_mm``. The lids and eye corners belong to the socket."""
    pivot: Vector
    """Centre of head rotation in the rig's own coordinates."""
    glasses: bpy.types.Object | None = None
    linings: tuple = ()
    """The socket lining of each eye (see eye.create_socket_lining), with its centre in the rig's own coordinates."""


def build_person(p: PersonParams) -> Person:
    from bl_ext.blender_org.mpfb.services.humanservice import HumanService
    from bl_ext.blender_org.mpfb.services.targetservice import TargetService

    macro = TargetService.get_default_macro_info_dict()
    macro.update({k: v for k, v in p.macro.items() if k != "race"})
    if "race" in p.macro:
        macro["race"] = dict(p.macro["race"])
    body = HumanService.create_human(scale=0.1, macro_detail_dict=macro)
    import bl_ext.blender_org.mpfb as mpfb

    targets = os.path.join(os.path.dirname(mpfb.__file__), "data", "targets", "eyes")
    for name, weight in p.eye_shape.items():  # before the rig, so the eye joints follow the shape
        for side in ("l", "r"):
            TargetService.load_target(body, os.path.join(targets, f"{side}-{name}.target.gz"), weight=weight)
    HumanService.add_builtin_rig(body, "default")
    rig = body.parent
    _stiffen_lid_edges(body, rig)  # before the lashes, which take their weights from the lids
    HumanService.set_character_skin(os.path.join(ASSETS, "skins", p.skin, p.skin + ".mhmat"), body)
    for kind, name in (("Eyebrows", p.eyebrows), ("Eyelashes", p.eyelashes)):
        folder = kind.lower()
        HumanService.add_mhclo_asset(os.path.join(ASSETS, folder, name, name + ".mhclo"), body, asset_type=kind)

    if p.lash_line > 0:
        _darken_lid_edges(body, rig, p.lash_line)
    if p.lower_lashes < 1 and p.eyelashes in LOWER_LASHES_IN_LOWER_HALF:
        _thin_lower_lashes(rig, p.lower_lashes)

    eyes, rest, socket, linings = [], [], [], []
    nasal, forward, up = p.eyeball_offset_mm
    for side, bone, params in (("R", "eye.R", p.eyes[0]), ("L", "eye.L", p.eyes[1])):
        joint = rig.data.bones[bone].head_local.copy()
        # The head is scaled later, the eyeball is not: a larger head does not have larger eyes.
        centre = joint + Vector((0.0, -(MAKEHUMAN_EYE_FRONT_MM - params.cornea_apex / p.scale) * MM, 0.0))
        to_nose = 1.0 if side == "R" else -1.0  # the right eye is at -x
        offset = Vector((nasal * to_nose, -forward, up)) * (MM / p.scale)
        obj = eyelib.create_eye(params, f"eye.{side}")
        eyes.append(obj)
        socket.append(centre)
        rest.append(centre + offset)
        # The lining belongs to the socket, like the lids. The eyeball turns and moves behind its opening.
        linings.append((eyelib.create_socket_lining(params, f"lining.{side}"), joint.copy()))
    mid = (socket[0] + socket[1]) / 2
    pivot = mid + Vector((0.0, 0.09, -0.04))  # about the top of the neck
    glasses = glasseslib.create_glasses(p.glasses) if p.glasses else None
    return Person(p, rig, body, (eyes[0], eyes[1]), (rest[0], rest[1]), (socket[0], socket[1]), pivot, glasses, tuple(linings))


def _thin_lower_lashes(rig, keep: float) -> None:
    """Make the lower lashes more transparent. The character has long, dense lower lashes; most real eyes do not."""
    for obj in rig.children:
        if "eyelash" not in obj.name.lower():
            continue
        for mat in obj.data.materials:
            nt = mat.node_tree
            bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
            links = [link for link in nt.links if link.to_node == bsdf and link.to_socket.name == "Alpha"]
            if not links:
                continue
            hair = links[0].from_socket
            coord = nt.nodes.new("ShaderNodeTexCoord")
            split = nt.nodes.new("ShaderNodeSeparateXYZ")
            nt.links.new(coord.outputs["UV"], split.inputs[0])
            factor = nt.nodes.new("ShaderNodeMapRange")  # keep in the texture's lower half, 1 in its upper half
            factor.clamp = True
            for key, value in (("From Min", 0.49), ("From Max", 0.51), ("To Min", keep), ("To Max", 1.0)):
                factor.inputs[key].default_value = value
            nt.links.new(split.outputs["Y"], factor.inputs["Value"])
            alpha = nt.nodes.new("ShaderNodeMath")
            alpha.operation = "MULTIPLY"
            nt.links.new(hair, alpha.inputs[0])
            nt.links.new(factor.outputs["Result"], alpha.inputs[1])
            nt.links.new(alpha.outputs[0], bsdf.inputs["Alpha"])


def _stiffen_lid_edges(body, rig) -> None:
    """Make the inner edge of each lid move with the lid.

    In the character, the lid bones move the outer skin of a lid fully, but its inner edge
    (next to the eyeball) only by a part: 0.5 for the upper lid, 0.1 to 0.2 for the lower lid.
    So when the eye looked down, the lower lid's lash line went down and its inner edge stayed.
    The edge then covered the lower iris, and the stretched skin between showed as a pale band.
    Measured on 15 real persons: the eye stays open on a down gaze, and the iris stays visible.

    Here each lid vertex that is nearer to the eye opening than a neighbour with more weight
    gets that neighbour's weight. The weights towards the brow, the cheek and the eye corners
    do not change.
    """
    mesh = body.data
    neighbours: dict[int, list[int]] = {}
    for edge in mesh.edges:
        a, b = edge.vertices
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    bones = {bone.name for bone in rig.data.bones}
    for suffix in ("R", "L"):
        groups = [body.vertex_groups.get(f"{name}.{suffix}") for name in ("orbicularis03", "orbicularis04")]
        if None in groups:
            continue
        weights = [{}, {}]
        for v in mesh.vertices:
            for g in v.groups:
                for k, group in enumerate(groups):
                    if g.group == group.index and g.weight > 0.02:
                        weights[k][v.index] = g.weight
        lid = set(weights[0]) | set(weights[1])
        centre = sum((mesh.vertices[i].co for i in lid), Vector()) / len(lid)

        def reach(i: int) -> float:  # distance from the middle of the eye opening, seen from the front
            d = mesh.vertices[i].co - centre
            return math.hypot(d.x, d.z)

        for group, weight in zip(groups, weights):
            new = dict(weight)
            for _ in range(4):  # the edge is some vertices deep
                for i in new:
                    for n in neighbours.get(i, ()):
                        if n in new and new[n] > new[i] + 0.02 and reach(i) < reach(n):
                            new[i] = new[n]
            for i, w in new.items():
                if w <= weight[i] + 0.02:
                    continue
                rest = (1.0 - w) / max(1.0 - weight[i], 1e-6)  # the other bones keep the rest of the vertex
                for g in list(mesh.vertices[i].groups):
                    other = body.vertex_groups[g.group]
                    if other.name in bones and other.index != group.index:
                        other.add([i], g.weight * rest, "REPLACE")
                group.add([i], w, "REPLACE")


def _darken_lid_edges(body, rig, strength: float) -> None:
    """Darken the skin where the lids meet the eyeball, and in the lid crease.

    The character's lashes are single thin hairs. In a camera image they nearly vanish. Real
    lashes are dense at the lid edge and show as a dark line, which is the strongest edge
    of a real eye image after the iris. Ambient occlusion finds the lid edge: it is the skin
    that has the eyeball right next to it.
    """
    joints = [rig.data.bones[name].head_local.copy() for name in ("eye.R", "eye.L")]
    trees = []
    for mat in body.data.materials:  # the skin shader is inside a node group, which the materials share
        if mat is None or mat.node_tree is None:
            continue
        for nt in [mat.node_tree] + [n.node_tree for n in mat.node_tree.nodes if n.type == "GROUP" and n.node_tree]:
            if nt not in trees:
                trees.append(nt)
    for nt in trees:
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        links = [link for link in nt.links if link.to_node == bsdf and link.to_socket.name == "Base Color"] if bsdf else []
        if not links:
            continue
        skin = links[0].from_socket
        coord = nt.nodes.new("ShaderNodeTexCoord")
        nearest = None
        for joint in joints:  # distance from the nearer eye centre
            dist = nt.nodes.new("ShaderNodeVectorMath")
            dist.operation = "DISTANCE"
            dist.inputs[1].default_value = joint
            nt.links.new(coord.outputs["Object"], dist.inputs[0])
            if nearest is None:
                nearest = dist.outputs["Value"]
            else:
                both = nt.nodes.new("ShaderNodeMath")
                both.operation = "MINIMUM"
                nt.links.new(nearest, both.inputs[0])
                nt.links.new(dist.outputs["Value"], both.inputs[1])
                nearest = both.outputs[0]
        near_eye = nt.nodes.new("ShaderNodeMapRange")  # 1 on the lids, 0 on the rest of the face
        near_eye.clamp = True
        for key, value in (("From Min", LID_REGION_MM * MM), ("From Max", (LID_REGION_MM + 4.0) * MM), ("To Min", 1.0), ("To Max", 0.0)):
            near_eye.inputs[key].default_value = value
        nt.links.new(nearest, near_eye.inputs["Value"])
        occlusion = nt.nodes.new("ShaderNodeAmbientOcclusion")
        occlusion.samples = 6
        occlusion.inputs["Distance"].default_value = 1.0 * MM  # short: only the skin right at the eyeball and at the lash roots
        power = nt.nodes.new("ShaderNodeMath")
        power.operation = "POWER"
        power.inputs[1].default_value = 6.0 * strength
        nt.links.new(occlusion.outputs["AO"], power.inputs[0])
        dark = nt.nodes.new("ShaderNodeMix")
        dark.data_type = "RGBA"
        dark.blend_type = "MULTIPLY"
        nt.links.new(near_eye.outputs["Result"], dark.inputs["Factor"])
        nt.links.new(skin, dark.inputs["A"])
        nt.links.new(power.outputs[0], dark.inputs["B"])
        nt.links.new(dark.outputs["Result"], bsdf.inputs["Base Color"])


@dataclass
class Pose:
    head_yaw_deg: float = 0.0
    """Positive: the person turns the face to the image's right."""
    head_pitch_deg: float = 0.0
    """Positive: the face tilts up."""
    head_roll_deg: float = 0.0
    eye_mid_cam_mm: tuple[float, float, float] = (0.0, -100.0, -550.0)
    """Midpoint between the eye sockets, camera space."""
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
    about_pivot = Matrix.Translation(person.pivot) @ rot @ Matrix.Translation(-person.pivot) @ Matrix.Scale(person.params.scale, 4)
    mid = about_pivot @ ((person.socket_rest[0] + person.socket_rest[1]) / 2)
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
    for (lining, centre), pitch in zip(person.linings, pitches):
        # It grows with the head. Its opening follows the eye up and down, as the tissue on a real
        # eyeball does. A fixed opening covered the lower part of the iris when the eye looked down.
        lining.matrix_world = head @ Matrix.Translation(centre) @ Matrix.Rotation(math.radians(LINING_SIGN * pitch), 4, "X")
    _pose_lids(person, pose, pitches)
    if person.glasses is not None:
        person.glasses.matrix_world = head @ _glasses_matrix(person)
    labels["head_matrix_cam"] = [list(cam_from_world(head.to_3x3().col[i] * MM)) for i in range(3)]
    labels["head_forward_cam"] = list(cam_from_world((head_rot @ Vector((0.0, -1.0, 0.0))) * MM))
    # The face tracker's eye point is the midpoint of the eye corners. On these characters it is
    # 10 mm in front of the eyeball centre, 3.6 mm below it and 0.7 mm towards the nose (measured
    # with the MediaPipe tracker on rendered faces). The tracker's corners follow the lids: the
    # point moves 0.065 mm down for each degree that the eye looks down in the head (288 rendered eyes).
    for side, (e, rest) in enumerate(zip(labels["eyes"], person.socket_rest)):
        nasal = 1.0 if side == 0 else -1.0
        offset = head_rot @ Vector((0.7 * nasal * MM, -10.0 * MM, (-3.6 + TRACKER_FOLLOWS_LIDS * e["pitch_in_head_deg"]) * MM))
        e["tracker_centre_cam_mm"] = cam_from_world(head @ rest + offset)
    labels["head_yaw_pitch_roll_deg"] = [pose.head_yaw_deg, pose.head_pitch_deg, pose.head_roll_deg]
    return labels


def _glasses_matrix(person: Person) -> Matrix:
    """Place of the glasses in the rig's own coordinates. The glasses keep their real size on a scaled head."""
    g, scale = person.params.glasses, person.params.scale
    apex = sum((p.cornea_apex for p in person.params.eyes)) / 2
    eyes_mid = (person.socket_rest[0] + person.socket_rest[1]) / 2
    front_of_eyes = eyes_mid + Vector((0.0, -apex * MM / scale, 0.0))
    thickness = person.glasses["lens_thickness_mm"]
    origin = front_of_eyes + Vector((0.0, -(g.vertex_distance + thickness) * MM / scale, g.height_offset * MM / scale))
    return Matrix.Translation(origin) @ Matrix.Rotation(math.radians(g.tilt_deg), 4, "X") @ Matrix.Scale(1.0 / scale, 4)


TRACKER_FOLLOWS_LIDS = 0.065
"""Vertical movement of the face tracker's eye point, mm for each degree of the eye's pitch in the head."""
LINING_SIGN = -1.0
"""Direction of the lining's turn about the head's left-right axis for a positive eye pitch."""


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
    # Blender cuts border * resolution down to a whole pixel. The quarter pixel keeps
    # rounding errors of the division from moving an edge by one pixel.
    r.border_min_x, r.border_max_x = (x0 + 0.25) / c.width, (x0 + width + 0.25) / c.width
    r.border_max_y, r.border_min_y = (c.height - y0 + 0.25) / c.height, (c.height - y0 - height + 0.25) / c.height
