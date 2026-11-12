"""Generate labelled synthetic eye images. Runs inside Blender:

    blender -b -P src/handsfree/blender/generate.py -- --out DIR --people 10 --poses 8 --seed 1

The command `handsfree synth generate` runs this script with the correct Blender profile.

Each sample is one render of the camera-image rectangle that holds both eyes,
in raw camera pixels, plus one JSON line of labels in ``labels-SEED.jsonl``.
All label positions are camera space in millimetres (see scene.py).

The images are clean renders with linear light values (.npz, float16). Exposure,
tone and the camera simulation (handsfree.camsim) are applied later, so one
render can make many training images. With ``--face`` each sample also gets a
render of the full face rectangle, for checks with the face tracker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import eye as eyelib  # noqa: E402
import scene as S  # noqa: E402

EYE_CROP = (192, 128)
"""The eye crop of handsfree recordings: (width, height) in camera pixels."""
MARGIN = 24
"""Extra pixels around the two eye crops, so the crop position can vary later."""

SKINS = [
    f"{age}_{race}_{sex}"
    for age in ("young", "middleage", "old")
    for race in ("african", "asian", "caucasian")
    for sex in ("female", "male")
]
# Iris colours, linear RGB, with their share of people. Brown is the most common eye colour.
IRIS = [
    (0.45, (0.060, 0.025, 0.010)),  # dark brown
    (0.20, (0.160, 0.075, 0.030)),  # brown
    (0.08, (0.220, 0.140, 0.050)),  # hazel
    (0.07, (0.120, 0.170, 0.080)),  # green
    (0.14, (0.130, 0.220, 0.330)),  # blue
    (0.06, (0.200, 0.230, 0.250)),  # grey
]


# MakeHuman eye-shape targets: (name, the two directions). Size and position targets are left
# out, because the anatomical eyeball does not follow them.
EYE_SHAPES = [
    ("eye-height1", ("incr", "decr")),
    ("eye-height2", ("incr", "decr")),
    ("eye-height3", ("incr", "decr")),
    ("eye-eyefold-angle", ("up", "down")),
    ("eye-eyefold", ("concave", "convex")),
    ("eye-eyefold", ("up", "down")),
    ("eye-epicanthus", ("in", "out")),
    ("eye-corner1", ("up", "down")),
    ("eye-corner2", ("up", "down")),
    ("eye-bag", ("incr", "decr")),
]


def random_person(rng: random.Random) -> S.PersonParams:
    race = [rng.gammavariate(0.6, 1.0) for _ in range(3)]
    total = sum(race)
    race = dict(zip(("asian", "caucasian", "african"), (r / total for r in race)))
    gender = rng.random()
    age = rng.uniform(0.4, 1.0)  # MakeHuman: 0.5 is 25 years, 1.0 is 90 years
    main_race = max(race, key=race.get)
    skin = f"{'young' if age < 0.6 else 'middleage' if age < 0.8 else 'old'}_{main_race}_{'female' if gender < 0.5 else 'male'}"
    if rng.random() < 0.15:
        skin = rng.choice(SKINS)

    r, acc = rng.random(), 0.0
    for share, colour in IRIS:
        acc += share
        if r <= acc:
            break
    colour = tuple(min(1.0, max(0.004, c * rng.uniform(0.7, 1.4))) for c in colour)
    shared = dict(
        eyeball_radius=rng.gauss(12.0, 0.35),
        limbus_radius=rng.gauss(5.85, 0.22),
        iris_colour=colour,
        sclera_colour=tuple(c * rng.uniform(0.85, 1.0) for c in (0.80, 0.76, 0.72)),
        vein_strength=rng.uniform(0.05, 0.6),
    )
    pupil = rng.uniform(1.3, 3.6)
    eyes = tuple(
        eyelib.EyeParams(pupil_radius=pupil * rng.uniform(0.95, 1.05), iris_seed=rng.uniform(0, 100), **shared) for _ in range(2)
    )
    shape = {f"{name}-{rng.choice(ways)}": rng.uniform(0.1, 0.7) for name, ways in EYE_SHAPES if rng.random() < 0.5}
    return S.PersonParams(
        eye_shape=shape,
        macro={"gender": gender, "age": age, "muscle": rng.uniform(0.2, 0.8), "weight": rng.uniform(0.2, 0.8), "race": race},
        skin=skin,
        eyebrows=f"eyebrow{rng.randint(1, 12):03d}",
        eyelashes=f"eyelashes{rng.randint(1, 4):02d}",
        eyes=eyes,
        kappa_h_deg=rng.gauss(5.0, 1.5),
        kappa_v_deg=rng.gauss(1.5, 1.0),
        scale=rng.uniform(0.98, 1.16),
        eyeball_offset_mm=(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.5)),
    )


def random_setup(rng: random.Random) -> dict:
    """Camera, screen and the person's position: one device and one way of sitting."""
    cam = S.CameraParams(vertical_fov_deg=rng.uniform(36.0, 70.0))
    width = rng.uniform(280.0, 620.0)
    screen = {"width_mm": width, "height_mm": width / rng.choice((16 / 10, 16 / 9, 3 / 2)), "camera_above_mm": rng.uniform(4.0, 20.0)}
    return {"camera": cam, "screen": screen}


def random_pose(rng: random.Random, setup: dict) -> tuple[S.Pose, dict]:
    cam, screen = setup["camera"], setup["screen"]
    z = -rng.uniform(350.0, 900.0)
    # Put the face inside the camera image.
    u, v = rng.uniform(0.2, 0.8) * cam.width, rng.uniform(0.2, 0.9) * cam.height
    x, y = (u - cam.width / 2) * -z / cam.focal_px, -(v - cam.height / 2) * -z / cam.focal_px
    # A target on the screen, sometimes a little outside it.
    sx = rng.uniform(-0.25, 1.25) * screen["width_mm"]
    sy = rng.uniform(-0.25, 1.25) * screen["height_mm"]
    target = (screen["width_mm"] / 2 - sx, -(screen["camera_above_mm"] + sy), 0.0)
    # The head turns part of the way to the target, plus its own variation.
    to_target = Vector(target) - Vector((x, y, z))
    yaw_t = math.degrees(math.atan2(to_target.x, to_target.z))
    pitch_t = math.degrees(math.atan2(to_target.y, math.hypot(to_target.x, to_target.z)))
    follow = rng.uniform(0.2, 0.9)
    pose = S.Pose(
        head_yaw_deg=yaw_t * follow + rng.gauss(0, 9),
        head_pitch_deg=pitch_t * follow + rng.gauss(-3, 8),  # a laptop camera looks up at the face
        head_roll_deg=rng.gauss(0, 5),
        eye_mid_cam_mm=(x, y, z),
        target_cam_mm=target,
        # Measured with the face tracker: real lids are wider open than the MakeHuman rest
        # shape (0.31 against 0.24 eye widths) and close less on a down gaze.
        upper_lid_follow=rng.uniform(0.65, 0.9),
        lower_lid_follow=rng.uniform(0.3, 0.55),
        lid_open=rng.gauss(3.5, 4.0) - (rng.uniform(4, 10) if rng.random() < 0.1 else 0),
    )
    return pose, {"screen_xy_mm": [sx, sy]}


def random_light(rng: random.Random, setup: dict) -> dict:
    scene = bpy.context.scene
    for obj in [o for o in scene.objects if o.type == "LIGHT" or o.name.startswith("screen")]:
        bpy.data.objects.remove(obj, do_unlink=True)
    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    ambient = 10 ** rng.uniform(-1.6, -0.3)
    tint = [rng.uniform(0.8, 1.0), rng.uniform(0.85, 1.0), rng.uniform(0.75, 1.0)]
    bg.inputs[0].default_value = (*tint, 1.0)
    bg.inputs[1].default_value = ambient
    info = {"ambient": ambient, "lamps": []}
    for _ in range(rng.choice((0, 1, 1, 2, 3))):
        data = bpy.data.lights.new("lamp", "AREA")
        data.size = rng.uniform(0.1, 1.2)
        data.energy = 10 ** rng.uniform(0.3, 1.8)
        data.color = (1.0, rng.uniform(0.75, 1.0), rng.uniform(0.5, 1.0))
        lamp = bpy.data.objects.new("lamp", data)
        scene.collection.objects.link(lamp)
        az, el, dist = rng.uniform(-110, 110), rng.uniform(-10, 75), rng.uniform(0.6, 2.5)
        lamp.location = (dist * math.sin(math.radians(az)) * math.cos(math.radians(el)), 0.6 - dist * math.cos(math.radians(az)) * math.cos(math.radians(el)), dist * math.sin(math.radians(el)))
        lamp.rotation_euler = (Vector((0, 0.6, -0.1)) - lamp.location).to_track_quat("-Z", "Y").to_euler()
        info["lamps"].append({"azimuth": az, "elevation": el, "energy": data.energy})
    # The screen lights the face and shows as a reflection in the cornea.
    scr = setup["screen"]
    data = bpy.data.lights.new("screen", "AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y = scr["width_mm"] * S.MM, scr["height_mm"] * S.MM
    data.energy = 10 ** rng.uniform(-0.3, 0.9)
    data.color = (rng.uniform(0.7, 1.0), rng.uniform(0.8, 1.0), 1.0)
    glow = bpy.data.objects.new("screen", data)
    scene.collection.objects.link(glow)
    glow.location = S.world_from_cam(0.0, -(scr["camera_above_mm"] + scr["height_mm"] / 2), 0.0)
    glow.rotation_euler = (math.radians(-90.0), 0.0, 0.0)  # emit towards +Y, where the person sits
    info["screen_energy"] = data.energy
    return info


def render_linear(path: str) -> None:
    """Render the current border and save it as ``path`` (.npz): key "rgb", float16, linear light, top row first.

    Blender writes OpenEXR; OpenCV wheels often cannot read it. So the EXR is converted here.
    """
    import numpy as np

    scene = bpy.context.scene
    tmp = path + ".exr"
    scene.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(tmp)
    w, h = img.size
    px = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    os.remove(tmp)
    rgb = px.reshape(h, w, -1)[::-1, :, :3]
    np.savez_compressed(path, rgb=rgb.astype(np.float16))


def eye_rectangle(cam: S.CameraParams, labels: dict) -> tuple[int, int, int, int, list]:
    """The camera-image rectangle (x0, y0, width, height) that holds both eye crops, and the crop centres."""
    centres = []
    for e in labels["eyes"]:
        centres.append(cam.project(e["tracker_centre_cam_mm"]))  # where the face tracker puts the eye
    xs, ys = [c[0] for c in centres], [c[1] for c in centres]
    x0 = int(min(xs)) - EYE_CROP[0] // 2 - MARGIN
    y0 = int(min(ys)) - EYE_CROP[1] // 2 - MARGIN
    x1 = int(max(xs)) + EYE_CROP[0] // 2 + MARGIN
    y1 = int(max(ys)) + EYE_CROP[1] // 2 + MARGIN
    x0, y0 = max(x0 - x0 % 16, 0), max(y0 - y0 % 8, 0)  # start on a JPEG block border
    return x0, y0, min(x1, cam.width) - x0, min(y1, cam.height) - y0, centres


def face_rectangle(cam: S.CameraParams, labels: dict) -> tuple[int, int, int, int]:
    """The camera-image rectangle (x0, y0, width, height) that holds the face."""
    mid = [(a + b) / 2 for a, b in zip(labels["eyes"][0]["centre_cam_mm"], labels["eyes"][1]["centre_cam_mm"])]
    u, v = cam.project(mid)
    k = cam.focal_px / -mid[2]  # pixels per millimetre at the face
    x0, x1 = max(int(u - 130 * k), 0), min(int(u + 130 * k), cam.width)
    y0, y1 = max(int(v - 140 * k), 0), min(int(v + 160 * k), cam.height)
    return x0, y0, x1 - x0, y1 - y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=12, help="Cycles samples per pixel")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for this Blender process")
    ap.add_argument("--face", action="store_true", help="also render the face rectangle of each sample")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    done = 0
    with open(os.path.join(args.out, f"labels-{args.seed:04d}.jsonl"), "a") as out:
        for pi in range(args.people):
            rng = random.Random(args.seed * 100003 + pi)
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.cycles.samples = args.samples
            scene.cycles.use_denoising = True
            # Few light bounces are enough for a face. The cornea needs several transmission bounces.
            scene.cycles.max_bounces, scene.cycles.diffuse_bounces = 4, 2
            scene.cycles.glossy_bounces, scene.cycles.transmission_bounces = 2, 6
            scene.render.threads_mode = "FIXED"
            scene.render.threads = args.threads
            scene.render.image_settings.file_format = "OPEN_EXR"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.color_depth = "16"
            scene.render.image_settings.exr_codec = "ZIP"
            params = random_person(rng)
            person = S.build_person(params)
            setup = random_setup(rng)
            S.make_camera(setup["camera"])
            for qi in range(args.poses):
                pose, extra = random_pose(rng, setup)
                labels = S.apply_pose(person, pose)
                light = random_light(rng, setup)
                x0, y0, w, h, centres = eye_rectangle(setup["camera"], labels)
                if w < EYE_CROP[0] or h < EYE_CROP[1]:
                    continue
                S.set_border(setup["camera"], x0, y0, w, h)
                name = f"s{args.seed:04d}_p{pi:04d}_{qi:02d}.npz"
                render_linear(os.path.join(args.out, name))
                cam = setup["camera"]
                if args.face:
                    fx, fy, fw, fh = face_rectangle(cam, labels)
                    S.set_border(cam, fx, fy, fw, fh)
                    render_linear(os.path.join(args.out, name.replace(".npz", "_face.npz")))
                    labels.update(face_image=name.replace(".npz", "_face.npz"), face_rect=[fx, fy, fw, fh])
                labels.update(
                    image=name,
                    rect=[x0, y0, w, h],
                    crop_centres_px=centres,
                    camera={"width": cam.width, "height": cam.height, "vertical_fov_deg": cam.vertical_fov_deg, "focal_px": cam.focal_px},
                    screen=setup["screen"],
                    light=light,
                    person={
                        "macro": params.macro,
                        "skin": params.skin,
                        "kappa_deg": [params.kappa_h_deg, params.kappa_v_deg],
                        "scale": params.scale,
                        "eye_shape": params.eye_shape,
                        "eyeball_offset_mm": params.eyeball_offset_mm,
                        "eyes": [vars(e) for e in params.eyes],
                    },
                    lids={"upper_follow": pose.upper_lid_follow, "lower_follow": pose.lower_lid_follow, "open": pose.lid_open},
                    **extra,
                )
                out.write(json.dumps(labels) + "\n")
                out.flush()
                done += 1
    print(f"generated {done} samples in {time.time() - t0:.0f} s: {args.out}")


if __name__ == "__main__":
    main()
