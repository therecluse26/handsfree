"""Procedural glasses for Blender. Runs inside Blender's Python.

The lenses are real lens shapes: a front and a back sphere surface with a
glass material. So the renderer bends the light as a real lens does. A minus
lens makes the eye smaller and moves it when the head turns; the lens also
reflects the screen and the lamps. The frame is a rim around each lens, a
bridge and two temples.

Local frame of the glasses object, in metres: x to the person's left, y
towards the face, z up. The origin is on the midline, at the front of the
lenses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import bmesh
import bpy
from mathutils import Matrix, Vector

MM = 0.001


@dataclass(frozen=True)
class GlassesParams:
    lens_width: float = 50.0
    """Width of one lens, mm."""
    lens_height: float = 36.0
    roundness: float = 3.0
    """Shape of the lens outline: 2 is an ellipse, larger values are more like a rectangle."""
    bridge: float = 17.0
    """Distance between the two lenses, mm."""
    power: float = -3.0
    """Lens power in dioptres. Negative for short-sighted persons."""
    index: float = 1.6
    """Refractive index of the lens material."""
    base_curve: float = 4.0
    """Power of the front surface in dioptres. It sets how much the lens is curved."""
    vertex_distance: float = 13.0
    """From the back of the lens to the front of the eye, mm."""
    height_offset: float = -4.0
    """Lens centre above (+) or below (-) the eyes, mm."""
    tilt_deg: float = 8.0
    """The lower lens edge is nearer to the face by this angle."""
    wrap_deg: float = 3.0
    """Each lens turns towards its temple by this angle."""
    rim: float = 2.5
    """Width of the rim around the lens, mm. 0 gives rimless glasses."""
    rim_depth: float = 3.5
    temple_height: float = 4.0
    frame_colour: tuple[float, float, float] = (0.02, 0.02, 0.02)
    frame_metallic: float = 0.0
    frame_roughness: float = 0.35
    coating: float = 0.6
    """Anti-reflection coating: 0 is plain glass with strong reflections, 1 has no reflections."""
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _sag(radius: float, r: float) -> float:
    """Depth of a sphere surface with ``radius`` at distance ``r`` from its axis (same units)."""
    if not math.isfinite(radius) or radius > 1e6:
        return 0.0
    return radius - math.sqrt(max(radius * radius - r * r, 0.0))


def lens_surfaces(p: GlassesParams) -> tuple[float, float, float]:
    """(front radius, back radius, centre thickness) in mm for the lens power."""
    base = max(p.base_curve, p.power + 2.0)  # the back surface must stay hollow
    r1 = 1000.0 * (p.index - 1.0) / base
    r2 = 1000.0 * (p.index - 1.0) / (base - p.power)
    r_max = math.hypot(p.lens_width, p.lens_height) / 2
    # A plus lens is thick in the centre, a minus lens at the edge. The thinnest place is 1.2 mm.
    centre = 1.2 + max(0.0, _sag(r1, r_max) - _sag(r2, r_max))
    return r1, r2, centre


def _outline(p: GlassesParams, segments: int) -> list[tuple[float, float]]:
    a, b, n = p.lens_width / 2, p.lens_height / 2, p.roundness
    pts = []
    for i in range(segments):
        t = 2 * math.pi * i / segments
        c, s = math.cos(t), math.sin(t)
        pts.append((a * math.copysign(abs(c) ** (2 / n), c), b * math.copysign(abs(s) ** (2 / n), s)))
    return pts


def _add_lens(bm, p: GlassesParams, to_object: Matrix, segments: int = 48, rings: int = 10) -> None:
    r1, r2, centre = lens_surfaces(p)
    outline = _outline(p, segments)

    def ring(k: int, back: bool) -> list:
        verts = []
        for x, z in outline:
            x, z = x * k / rings, z * k / rings
            r = math.hypot(x, z)
            y = centre + _sag(r2, r) if back else _sag(r1, r)
            verts.append(bm.verts.new(to_object @ Vector((x * MM, y * MM, z * MM))))
        return verts

    edge = {}
    for back in (False, True):
        centre_vert = bm.verts.new(to_object @ Vector((0.0, (centre if back else 0.0) * MM, 0.0)))
        prev = None
        for k in range(1, rings + 1):
            cur = ring(k, back)
            for i in range(segments):
                j = (i + 1) % segments
                f = bm.faces.new([centre_vert, cur[i], cur[j]] if prev is None else [prev[i], cur[i], cur[j], prev[j]])
                f.material_index, f.smooth = 1, True
            prev = cur
        edge[back] = prev
    for i in range(segments):  # the ground edge of the lens
        j = (i + 1) % segments
        f = bm.faces.new([edge[False][i], edge[True][i], edge[True][j], edge[False][j]])
        f.material_index = 1
    if p.rim > 0:
        loops = []
        for x, z in outline:
            r = math.hypot(x, z)
            nx, nz = x / r, z / r
            y0 = _sag(r1, r) - 0.4
            y1 = max(centre + _sag(r2, r) + 0.4, y0 + p.rim_depth)
            corners = [(x - 0.3 * nx, y0, z - 0.3 * nz), (x + p.rim * nx, y0, z + p.rim * nz),
                       (x + p.rim * nx, y1, z + p.rim * nz), (x - 0.3 * nx, y1, z - 0.3 * nz)]  # fmt: skip
            loops.append([bm.verts.new(to_object @ (Vector(c) * MM)) for c in corners])
        for i in range(segments):
            j = (i + 1) % segments
            for c in range(4):
                d = (c + 1) % 4
                f = bm.faces.new([loops[i][c], loops[i][d], loops[j][d], loops[j][c]])
                f.material_index, f.smooth = 0, True


def _add_bar(bm, a: Vector, b: Vector, width: float, height: float) -> None:
    """A straight bar from ``a`` to ``b`` (metres) with a rectangular section: ``width`` across, ``height`` in z (mm)."""
    along = (b - a).normalized()
    up = Vector((0.0, 0.0, 1.0))
    side = along.cross(up).normalized()
    up = side.cross(along).normalized()
    ends = []
    for point in (a, b):
        ends.append([bm.verts.new(point + side * (sx * width / 2 * MM) + up * (sz * height / 2 * MM)) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    for c in range(4):
        d = (c + 1) % 4
        bm.faces.new([ends[0][c], ends[0][d], ends[1][d], ends[1][c]]).material_index = 0
    bm.faces.new(ends[0]).material_index = 0
    bm.faces.new(ends[1][::-1]).material_index = 0


def _frame_material(p: GlassesParams):
    mat = bpy.data.materials.new("glasses.frame")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*p.frame_colour, 1.0)
    bsdf.inputs["Metallic"].default_value = p.frame_metallic
    bsdf.inputs["Roughness"].default_value = p.frame_roughness
    return mat


def _lens_material(p: GlassesParams):
    """Glass that bends light. Shadow rays pass straight through, so the lens does not darken the eye."""
    mat = bpy.data.materials.new("glasses.lens")
    mat.use_nodes = True
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = p.index
    glass.inputs["Color"].default_value = (*p.tint, 1.0)
    clear = nt.nodes.new("ShaderNodeBsdfRefraction")  # bends light, reflects nothing: a perfect coating
    clear.inputs["Roughness"].default_value = 0.0
    clear.inputs["IOR"].default_value = p.index
    clear.inputs["Color"].default_value = (*p.tint, 1.0)
    coated = nt.nodes.new("ShaderNodeMixShader")
    coated.inputs["Fac"].default_value = p.coating
    nt.links.new(glass.outputs[0], coated.inputs[1])
    nt.links.new(clear.outputs[0], coated.inputs[2])
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(coated.outputs[0], mix.inputs[1])
    nt.links.new(transparent.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def create_glasses(p: GlassesParams, name: str = "glasses") -> bpy.types.Object:
    bm = bmesh.new()
    centre_x = (p.bridge + p.lens_width) / 2 * MM
    _, _, thickness = lens_surfaces(p)
    for side in (-1.0, 1.0):
        wrap = Matrix.Rotation(math.radians(p.wrap_deg) * side, 4, "Z")  # the outer edge goes towards the face
        _add_lens(bm, p, Matrix.Translation((side * centre_x, 0.0, 0.0)) @ wrap)
        if p.temple_height > 0:
            hinge = Vector((side * (centre_x + (p.lens_width / 2 + p.rim) * MM), 0.004, p.lens_height * 0.2 * MM))
            ear = hinge + Vector((side * 0.012, 0.11, -0.004))
            _add_bar(bm, hinge, ear, max(p.rim, 1.2), p.temple_height)
    half = (p.bridge / 2 + 0.5) * MM
    z = p.lens_height * 0.22 * MM
    _add_bar(bm, Vector((-half, 0.001, z)), Vector((half, 0.001, z)), max(p.rim_depth * 0.6, 1.5), max(p.rim, 1.5))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(_frame_material(p))
    mesh.materials.append(_lens_material(p))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj["lens_thickness_mm"] = thickness
    return obj
