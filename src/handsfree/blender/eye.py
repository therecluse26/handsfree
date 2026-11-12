"""An anatomical eyeball for Blender. Runs inside Blender's Python.

Dimensions are population averages for an adult eye, in millimetres:

    eyeball (sclera) radius            12.0
    cornea radius of curvature          7.8
    visible iris (limbus) radius        5.85   (diameter 11.7)
    cornea apex from the eye centre    13.12   (follows from the three values above)
    iris plane from the eye centre      9.9
    refractive index, cornea + fluid    1.336

The eye object's origin is the eyeball centre, which is also the centre of
rotation. Its optical axis is the local -Y axis (Blender's "front" for a
character that faces -Y). The cornea refracts, so the iris and pupil appear
where a real camera sees them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import bmesh
import bpy

MM = 0.001


@dataclass(frozen=True)
class EyeParams:
    eyeball_radius: float = 12.0
    cornea_radius: float = 7.8
    limbus_radius: float = 5.85
    iris_depth: float = 9.9
    """Iris plane, distance from the eye centre."""
    pupil_radius: float = 2.0
    ior: float = 1.336
    iris_colour: tuple[float, float, float] = (0.20, 0.10, 0.05)
    """Base iris colour, linear RGB. Brown by default."""
    iris_seed: float = 0.0
    sclera_colour: tuple[float, float, float] = (0.80, 0.76, 0.72)
    vein_strength: float = 0.25

    @property
    def limbus_depth(self) -> float:
        """Limbus plane, distance from the eye centre."""
        return math.sqrt(self.eyeball_radius**2 - self.limbus_radius**2)

    @property
    def cornea_centre(self) -> float:
        """Centre of the cornea sphere, distance from the eye centre."""
        return self.limbus_depth - math.sqrt(self.cornea_radius**2 - self.limbus_radius**2)

    @property
    def cornea_apex(self) -> float:
        return self.cornea_centre + self.cornea_radius


def _revolve(bm, profile: list[tuple[float, float]], segments: int, material_index: int, flip: bool = False):
    """Make a surface of revolution about the optical axis from (radius, depth) points in mm.

    Depth is along the optical axis (local -Y). A radius of 0 makes a pole.
    """
    rings = []
    for r, d in profile:
        if r < 1e-9:
            rings.append([bm.verts.new((0.0, -d * MM, 0.0))])
        else:
            rings.append(
                [
                    bm.verts.new((r * MM * math.cos(a), -d * MM, r * MM * math.sin(a)))
                    for a in (2 * math.pi * i / segments for i in range(segments))
                ]
            )
    for a, b in zip(rings, rings[1:]):
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                verts = [a[0], b[i], b[j]]
            elif len(b) == 1:
                verts = [a[i], b[0], a[j]]
            else:
                verts = [a[i], b[i], b[j], a[j]]
            f = bm.faces.new(verts[::-1] if flip else verts)
            f.material_index = material_index
            f.smooth = True


def build_eye_mesh(p: EyeParams, name: str, segments: int = 64) -> bpy.types.Mesh:
    """Mesh with 3 material slots: 0 sclera, 1 iris (with the pupil), 2 cornea."""
    bm = bmesh.new()
    R = p.eyeball_radius
    # Sclera: from the back pole to the limbus.
    a_limbus = math.acos(p.limbus_depth / R)
    n = 40
    sclera = [(R * math.sin(a), R * math.cos(a)) for a in (math.pi - (math.pi - a_limbus) * i / n for i in range(n + 1))]
    _revolve(bm, sclera, segments, 0, flip=True)
    # Iris: a shallow cone from the limbus to the centre, behind the cornea.
    iris = [(p.limbus_radius * (1 - i / 16), p.limbus_depth - (p.limbus_depth - p.iris_depth) * min(1.0, 4 * i / 16)) for i in range(17)]
    _revolve(bm, iris, segments, 1, flip=True)
    # Cornea: a spherical cap from the limbus to the apex.
    a_max = math.asin(p.limbus_radius / p.cornea_radius)
    cornea = [
        (p.cornea_radius * math.sin(a), p.cornea_centre + p.cornea_radius * math.cos(a))
        for a in (a_max * (1 - i / 24) for i in range(25))
    ]
    _revolve(bm, cornea, segments, 2, flip=True)
    bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.material_index != 2])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _principled(mat):
    nodes = mat.node_tree.nodes
    return next(n for n in nodes if n.type == "BSDF_PRINCIPLED"), next(n for n in nodes if n.type == "OUTPUT_MATERIAL")


def _radius_mm(nt):
    """Nodes that give the distance from the optical axis in mm, and the angle around it."""
    geo = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Object"], sep.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
    length = nt.nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    nt.links.new(comb.outputs[0], length.inputs[0])
    mm = nt.nodes.new("ShaderNodeMath")
    mm.operation = "MULTIPLY"
    mm.inputs[1].default_value = 1000.0
    nt.links.new(length.outputs["Value"], mm.inputs[0])
    angle = nt.nodes.new("ShaderNodeMath")
    angle.operation = "ARCTAN2"
    nt.links.new(sep.outputs["Z"], angle.inputs[0])
    nt.links.new(sep.outputs["X"], angle.inputs[1])
    return mm.outputs[0], angle.outputs[0], geo


def _ramp(nt, value, stops):
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, col) in zip(els, stops):
        el.position, el.color = pos, (*col, 1.0)
    nt.links.new(value, ramp.inputs[0])
    return ramp.outputs[0]


def iris_material(p: EyeParams, name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf, _ = _principled(mat)
    r_mm, angle, _ = _radius_mm(nt)
    # Fibres: noise that is long along the radius and short around the circle.
    coord = nt.nodes.new("ShaderNodeCombineXYZ")
    scale_r = nt.nodes.new("ShaderNodeMath")
    scale_r.operation = "MULTIPLY"
    scale_r.inputs[1].default_value = 0.25
    nt.links.new(r_mm, scale_r.inputs[0])
    nt.links.new(scale_r.outputs[0], coord.inputs["X"])
    scale_a = nt.nodes.new("ShaderNodeMath")
    scale_a.operation = "MULTIPLY"
    scale_a.inputs[1].default_value = 9.0
    nt.links.new(angle, scale_a.inputs[0])
    nt.links.new(scale_a.outputs[0], coord.inputs["Y"])
    coord.inputs["Z"].default_value = p.iris_seed
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(coord.outputs[0], noise.inputs["Vector"])
    c = p.iris_colour
    fibres = _ramp(nt, noise.outputs["Fac"], [(0.25, tuple(v * 0.35 for v in c)), (0.75, tuple(min(1.0, v * 1.9 + 0.03) for v in c))])
    # Radial structure: pupil (black), lighter collarette, dark limbal ring.
    rel = nt.nodes.new("ShaderNodeMath")
    rel.operation = "DIVIDE"
    rel.inputs[1].default_value = p.limbus_radius
    nt.links.new(r_mm, rel.inputs[0])
    pr = p.pupil_radius / p.limbus_radius
    radial = _ramp(
        nt,
        rel.outputs[0],
        [(pr, (0, 0, 0)), (min(pr + 0.03, 0.9), (0.55, 0.55, 0.55)), (min(pr + 0.18, 0.92), (1, 1, 1)), (0.86, (0.8, 0.8, 0.8)), (0.99, (0.12, 0.12, 0.12))],
    )
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    nt.links.new(fibres, mix.inputs["A"])
    nt.links.new(radial, mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def sclera_material(p: EyeParams, name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf, _ = _principled(mat)
    r_mm, _, geo = _radius_mm(nt)
    veins = nt.nodes.new("ShaderNodeTexNoise")
    veins.inputs["Scale"].default_value = 260.0
    veins.inputs["Detail"].default_value = 8.0
    veins.inputs["Distortion"].default_value = 2.5
    nt.links.new(geo.outputs["Object"], veins.inputs["Vector"])
    s = p.sclera_colour
    red = (s[0] * 0.95, s[1] * (1 - 0.6 * p.vein_strength), s[2] * (1 - 0.7 * p.vein_strength))
    vein_col = _ramp(nt, veins.outputs["Fac"], [(0.50, s), (0.62, red)])
    # The sclera gets darker towards the limbus, where the cornea starts.
    rel = nt.nodes.new("ShaderNodeMath")
    rel.operation = "DIVIDE"
    rel.inputs[1].default_value = p.limbus_radius
    nt.links.new(r_mm, rel.inputs[0])
    edge = _ramp(nt, rel.outputs[0], [(0.5, (0.30, 0.28, 0.27)), (0.62, (1, 1, 1))])
    edge.node.inputs[0].default_value = 0.0
    scaled = nt.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = 0.5
    nt.links.new(rel.outputs[0], scaled.inputs[0])
    nt.links.new(scaled.outputs[0], edge.node.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    nt.links.new(vein_col, mix.inputs["A"])
    nt.links.new(edge, mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35
    for key, value in (("Coat Weight", 1.0), ("Coat Roughness", 0.04), ("Subsurface Weight", 0.15)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = value
    return mat


def cornea_material(p: EyeParams, name: str):
    """Clear, refracting and glossy. Shadow rays pass straight through, so light reaches the iris."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf, out = _principled(mat)
    for key, value in (("Transmission Weight", 1.0), ("Roughness", 0.0), ("IOR", p.ior)):
        bsdf.inputs[key].default_value = value
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs[0], mix.inputs[1])
    nt.links.new(transparent.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def create_eye(p: EyeParams, name: str) -> bpy.types.Object:
    mesh = build_eye_mesh(p, name)
    for mat in (sclera_material(p, name + ".sclera"), iris_material(p, name + ".iris"), cornea_material(p, name + ".cornea")):
        mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj
