# Gmsh Python API cheat sheet — 2D quad meshing for explicit FEA

Curated reference for the quad meshing agent. Everything runs headless
(`gmsh.fltk.run()` is NOT available in the sandbox). Units: millimetres.

## Session lifecycle

```python
import gmsh
gmsh.initialize()          # start (argv=[] to avoid option parsing)
try:
    gmsh.model.add("board")
    # ... geometry + mesh ...
    gmsh.write("mesh.msh") # extension picks the format
finally:
    gmsh.finalize()        # ALWAYS release the licence/state
```

Gmsh is a global singleton: one model at a time, `finalize()` before
starting a new session in the same process.

## Geometry — two kernels

Built-in `geo` kernel (exact, best for transfinite structure):

```python
gmsh.model.geo.addPoint(x, y, 0) -> tag
gmsh.model.geo.addLine(p1, p2) -> tag
gmsh.model.geo.addCircleArc(pStart, pCenter, pEnd) -> tag
gmsh.model.geo.addCurveLoop([l1, l2, -l3]) -> tag     # negative = reversed
gmsh.model.geo.addPlaneSurface([outerLoop, holeLoop1, ...]) -> tag
gmsh.model.geo.synchronize()                          # REQUIRED before meshing
```

OCC kernel (boolean operations, more robust for many voids):

```python
gmsh.model.occ.addRectangle(x, y, z, dx, dy) -> tag
gmsh.model.occ.addDisk(x, y, z, rx, ry) -> tag
out, _ = gmsh.model.occ.cut([(2, board)], [(2, hole), ...], removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()
```

Rectangular outline with corner chamfers: draw the polygon with
`addPoint`/`addLine` (a chamfer replaces a corner with a 45-degree edge);
circles for holes; rectangles for cut-outs/slots/header. Voids become extra
curve loops (geo) or `cut` tools (occ).

## Element size control

```python
gmsh.option.setNumber("Mesh.MeshSizeMin", t)
gmsh.option.setNumber("Mesh.MeshSizeMax", t)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), t)   # all points to size t
# distance-based gradation near holes:
f = gmsh.model.mesh.field.add("Distance"); ... setNumbers("CurvesList", [...])
th = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumbers(th, "InField", [f])
gmsh.model.mesh.field.setNumber(th, "SizeMin", t/2); ... ("SizeMax", t); ("DistMin", ...)
gmsh.model.mesh.field.setAsBackgroundMesh(th)
```

## Getting quads (the core options)

```python
gmsh.option.setNumber("Mesh.Algorithm", 6)              # 6 = Frontal-Delaunay (good quads after recombine)
gmsh.option.setNumber("Mesh.RecombineAll", 1)           # recombine everything
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1) # 1 = Blossom (best general choice)
gmsh.option.setNumber("Mesh.Smoothing", 10)             # Laplacian smoothing passes
# per-surface alternative to RecombineAll:
gmsh.model.geo.mesh.setRecombine(2, surfaceTag, bump=0.5)
```

Structured quads via transfinite (per surface):

```python
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag, nNodes)              # nNodes INCLUDING endpoints
gmsh.model.geo.mesh.setTransfiniteCurve(lineTag, nNodes, "Progression", 1.2)
gmsh.model.geo.mesh.setTransfiniteSurface(surf, "Left", [cornerP1, cornerP2, cornerP3, cornerP4])
gmsh.model.geo.mesh.setRecombine(2, surf)
```

O-grid around a circular hole of radius r (the classic explicit-FEA recipe):
split the annulus `r .. r+ring` into 4 quadrant surfaces (circle arcs +
radial lines), transfinite each (same node count on inner/outer arcs,
`ceil(pi*r/2/t)` nodes per inner quadrant arc), recombine each. Seed size
on the remaining unstructured region at the board target size.

## Generate, order, export

```python
gmsh.model.mesh.generate(2)      # 2D mesh
gmsh.model.mesh.setOrder(1)      # first-order: 3-node tris / 4-node quads (explicit FEA wants 4-node quads)
gmsh.model.addPhysicalGroup(2, [s], name="material")
gmsh.write("mesh.msh")           # MSH 4.1
gmsh.write("mesh.vtk")           # legacy VTK, quads become VTK_QUAD (9)
```

`.vtk` export works straight from the extension — mesh the same model once,
then write both files.

## Reading mesh data back (host-side quality tooling)

```python
gmsh.open("mesh.msh")
nodeTags, coords, _ = gmsh.model.mesh.getNodes()          # coords flat (x,y,z,...)
etypes, etags, enodes = gmsh.model.mesh.getElements(2)    # dim 2 elements only
gmsh.model.mesh.getElementQualities(tags, "minSICN")      # signed inverse condition number
```

Element type ids: 2 = 3-node triangle, 3 = 4-node quad.

## Common pitfalls

- Forgetting `synchronize()` after geo/occ construction -> empty mesh.
- `RecombineAll` without `Algorithm 6` gives poor quad layers.
- Triangles survive recombination when surfaces are too skinny: check the
  quad fraction and refine locally instead of globally.
- Gmsh silently succeeds on empty surface lists; always assert element
  counts > 0 before writing.
- Node count for `setTransfiniteCurve` includes both endpoints.
- All boards are in the x-y plane at z=0; keep first-order elements.
