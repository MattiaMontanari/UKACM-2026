# Quad meshing playbook — PCB boards for explicit FEA

The tedious task this agent addresses: Gmsh has no automatic way to produce
structured, well-shaped quads around holes and slots (the O-grid problem;
see gmsh work items
[1236](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1236) and
[1257](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1257)). Humans do
it by hand, per hole, with transfinite annuli. **You** will first deliver a
solid *mostly-quad* baseline (Level 0-1 below). For boards with circular
holes, Level 2 (O-grid rings) is the **standard goal** once the baseline is
green: it removes the sliver population that recombined unstructured meshes
leave at hole boundaries.

## Definitions

- Board = rectangle (maybe corner chamfers) MINUS circular holes MINUS
  axis-aligned rectangular cut-outs/slots/header. All dimensions mm, z=0.
- Explicit FEA (drop/shock) wants: 4-node quads, uniform element size,
  no slivers. Small `h_min` punishes the stable time step
  (`dt ~ h_min / c`, FR-4 wave speed c ~ 3000 m/s).

## Level 0 — baseline mesh (always do this first)

1. Parse `board.json`; compute the analytic area (outline minus voids) as
   a sanity reference.
2. Build the domain (OCC `cut` is fine, or geo kernel curve loops).
3. Uniform target size `t` everywhere (`Mesh.MeshSizeMin/Max` + `setSize`
   on all points).
4. Quad recipe:
   - `Mesh.Algorithm = 6` (Frontal-Delaunay)
   - `Mesh.RecombineAll = 1`
   - `Mesh.RecombinationAlgorithm = 1` (Blossom)
   - `Mesh.Smoothing = 10`
5. `generate(2)`, `setOrder(1)`, write `mesh.msh` AND `mesh.vtk`.
6. Render `mesh.png` (matplotlib, Agg): draw quad edges in light gray,
   remaining triangles in red so they are immediately visible.
7. Call `mesh_quality` and compare against the gates below.

## Level 1 — targeted fixes (iterate, one change at a time)

- Triangle survivors clustered at holes/ slots -> local refinement:
  Distance+Threshold background field (`SizeMin = t/2`, `DistMin ~ 0.5 t`,
  `SizeMax = t`) on the void curves instead of global refinement.
- Slivers / low scaled Jacobian -> raise `Mesh.Smoothing`, or split the
  trouble spot with a size bump on nearby points.
- Aspect ratio spikes on thin slots -> element size along the slot edges
  comparable to the slot width.
- Never shrink the whole mesh to fix a local defect: that wastes elements
  and cuts `h_min` (and the time step) everywhere.

## Level 2 — O-grid primitive (single hole) — compose it yourself

You get the PRIMITIVE for one hole; composing the whole board (which holes
get rings, how rings interact with walls/voids/each other, what fills the
remainder) is YOUR design decision and must be reasoned per board.

**One transfinite quadrant annulus** (geo kernel; t = target edge, mm):
- clearance = min distance from the hole boundary to the outline polygon
  edges (chamfer diagonals included), every rectangular void, and every
  other hole
- ring thickness = clamp(0.5 t .. 1.2 t), capped at 0.8 * clearance
- inner arc nodes per quadrant = ceil(pi * r / 2 / (0.7..1.0 t));
  inner and outer arcs of a quadrant share the SAME node count
- radial rows = round(thickness / mean tangential step); connect box
  corners to the circle and setTransfiniteSurface + setRecombine per
  quadrant (4 four-sided patches per hole)
- opposite sides of a transfinite patch MUST have equal node counts —
  counts propagate across shared edges (see the Level-3 ground rules)

If clearance < t the ring floor collapses the channel — decide: shrink the
ring, skip it for that hole, or refine locally with a Distance+Threshold
field. Measure, don't guess.

## Level 3 — block decomposition primitive (Cubit-style) — invent your domain split

Target: ZERO unstructured surfaces — every region a 4-sided transfinite +
recombined patch, grid lines flowing continuously (no T-junctions).

Primitive you are given: a hole inside a box decomposes into exactly 4
four-sided patches (each: one box edge, 2 corner-to-circle connectors, 1
arc quarter). A rectangle of any aspect ratio is itself one valid patch.
That is ALL you are given.

The whole-domain decomposition is the open problem (gmsh work items
1236/1257) — invent, test, and justify your own scheme. Lattice ground
rules (constraints, not a recipe):
- every patch: exactly 4 sides, one curve loop per Plane Surface,
  Transfinite Surface + Recombine Surface; shared edges are single Line
  entities reused by both neighbours
- run cuts wall-to-wall (full span): a cut that stops mid-material
  creates a T-junction
- make every void a union of whole cells (extend each void edge into a
  full cut; never terminate a cut on a void boundary)
- assign node counts per band: all collinear segments in one column or
  row band share a single count (any sizing law you like) — opposite
  sides then match by construction; assert it before meshing
- collinear curves between two corner points are ONE transfinite side
  when the corners are passed explicitly; absorb chamfer corner
  triangles into a neighbouring cell so every patch stays 4-sided
- validate with mesh_quality: quad fraction must be exactly 1.0; it
  reports worst-element positions and flags odd-parity surfaces (they
  cannot fully recombine)

## Quality gates (explicit FEA, report all)

| metric             | target    | acceptable | hard fail |
|--------------------|-----------|------------|-----------|
| quad fraction      | >= 0.98   | >= 0.90    | < 0.80    |
| scaled Jacobian min| >= 0.60   | >= 0.40    | <= 0      |
| max aspect ratio   | <= 3      | <= 5       | > 10      |
| min interior angle | >= 45 deg | >= 35 deg  | < 20 deg  |
| element count      | 1k-10k    | <= 20k     | > 50k     |

Deliverables per run: `mesh.vtk` + `mesh.png` + a short report quoting the
measured metrics. Leave the last working script as `mesh.py` in the
workspace.
