# MAINTAINER NOTES — whole-board algorithms (NEVER agent-visible)

These are the certified whole-PCB implementations behind recipes/
(level2_ogrid.py, level3_block.py). The agent only gets single-hole
primitives in playbook.md; composing the domain strategy is its job.

## Level 2 — O-grid rings around every hole (standard for boards with holes)

Motivation: gmsh cannot build structured quads around holes automatically
(work items
[1236](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1236),
[1257](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1257)). Slivers of
recombined unstructured meshes cluster exactly there. The fix is a
hand-built transfinite annulus per hole, geo kernel (not OCC — you need
explicit curve tags for the transfinite constraints).

**Geometry per hole** (t = target edge length; all mm; clearance = min
distance from the hole boundary to the outline polygon edges, every
rectangular void, and every other hole):

1. Ring thickness `= clamp(clearance - 1.0 t, 0.5 t, 1.2 t)`. The
   `- 1.0 t` term is critical: the unstructured remainder needs a channel
   at least ~t wide between the ring's outer arc and any neighbouring
   feature, or it squeezes slivers into the channel (measured: SJ min
   0.12 with `0.8 * clearance` rings vs 0.63 with the gap rule).
2. Split the annulus `r .. r+thickness` into 4 quadrant surfaces:
   4 inner circle arcs + 4 outer circle arcs + 4 radial lines
   (`addCircleArc(start, center, end)`), one curve loop per quadrant
   (`[radial_k, outer_arc_k, -radial_k+1, -inner_arc_k]`, negative tag =
   reversed direction).
3. Node counts: `n_arc = max(3, ceil(pi * outer_r / 2 / (0.7 t)))` on BOTH
   the inner and outer arc (same count keeps quadrants balanced). The
   0.7 t arc step (not t!) matters: ring arcs near walls sit inside the
   refined transition zone, and full-t arc spacing mismatches the ~0.5 t
   local sizes there, shearing the first quad layer (measured: worst SJ
   0.46 -> 0.58 across the sweep after densifying the arcs). Radials get
   `n_radial = max(1, round(thickness / mean(inner_step, outer_step)))`
   nodes with `"Progression"` ratio `outer_step / inner_step` clamped to
   [1.0, 2.2]. Transfinite node counts INCLUDE both endpoints, so pass
   `n + 1`.
4. `setTransfiniteSurface(s, "Left", [4 corner points])` +
   `setRecombine(2, s)` per quadrant. Recombined transfinite surfaces
   yield all-quad structured layers regardless of `RecombineAll`.
5. The outer arcs become extra inner curve loops of ONE remainder plane
   surface that also contains the outline loop and the rect-void loops.
6. Mesh the remainder with the full Level-0 recipe (Algorithm 6,
   RecombineAll 1, Blossom, Smoothing 10), **plus** a Distance +
   Threshold background field on the fixed straight curves (outline
   polygon + rectangular voids, NOT the ring arcs): `SizeMin = 0.5 t`
   within `DistMin = 0.2 t` of those curves, growing back to
   `SizeMax = t` at `DistMax = 1.2 t`. Use
   `gmsh.model.mesh.field.setNumber(threshold, "InField", distance)` —
   in gmsh 4.15 `InField` is a scalar option and the `setNumbers` form
   fails silently.

Rectangular cutouts/slots/header stay unstructured (O-grid scope is
holes only); the threshold field above is what keeps pinch channels
between rect voids healthy.

**Pitfalls (all hit here, all fixed in `quadagent.ogrid_mesh`):**

- Ring too fat near walls (`0.8 * clearance` sizing) leaves an
  unmeshable ~0.3 t channel to the outline -> worst quads end up in the
  wall pockets, not at the holes. Size the ring by clearance MINUS a
  full t.
- Refining near the ring arcs (field on ALL curves) mismatches the
  fixed arc node spacing against 0.5 t interior sizes -> sheared first
  layers, min interior angle drops ~10 deg. Keep refinement to the
  straight fixed boundaries. (Conversely, if arcs stay at full-t
  spacing while sitting near refined walls, the same mismatch appears -
  that is why the arc step above is 0.7 t, not t.)
- No refinement at all: pinch channels between rectangular voids (0.7 -
  0.8 t wide) produce SJ ~ 0.17 slivers even with perfect rings.
- `Mesh.MeshSizeMin` should be relaxed (~0.3 t) so the field can act;
  keep `MeshSizeMax = t` and `setSize(points, t)` for the far field.
- Rect voids that TOUCH each other or the outline (shared edge, e.g. a
  slot abutting the header) break the single plane surface ("Impossible
  to recover edge" -> empty remainder). All boards in `boards/` are
  clean; if you meet one, merge the touching voids into a single union
  curve loop (cancel the shared boundary segments) before adding loops.
- Expect ~3-3.5x the baseline element count and h_min ~ 0.25 t (inner
  arc chords ~0.4 t, recombination transitions shorter), so `dt_crit`
  roughly halves; that is the accepted price for raising the worst
  scaled-Jacobian from ~0.32 to ~0.58+ and SJ-mean on every board.


## Level 3 — pure block-structured decomposition (Cubit-style)

Motivation: Levels 0-2 always leave an unstructured recombined remainder,
which means valence-3/5 irregular nodes, a residual triangle population,
and sliver risk wherever the mesher transitions (gmsh still has no
automatic structured quad transition around holes — work items
[1236](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1236),
[1257](https://gitlab.onelab.info/gmsh/gmsh/-/work_items/1257)). Level 3
removes the unstructured mesher entirely: the domain is decomposed by
hand into 4-sided topological patches, every patch gets
`Transfinite Surface` (4 explicit corners) + `Recombine Surface`, and
`Mesh.RecombineAll = 0` — per-surface recombination only, proving the
mesh is explicit end to end. Result: 100% quads, zero triangles,
grid lines flowing continuously across the whole board.

Implemented in `quadagent.blockstructured_mesh` (web-grid multi-block).

### The web-grid algorithm

1. **Hole boxes.** Per hole a square box `[cx-b, cx+b] x [cy-b, cy+b]`,
   `b = r + ring`, `ring = clamp(0.8 * clearance, 0.5 t, 1.2 t)`
   additionally capped so the box stays inside the chamfered outline
   (chamfer diagonal fit: `b <= (cx+cy-c)/2` etc.; the 0.35 t margin
   validation below is the real floor, not 0.5 t — the caps may legally
   pull the ring below 0.5 t). Box edges within 0.75 t of a wall, a
   rect-void edge or another box edge snap EXACTLY onto it — snap
   walls/voids first, then box-box, otherwise two boxes snap onto each
   other's un-snapped edge and both miss the wall (hit here). After
   snapping, boxes are grown toward margin balance (each margin toward
   the largest, validity re-checked, grown edges re-snapped): this keeps
   the quadrant corner wedges near 45 deg (see pitfalls).
2. **Global cuts.** `xs = {0, W} + void x-edges + box x-edges + chamfer
   tangent coords` (a sw chamfer of size c contributes x=c and y=c; se:
   x=W-c, y=c; ne: x=W-c, y=H-c; nw: x=c, y=H-c), same for `ys`, deduped
   and sorted. Cuts span the FULL domain, so every feature is a union of
   whole cells and no T-junctions can exist. **Hole boxes are opaque
   lattice regions**: a foreign cut (e.g. the header's y-edge, or a
   chamfer tangent) that falls inside a box simply terminates on the box
   edge, subdividing it into a chain of collinear lattice segments; the
   quadrant patches absorb the chain as a multi-curve transfinite side
   (multiple curves between two corner points are fine when corners are
   passed explicitly to `setTransfiniteSurface`). Adjacent cuts closer
   than 0.5 t are allowed only when both are feature edges (accept +
   warn); on the dense boards this happens a lot because voids are
   mutually offset by 0.05-0.45 mm — those thin virtual strips are the
   price of full-domain alignment and show up as aspect-ratio spikes.
3. **Cell classification.** Cell = VOID (inside a rect void; header
   counts) / BOX (inside a hole box) / SOLID (one transfinite rectangle
   patch each).
4. **Chamfer absorption.** The corner triangle `{(c,0),(c,c),(0,c)}` (sw
   case) merges with the SOLID cell directly north (column [0,c], first
   row above y=c) into one 4-sided patch: chamfer diagonal, east chain
   on x=c, row-top segment, west wall segment (mirrored for se/ne/nw;
   poleward neighbor first, east/west as fallback; neither SOLID → fail
   the board).
5. **Interval law (the consistency law).** Horizontal segments in column
   band i share one count `H[i] = max(2, round(w_i/t)+1)`; vertical
   segments in row band j share `V[j] = max(2, round(h_j/t)+1)`. Per
   hole: `A_ns = max(ceil(r*pi/2/t), ceil(box_w/t)) + 1` forces the box
   north/south chain totals (and thus the north/south arcs, which must
   equal their box edge), `A_ew` likewise for west/east; connectors get
   `R = max(2, round(max_connector_len/(0.75 t))+1)`. Counts INCLUDE
   both endpoints; opposite-side totals of every patch are asserted
   equal before meshing (registering a Line twice with different counts
   is a hard error — that is the no-T-junction invariant).

### Pitfalls hit here (all in `quadagent.blockstructured_mesh`)

- `setTransfiniteCurve` must be called BEFORE `synchronize()`; applying
  the counts after sync is silently ignored and every quadrant falls
  back to unstructured Frontal-Delaunay with Blossom (16 leaked
  triangles on the first try).
- Transfinite recombination has no parity constraint: odd x odd
  interval counts recombine to pure quads (verified in isolation), so no
  even-count forcing is needed.
- The 90 deg angle at each box corner splits between two quadrant
  patches, so the radial connector construction caps the minimum
  interior angle at 45 deg — measured 45.00 on a symmetric fixture and
  38-42 on real boards (wall snapping makes boxes asymmetric; the
  margin-balanced growth above recovers a few degrees). Do not expect
  Level 3 to beat Level 2 on min angle; it wins scaled Jacobian
  (0.63-0.78 vs 0.32-0.68 baseline), quad fraction (exactly 1.0) and
  node-count determinism instead.
- Sizing `R` by `ring` under-counts: the corner connector length is
  `b*sqrt(2)-r ~ 2*ring`, so a `round(ring/t)+1` count leaves single
  2.2 mm radial edges (aspect ~6-8). Size R by the actual connector
  length at a 0.75 t step.
- A hole box edge (or any cut) falling strictly inside a chamfer corner
  square [0,c]x[0,c] breaks the corner-triangle absorption and the
  lattice cannot stay conformant — the board FAILS validation with a
  clear message. On `boards/` this hits every chamfered board
  (train-dense-01/03/04, train-slots-01/04, holdout-dense-05/07,
  holdout-slots-06): the near-corner hole boxes always intrude because
  the circle itself sits inside the tangent square. **Fall back to
  Level 2 there.**
- Thin virtual strips: full-domain cuts project void edges across open
  material; voids offset by < 0.5 t create 0.05-0.45 mm lattice bands
  (train-dense-00, holdout-slots-05...). Accepted per the cut rule, but
  they cap `h_min` (dt_crit) and spike aspect ratio; the unstructured
  recipes don't align to those edges and keep h_min larger.

### When to fall back to Level 2

- Any Level-3 validation failure (chamfer corner-square intrusion, box
  margin < 0.35 t after snapping, box overlapping a void/box): the
  script exits with `blockstructured: board ...` and the reason; the
  sweep records it as a red failure card and the O-grid result stands.
- Boards whose features are mutually offset by less than ~0.5 t
  everywhere (pathological virtual-strip density): Level 3 still
  meshes 100% quads, but dt_crit and aspect ratio gates read worse
  than Level 2 — judge by the sweep table, not by quad fraction alone.

