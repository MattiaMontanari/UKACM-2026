"""One offline smoke test for the mesh-quality seam."""

from pathlib import Path

import gmsh

from quadagent import quality


def test_scores_a_gmsh_quad_mesh(tmp_path: Path) -> None:
    mesh = tmp_path / "quad.msh"
    gmsh.initialize()
    try:
        gmsh.model.add("quad")
        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh))
    finally:
        gmsh.finalize()

    report = quality.analyze(mesh, expected_area_mm2=1.0)

    assert set(report) == {
        "area_ratio",
        "expected_area_mm2",
        "mesh_area_mm2",
        "mesh_file",
        "n_elements_2d",
        "n_nodes",
        "n_quads",
        "n_triangles",
        "quad_fraction",
        "scaled_jacobian_min",
    }
    assert report["n_quads"] > 0
    assert report["scaled_jacobian_min"] > 0
    assert report["area_ratio"] == 1.0
