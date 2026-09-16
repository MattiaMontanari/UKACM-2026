"""Quality metrics against meshes with known analytic values."""

from __future__ import annotations

from pathlib import Path

import gmsh
import pytest

from quadagent import quality


@pytest.fixture
def two_square_mesh(tmp_path: Path) -> Path:
    """A 2x1 rectangle built as two transfinite-recombined unit-square quads."""
    gmsh.initialize()
    try:
        gmsh.model.add("squares")
        geo = gmsh.model.geo
        a = geo.addPoint(0.0, 0.0, 0.0)
        b = geo.addPoint(1.0, 0.0, 0.0)
        c = geo.addPoint(2.0, 0.0, 0.0)
        d = geo.addPoint(2.0, 1.0, 0.0)
        e = geo.addPoint(1.0, 1.0, 0.0)
        f = geo.addPoint(0.0, 1.0, 0.0)
        ab = geo.addLine(a, b)
        be = geo.addLine(b, e)
        ef = geo.addLine(e, f)
        fa = geo.addLine(f, a)
        bc = geo.addLine(b, c)
        cd = geo.addLine(c, d)
        de = geo.addLine(d, e)
        bottom = geo.addPlaneSurface([geo.addCurveLoop([ab, be, ef, fa])])
        top = geo.addPlaneSurface([geo.addCurveLoop([bc, cd, de, -be])])
        for line in (ab, be, ef, fa, bc, cd, de):
            geo.mesh.setTransfiniteCurve(line, 2)
        geo.mesh.setTransfiniteSurface(bottom)
        geo.mesh.setTransfiniteSurface(top)
        geo.mesh.setRecombine(2, bottom)
        geo.mesh.setRecombine(2, top)
        geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(tmp_path / "squares.msh"))
    finally:
        gmsh.clear()
        gmsh.finalize()
    return tmp_path / "squares.msh"


def test_perfect_quads(two_square_mesh: Path) -> None:
    report = quality.analyze(
        two_square_mesh, expected_area_mm2=2.0, wave_speed_m_s=3000.0
    )
    assert report["n_quads"] == 2
    assert report["n_triangles"] == 0
    assert report["quad_fraction"] == 1.0
    assert report["scaled_jacobian_min"] == pytest.approx(1.0, abs=1e-6)
    assert report["aspect_ratio_max"] == pytest.approx(1.0, abs=1e-6)
    assert report["min_angle_deg"] == pytest.approx(90.0, abs=1e-4)
    assert report["edge_len_min_mm"] == pytest.approx(1.0)
    assert report["mesh_area_mm2"] == pytest.approx(2.0)
    assert report["area_ratio"] == pytest.approx(1.0)
    assert report["dt_critical_us"] == pytest.approx(1.0e-3 / 3000.0 * 1e6, abs=1e-4)


def test_summary_contains_gates(two_square_mesh: Path) -> None:
    text = quality.summarize(quality.analyze(two_square_mesh))
    assert "mesh quality report" in text
    assert "[PASS] quad_fraction = 1.0" in text
    assert "[PASS] scaled_jacobian_min" in text
    assert "[PASS] element_count" in text


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        quality.analyze("/nonexistent/mesh.msh")
