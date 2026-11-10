"""Browser entry points reject invalid scalar and lossy integer protocols."""
import json

import numpy as np
import pytest

from dichromatic_map.cells import StrainedCell
from dichromatic_map.crystal import ProjectedGrain, csl_presets
from dichromatic_map.matching import exact_csl_cell
from .test_web_state_contract import bridge


def vector_request(repeat):
    return {"atoms":[{"position":[0,0],"grain":0,"layer":0,"half_indices":[0,0,0]},
                     {"position":[1,0],"grain":0,"layer":0,"half_indices":[2,0,0]}],
            "lattice":"FCC","axis":"110","angle":22,
            "deformations":[np.eye(2).tolist(),np.eye(2).tolist()],"axial_repeat":repeat}


@pytest.mark.parametrize("repeat", [1.5, -4.5, -5, 5, True, None, "1", "", float("nan"), float("inf")])
def test_vector_rejects_invalid_axial_repeat_without_truncation(bridge, repeat):
    with pytest.raises(ValueError, match="Axial repeat"):
        bridge._vector(vector_request(repeat))


@pytest.mark.parametrize("repeat", [-4, 0, 4])
def test_vector_accepts_bounded_integer_axial_repeat(bridge, repeat):
    result = bridge._vector(vector_request(repeat))
    assert result["displacement"][2] == pytest.approx((repeat+1)*np.sqrt(2)/2)


def test_count_exposes_unique_edges_corners_and_availability(bridge):
    angle=next(p.angle_deg for p in csl_presets("110") if p.name=="Σ9")
    cell=exact_csl_cell(angle)
    polygon=(np.array([[0,0],[1,0],[1,1],[0,1]])@cell.cell.T).tolist()
    request={"polygons":[polygon,polygon],"angle":angle,"lattice":"FCC","axis":"110",
        "deformations":[np.eye(2).tolist(),np.eye(2).tolist()],"translations":[[0,0],[0,0]],"layer":0}
    result=bridge._count(request)
    assert result["half_open_available"] == [True,True]
    for grain in (0,1):
        assert result["half_open"][grain][0] == 9
        assert result["interior"][grain][0]+result["boundary"][grain][0] == 12
        assert result["interior"][grain][0]+result["half_open_edges"][grain][0]+result["half_open_corners"][grain][0] == 9
    request["polygons"][1]=[[0,0],[2,0],[1,1],[0,1]]
    irregular=bridge._count(request)
    assert irregular["half_open_available"] == [True,False]


@pytest.mark.parametrize("action", ["fit_selected","complete"])
def test_large_exact_integer_counts_are_explicitly_rejected_by_browser_protocol(bridge, action):
    uv=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=float)*3_100_000_000
    if action == "complete":uv=uv[:3]
    request={"polygons":[uv.tolist(),uv.tolist()],"angle":0,"percent":2,
        "rotation":0,"lattice":"SC","axis":"100","layer":0}
    with pytest.raises(ValueError,match="safe integer"):
        bridge._ACTIONS[action](request)


def test_cell_protocol_preserves_the_largest_safe_integer(bridge):
    limit=2**53-1
    matrix=np.diag([limit,1]).astype(np.int64)
    cell=StrainedCell(matrix,matrix,np.eye(2),np.eye(2),matrix.astype(float),0,"SC","100")
    encoded=bridge._cell(cell)
    assert encoded["m1"][0][0] == limit
    assert encoded["atoms"] == [limit,limit]


def test_transfer_buffers_reject_atom_indices_that_float64_cannot_preserve(bridge, monkeypatch):
    grain=ProjectedGrain(np.array([[0.,0.]]),np.array([0]),
        np.array([[2**53+1,0,0]],dtype=np.int64),1)
    monkeypatch.setattr(bridge,"projected_columns",lambda *args,**kwargs:grain)
    with pytest.raises(ValueError,match="safe integer"):
        bridge.web_render('{"angle":22,"width":2,"height":2,"lattice":"SC","axis":"100"}')


def test_cell_restore_uses_explicit_int64_for_safe_values_above_int32(bridge):
    matrix=np.diag([3_100_000_000,1]).astype(np.int64)
    cell=StrainedCell(matrix,matrix,np.eye(2),np.eye(2),matrix.astype(float),0,"SC","100")
    restored=bridge._from_cell(bridge._cell(cell),"SC","100")
    assert restored.m1.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(restored.m1,matrix)
    assert restored.atoms == (3_100_000_000,3_100_000_000)


def test_cell_restore_rejects_fractional_and_unsafe_identity_values(bridge):
    cell={"m1":[[1.5,0],[0,1]],"m2":[[1,0],[0,1]],"f1":np.eye(2).tolist(),
        "f2":np.eye(2).tolist(),"cell":np.eye(2).tolist(),"max_strain":0}
    with pytest.raises(ValueError,match="integers"):
        bridge._from_cell(cell,"SC","100")
    cell["m1"][0][0]=2**53+1
    with pytest.raises(ValueError,match="safe integer"):
        bridge._from_cell(cell,"SC","100")


def test_vector_reports_shared_readout_lines_and_in_plane_components(bridge):
    request = vector_request(0)
    request["atoms"][1] = {"position":[1,.5],"grain":0,"layer":1,"half_indices":[1,0,1]}
    result = json.loads(bridge.web_dispatch(json.dumps({"action":"vector", **request})))
    assert result["split"] is True and result["cross_grain"] is False
    assert result["displacement"][2] == pytest.approx(np.sqrt(2)/4)
    assert result["in_plane_length"] == pytest.approx(np.hypot(1, .5))
    assert [line["grain"] for line in result["lines"]] == [None, 0, 0, None]
    texts = [line["text"] for line in result["lines"]]
    assert texts[0] == "Crystal P1→P2 · G1-A → G1-B · axial +0"
    assert texts[1].startswith("G1 current (polar) · full: ")
    assert texts[2].startswith("G1 current (polar) · in-plane: ")
    assert texts[3] == "Current |Δr|/a₀ = 1.1726 · in-plane |Δxy|/a₀ = 1.1180 · Δz/a₀ = +0.3536"
    grain = result["vectors"]["G1"]
    axis = np.array([1, 1, 0]) / np.sqrt(2)
    full = np.array(grain["current"])
    np.testing.assert_allclose(grain["in_plane"]["current"], full - (full @ axis) * axis, atol=1e-12)
    assert grain["in_plane"]["current_formatted"] in texts[2]
    assert grain["current_formatted"] in texts[1]
    # The browser appends the displayed x/y line itself; the bridge never adds it.
    assert not any(text.startswith("View") for text in texts)


def test_cross_grain_vector_tags_lines_with_both_grains(bridge):
    request = vector_request(0)
    request["atoms"][1] = {"position":[1,0],"grain":1,"layer":0,"half_indices":[1,-1,0]}
    result = bridge._vector(request)
    assert result["cross_grain"] is True and result["split"] is False
    assert [line["grain"] for line in result["lines"]] == [None, 0, 1, None]
    assert result["lines"][0]["text"] == "Cross-grain P1→P2 · G1-A → G2-A · axial +0"
    assert set(result["vectors"]) == {"G1", "G2"}
    assert "in-plane" not in json.dumps(result["lines"], ensure_ascii=False)


def test_vector_rejects_atoms_without_a_layer_with_a_clear_error(bridge):
    request = vector_request(0)
    del request["atoms"][1]["layer"]
    with pytest.raises(ValueError, match="layer"):
        bridge._vector(request)
