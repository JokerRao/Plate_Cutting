"""OR-Tools 分板与 CP-2D 单张板装箱（需安装 ortools）。"""
import pytest

pytest.importorskip("ortools", reason="ortools not installed")

from services.cutting_service import (  # noqa: E402
    optimize_cutting,
    run_single_algorithm,
)
from engine.ortools_packing import (  # noqa: E402
    pack_single_plate_cp2d,
    solve_area_assignment_to_bins,
)
from core.models import CuttingConfig, SmallPlate  # noqa: E402


def test_solve_area_assignment_to_bins():
    areas = [30, 30, 30]
    caps = [100, 100]
    a = solve_area_assignment_to_bins(areas, caps, 2.0)
    assert a is not None
    assert len(a) == 3
    assert all(0 <= b < 2 for b in a)


def test_pack_single_plate_cp2d_two_pieces():
    cfg = CuttingConfig(blade_thickness=4)
    bp = SmallPlate(100, 100, quantity=1)
    orders = [
        SmallPlate(40, 40, plate_id="a"),
        SmallPlate(40, 40, plate_id="b"),
    ]
    cuts, rem = pack_single_plate_cp2d(
        bp, orders, cfg, lambda i: False, 5.0)
    assert len(rem) == 0
    assert len(cuts) == 2


def test_optimize_cutting_ortools_cp2d():
    plates = [
        {"id": "p1", "length": 2000, "width": 1000, "quantity": 1},
    ]
    orders = [
        {"id": "o1", "length": 400, "width": 300, "quantity": 3},
    ]
    out = optimize_cutting(
        plates,
        orders,
        others=[],
        algorithm="ORToolsCP2D",
        saw_blade=4.0,
    )
    assert len(out) >= 1
    assert out[0]["rate"] > 0


def test_run_single_algorithm_ortools_assign():
    plates = [
        {"id": "p1", "length": 1000, "width": 500, "quantity": 2},
    ]
    orders = [
        {"id": "o1", "length": 200, "width": 100, "quantity": 5},
    ]
    results, metrics = run_single_algorithm(
        plates,
        orders,
        [],
        optim=0,
        saw_blade=4.0,
        algorithm="ORToolsAssignMaxRects",
        stock_algorithm="maxrects_baf",
    )
    assert metrics["remaining_orders"] == 0
    assert len(results) >= 1
