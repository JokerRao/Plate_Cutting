"""
对比「行式互补排布」开/关：同一订单下张数、利用率差异（527×400 与 415×400 混排典型场景）。
"""
import rectpack
import pytest

from services.cutting_service import REFINE_LOW_UTIL_THRESHOLD, run_single_algorithm


PLATES = [{"length": 2440, "width": 1220, "quantity": 200}]
ORDERS = [
    {"id": "P1", "length": 527, "width": 400, "quantity": 200},
    {"id": "P2", "length": 415, "width": 400, "quantity": 200},
]


def _metrics_row_toggle(enable_row: bool):
    results, m = run_single_algorithm(
        PLATES,
        ORDERS,
        [],
        0,
        4.0,
        rectpack.MaxRectsBaf,
        "maxrects_baf",
        enable_row_complementary=enable_row,
    )
    low = sum(1 for r in results if r["rate"] < REFINE_LOW_UTIL_THRESHOLD)
    return results, m, low


@pytest.mark.parametrize("enable_row", [True, False])
def test_mixed_sizes_all_placed(enable_row: bool):
    """两种开关下均应排完 400 件。"""
    results, m, _ = _metrics_row_toggle(enable_row)
    assert m["remaining_orders"] == 0
    assert m["order_completion"] == 400
    assert len(results) == m["used_plates"]


def test_row_toggle_benchmark_summary():
    """
    汇总开/关行式互补的指标（看表: pytest -s tests/test_row_complementary.py -k summary）。
    """
    _, m_on, low_on = _metrics_row_toggle(True)
    _, m_off, low_off = _metrics_row_toggle(False)

    print(
        "\n=== 527/415×400×200 各 200 件 | 2440×1220 | MaxRectsBaf | 锯片4 ==="
    )
    print(
        f"{'行式互补':^12} | {'板数':>4} | {'平均利用率%':>10} | {'最低板%':>8} | "
        f"{'低于82%板数':>10}"
    )
    print("-" * 62)
    print(
        f"{'开启':^12} | {m_on['used_plates']:>4} | "
        f"{m_on['overall_rate'] * 100:>10.2f} | {m_on['min_rate'] * 100:>8.2f} | {low_on:>10}"
    )
    print(
        f"{'关闭':^12} | {m_off['used_plates']:>4} | "
        f"{m_off['overall_rate'] * 100:>10.2f} | {m_off['min_rate'] * 100:>8.2f} | {low_off:>10}"
    )

    assert m_on["remaining_orders"] == 0
    assert m_off["remaining_orders"] == 0
