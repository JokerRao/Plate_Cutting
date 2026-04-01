"""core.metrics：指标与字典序选优（无 HTTP、无 rectpack）。"""
from core.metrics import (
    calculate_cutting_metrics,
    compare_algorithms,
    select_best_solution,
)


def test_calculate_cutting_metrics_empty_with_remaining():
    m = calculate_cutting_metrics([], remaining_orders=7)
    assert m["used_plates"] == 0
    assert m["remaining_orders"] == 7
    assert m["overall_rate"] == 0


def _m(
    used_plates: int,
    min_rate: float,
    overall_rate: float,
    rate_variance: float = 0.0,
    total_cuts: int = 0,
    max_rate: float = 0.9,
):
    return {
        "used_plates": used_plates,
        "min_rate": min_rate,
        "overall_rate": overall_rate,
        "rate_variance": rate_variance,
        "total_cuts": total_cuts,
        "max_rate": max_rate,
        "remaining_orders": 0,
        "order_completion": 0,
        "avg_cuts_per_plate": 0.0,
        "last_rate": overall_rate,
    }


def test_compare_algorithms_fewer_plates_wins():
    a = _m(used_plates=3, min_rate=0.5, overall_rate=0.8)
    b = _m(used_plates=5, min_rate=0.9, overall_rate=0.95)
    # metrics1 更优时返回 -1（compare 约定：-1 表示第一参数更优，见 cutting_service 用法）
    assert compare_algorithms(a, b) == -1
    assert compare_algorithms(b, a) == 1


def test_compare_algorithms_same_plates_higher_variance_wins():
    a = _m(used_plates=2, min_rate=0.7, overall_rate=0.85, rate_variance=0.05)
    b = _m(used_plates=2, min_rate=0.6, overall_rate=0.95, rate_variance=0.01)
    assert compare_algorithms(a, b) == -1


def test_select_best_solution_picks_fewest_plates():
    candidates = [
        ("algo_heavy", [], _m(4, 0.8, 0.82)),
        ("algo_light", [], _m(3, 0.75, 0.80)),
    ]
    name, _results, best_m = select_best_solution(candidates)
    assert name == "algo_light"
    assert best_m["used_plates"] == 3


def test_select_best_solution_empty():
    assert select_best_solution([]) == (None, None, None)
