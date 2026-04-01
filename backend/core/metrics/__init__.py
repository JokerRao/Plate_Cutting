from core.metrics.cutting_metrics import calculate_cutting_metrics
from core.metrics.solution_selection import (
    compare_solutions_lexicographic,
    log_candidate_metrics,
    log_selection_rationale,
    select_best_solution,
)

# 历史名称兼容（与旧 compare_algorithms 等价）
compare_algorithms = compare_solutions_lexicographic

__all__ = [
    "calculate_cutting_metrics",
    "compare_algorithms",
    "compare_solutions_lexicographic",
    "select_best_solution",
    "log_candidate_metrics",
    "log_selection_rationale",
]
