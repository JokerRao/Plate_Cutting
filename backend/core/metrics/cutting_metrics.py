"""切割方案指标统计（与算法、选优逻辑解耦，便于单独演进）。"""
from typing import Any, Dict, List


def calculate_cutting_metrics(
        results: List[Dict[str, Any]], remaining_orders: int) -> Dict[str, Any]:
    """
    计算切割方案的详细指标

    Returns:
        包含多个评价指标的字典
    """
    if not results:
        return {
            'used_plates': 0,
            'overall_rate': 0,
            'last_rate': 0,
            'min_rate': 0,
            'max_rate': 0,
            'rate_variance': 0,
            'distinct_patterns': 0,
            'total_cuts': 0,
            'avg_cuts_per_plate': 0,
            'order_completion': 0,
            'remaining_orders': remaining_orders,
            'max_cuts_single_plate': 0,
        }

    used_plates = len(results)
    rates = [r['rate'] for r in results]
    overall_rate = sum(rates) / used_plates if used_plates > 0 else 0
    last_rate = rates[-1]
    min_rate = min(rates) if rates else 0
    max_rate = max(rates) if rates else 0
    rate_variance = (
        sum((r - overall_rate) ** 2 for r in rates) / used_plates
        if used_plates > 0 else 0
    )
    
    # Calculate distinct patterns
    from collections import defaultdict
    def hash_cut(c):
        return (round(float(c[2]), 1), round(float(c[3]), 1))
        
    pattern_groups = defaultdict(list)
    for i, r in enumerate(results):
        # Only count order pieces (is_stock is at index 4)
        pieces = sorted([hash_cut(c) for c in r['cutted'] if not c[4]])
        pattern_groups[str(pieces)].append(i)
        
    distinct_patterns = len(pattern_groups)

    total_cuts = sum(len(r['cutted']) for r in results)
    avg_cuts_per_plate = total_cuts / used_plates if used_plates > 0 else 0
    max_cuts_single_plate = max(len(r['cutted']) for r in results) if results else 0
    total_order_cuts = sum(
        1 for r in results
        for cut in r['cutted']
        if cut[4] == 0
    )

    return {
        'used_plates': used_plates,
        'overall_rate': overall_rate,
        'last_rate': last_rate,
        'min_rate': min_rate,
        'max_rate': max_rate,
        'rate_variance': rate_variance,
        'distinct_patterns': distinct_patterns,
        'total_cuts': total_cuts,
        'avg_cuts_per_plate': avg_cuts_per_plate,
        'max_cuts_single_plate': max_cuts_single_plate,
        'order_completion': total_order_cuts,
        'remaining_orders': remaining_orders,
    }
