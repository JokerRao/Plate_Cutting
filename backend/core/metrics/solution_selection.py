"""方案比较与选优（与具体装箱算法解耦，可替换比较策略）。"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger('plate_cutting')


def compare_solutions_lexicographic(
        metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> int:
    """
    字典序比较两个方案的指标。

    优先级从高到低：
    1. 使用大板张数越少越好
    2. 利用率方差越大越好（鼓励拼满单板形成完整大片余料，而不是均摊成碎片）
    3. 平均单板利用率越高越好
    4. 最高单板利用率越高越好
    5. 总切割次数越少越好

    Returns:
        -1: metrics1 更优
         0: 实质相同
         1: metrics2 更优
    """
    _RATE_EPS = 0.0005
    _VAR_EPS = 1e-9

    if metrics1['used_plates'] != metrics2['used_plates']:
        return -1 if metrics1['used_plates'] < metrics2['used_plates'] else 1

    if metrics1.get('distinct_patterns', 0) != metrics2.get('distinct_patterns', 0):
        # 版型越少越好，极大降低工人的分拣和生产换线成本
        return -1 if metrics1.get('distinct_patterns', 0) < metrics2.get('distinct_patterns', 0) else 1

    if abs(metrics1['rate_variance'] - metrics2['rate_variance']) > _VAR_EPS:
        # Variance 越大越好：大意味着有的板极满，有的很空，这非常利于余料留存
        return -1 if metrics1['rate_variance'] > metrics2['rate_variance'] else 1

    if abs(metrics1['overall_rate'] - metrics2['overall_rate']) > _RATE_EPS:
        return -1 if metrics1['overall_rate'] > metrics2['overall_rate'] else 1

    if abs(metrics1['max_rate'] - metrics2['max_rate']) > _RATE_EPS:
        return -1 if metrics1['max_rate'] > metrics2['max_rate'] else 1

    if metrics1['total_cuts'] != metrics2['total_cuts']:
        return -1 if metrics1['total_cuts'] < metrics2['total_cuts'] else 1

    return 0


def select_best_solution(
    candidates: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    从若干 (算法名, 切割结果, 指标) 中选出最优。

    Returns:
        (best_name, best_results, best_metrics)；candidates 为空时全为 None。
    """
    if not candidates:
        return None, None, None

    best_name: Optional[str] = None
    best_results: Optional[List[Dict[str, Any]]] = None
    best_metrics: Optional[Dict[str, Any]] = None

    for name, results, metrics in candidates:
        if best_metrics is None or compare_solutions_lexicographic(
                metrics, best_metrics) < 0:
            best_name = name
            best_results = results
            best_metrics = metrics

    return best_name, best_results, best_metrics


def log_candidate_metrics(algo_name: str, metrics: Dict[str, Any]) -> None:
    """记录单次算法跑出的指标（供 auto 模式排查）。"""
    logger.info(f"  {algo_name} 结果:")
    logger.info(f"    - 使用板材: {metrics['used_plates']} 块")
    logger.info(f"    - 平均利用率: {metrics['overall_rate']:.2%}")
    logger.info(f"    - 最低利用率: {metrics['min_rate']:.2%}")
    logger.info(f"    - 利用率方差: {metrics['rate_variance']:.4f}")
    logger.info(f"    - 独有版型: {metrics.get('distinct_patterns', 0)} 种")
    logger.info(
        f"    - 平均切割数: {metrics['avg_cuts_per_plate']:.1f} 次/板")
    logger.info(
        f"    - 最大单板切割: {metrics['max_cuts_single_plate']} 次")
    logger.info(f"    - 剩余订单: {metrics['remaining_orders']} 个")


def log_selection_rationale(
    best_name: str,
    algorithm_results: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]],
) -> None:
    """相对最优算法，输出与其它算法的差异说明。"""
    best_metrics = None
    for n, _, m in algorithm_results:
        if n == best_name:
            best_metrics = m
            break
    if best_metrics is None:
        return
    logger.info(f"\n最优算法: {best_name}")
    logger.info("选择理由:")
    for algo_name, _, metrics in algorithm_results:
        if algo_name == best_name:
            continue
        if best_metrics['used_plates'] < metrics['used_plates']:
            logger.info(
                f"  - 比 {algo_name} 少用 {metrics['used_plates'] - best_metrics['used_plates']} 块板")
        elif best_metrics.get('distinct_patterns', 0) < metrics.get('distinct_patterns', 0):
            logger.info(
                f"  - 比 {algo_name} 版型更少，降低生产换线成本 ({best_metrics['distinct_patterns']} vs {metrics['distinct_patterns']})")
        elif best_metrics['rate_variance'] > metrics['rate_variance']:
            logger.info(
                f"  - 比 {algo_name} 拼板紧凑度(利用率方差)更高，产生的离散碎片更少")
        elif best_metrics['overall_rate'] > metrics['overall_rate']:
            logger.info(
                f"  - 比 {algo_name} 平均利用率高 {(best_metrics['overall_rate'] - metrics['overall_rate']) * 100:.2f}%")
        elif best_metrics['total_cuts'] < metrics['total_cuts']:
            logger.info(
                f"  - 比 {algo_name} 总切割次数少 {metrics['total_cuts'] - best_metrics['total_cuts']} 次")
