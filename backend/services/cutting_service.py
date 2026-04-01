import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from core.metrics import (
    calculate_cutting_metrics,
    compare_algorithms,
    log_candidate_metrics,
    log_selection_rationale,
    select_best_solution,
)
from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.cutting_algorithms import (
    FALLBACK_PACKING_CLASS,
    ORTOOLS_PACKING_IDS,
    STOCK_ALGORITHM_LABELS,
    is_registered_packing_id,
    iter_enabled_packing_algorithms,
    normalize_enabled_packing_ids,
    normalize_enabled_stock_ids,
    resolve_packing_class,
    resolve_stock_algorithm,
)
from engine.optimizers import PlateOptimizer, StockOptimizer
from engine.ortools_plate_engines import ORToolsCP2DEngine
from engine.pipeline.constants import REFINE_LOW_UTIL_THRESHOLD, REFINE_MAX_PASSES
from engine.pipeline.common_dim_strip import run_common_dim_strip_then_rectpack
from engine.pipeline.consolidate import consolidate_layout_groups, layout_groups
from engine.pipeline.cut_simplifier import apply_column_sort_pass, simplify_board_cuts
from engine.pipeline.ortools_assign import run_ortools_assign_then_rectpack as ortools_assign_pipeline
from engine.pipeline.prepare import empty_stock_if_none, load_converted_inputs
from engine.pipeline.sequential import finalize_metrics_and_refine, run_sequential_plate_loop
from engine.pipeline.trace_context import CuttingTraceContext

logger = logging.getLogger("plate_cutting")


def _pre_sort_composite_affinity(
    small_plates: List[SmallPlate],
    big_plates: List[SmallPlate],
    config: CuttingConfig,
) -> List[SmallPlate]:
    """
    Pre-sort small plates so that pieces with the highest per-board density affinity
    with the big plates come first (are used in the earliest waves).

    When multiple small piece types exist along with big pieces, mixing types on each
    board is suboptimal.  E.g. a 2-big + 4-small_806 layout (93.4%) beats a mixed
    2-big + 2-small_806 + 1-small_1006 layout (88.6%).

    Strategy: score each distinct small-piece dimension group by how many of them
    can pack alongside one big piece on a board, and sort groups from highest
    per-board density to lowest.  Inter-group ordering preserves original order.
    """
    if not big_plates or not small_plates:
        return small_plates

    bt = config.blade_thickness
    bp = big_plates[0]
    L, W = float(bp.length), float(bp.width)

    # Count identical big pieces per board (rough estimate)
    big_w = float(big_plates[0].length) + bt
    big_h = float(big_plates[0].width) + bt
    bigs_per_board = max(1, int(L // big_w) * int(W // big_h))
    big_area_per_board = bigs_per_board * big_w * big_h
    remaining_area = L * W - big_area_per_board

    # Group small pieces by their canonical size key
    from collections import defaultdict
    groups: Dict[tuple, List[SmallPlate]] = defaultdict(list)
    for p in small_plates:
        pw, ph = float(p.length) + bt, float(p.width) + bt
        # Try both orientations, pick the one that fits better alongside big pieces
        fit1 = int(L // pw) * int(W // ph) if pw <= L and ph <= W else 0
        fit2 = int(L // ph) * int(W // pw) if ph <= L and pw <= W else 0
        if fit2 > fit1:
            key = (int(round(ph)), int(round(pw)))
        else:
            key = (int(round(pw)), int(round(ph)))
        groups[key].append(p)

    if len(groups) <= 1:
        return small_plates  # Nothing to reorder

    # Score each group by density when packed into the remaining space per board
    def group_score(key: tuple) -> float:
        gw, gh = float(key[0]), float(key[1])
        n_fit = int((L) // gw) * int((W) // gh)
        area_util = n_fit * gw * gh / (L * W) if L * W > 0 else 0
        # Prefer groups that fill the board well (high density)
        return area_util

    sorted_keys = sorted(groups.keys(), key=group_score, reverse=True)

    # Rebuild the list group by group (highest density first)
    result: List[SmallPlate] = []
    for key in sorted_keys:
        result.extend(groups[key])
    return result


def _packing_trace_label(algorithm: Any) -> str:
    if isinstance(algorithm, str):
        return algorithm
    return getattr(algorithm, "__name__", type(algorithm).__name__)


def _run_rectpack_sequential_from_prepared(
    converter: Any,
    big_plates: List[SmallPlate],
    plate_templates: List[SmallPlate],
    small_plates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    rectpack_algo_class: Any,
    stock_algorithm: str,
    optim: int,
    trace: Optional[CuttingTraceContext],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    plate_engine = PlateOptimizer(config, rectpack_algo_class)
    stock_optimizer = StockOptimizer(config, stock_algorithm)
    sorted_small = _pre_sort_composite_affinity(small_plates, big_plates, config)
    results, row_idx, pool = run_sequential_plate_loop(
        big_plates,
        plate_templates,
        list(sorted_small),
        plate_engine,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
        trace,
    )
    return finalize_metrics_and_refine(
        results,
        row_idx,
        pool,
        plate_templates,
        plate_engine,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
        trace,
    )
from engine.pipeline.cut_simplifier import simplify_board_cuts
from engine.pipeline.row_sort_pass import apply_row_sort_pass
def _run_single_algorithm(
    converter: Any,
    big_plates: List[SmallPlate],
    plate_templates: List[SmallPlate],
    small_plates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    optim: int,
    algorithm: Any,
    stock_algorithm: str = "maxrects_baf",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    运行单个算法的切割方案。

    每张板在 PlateOptimizer 内会对多种零件加入顺序做 rectpack 试探并取本张最优。
    若全部排完，会对利用率低于 REFINE_LOW_UTIL_THRESHOLD 的板做拆件重排（最多
    REFINE_MAX_PASSES 轮），且仅当整体指标按 compare_algorithms 严格优于重排前时才替换结果。

    Args:
        stock_algorithm: 库存填充算法
            - "maxrects_baf": MaxRects Best Area Fit（默认）
            - "guillotine_bssf_llas": "Guillotine BSSF+LLAS"

    Returns:
        (切割方案列表, 评价指标字典)
    """
    if not big_plates:
        return [], calculate_cutting_metrics([], len(small_plates))

    settings = get_settings()
    trace = CuttingTraceContext.from_settings(
        _packing_trace_label(algorithm), settings)
    trace.summarize_plates_orders(
        len(big_plates), len(small_plates), len(stock_plates))

    def fallback_rectpack() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        inner_fallback = resolve_packing_class(
            settings.ORTOOLS_ASSIGN_INNER_PACKING_ID if algorithm == "ORToolsAssignMaxRects" else "MaxRectsBaf"
        )
        return _run_rectpack_sequential_from_prepared(
            converter,
            big_plates,
            plate_templates,
            list(small_plates),
            stock_plates,
            config,
            inner_fallback,
            stock_algorithm,
            optim,
            trace,
        )

    if algorithm == "ORToolsAssignMaxRects":
        inner_cls = resolve_packing_class(settings.ORTOOLS_ASSIGN_INNER_PACKING_ID)
        return ortools_assign_pipeline(
            big_plates,
            plate_templates,
            list(small_plates),
            stock_plates,
            config,
            converter,
            inner_cls,
            stock_algorithm,
            optim,
            fallback_rectpack,
            trace,
        )

    if algorithm == "CommonDimStrip":
        inner_cls = resolve_packing_class("MaxRectsBaf")
        return run_common_dim_strip_then_rectpack(
            big_plates,
            plate_templates,
            list(small_plates),
            stock_plates,
            config,
            converter,
            inner_cls,
            stock_algorithm,
            optim,
            fallback_rectpack,
            trace,
        )

    if algorithm == "ORToolsCP2D":
        plate_engine: Any = ORToolsCP2DEngine(
            config,
            settings.ORTOOLS_CP2D_TIME_LIMIT_SEC,
            settings.ORTOOLS_MAX_PIECES_CP2D,
        )
    else:
        plate_engine = PlateOptimizer(config, algorithm)

    stock_optimizer = StockOptimizer(config, stock_algorithm)
    results, row_idx, pool = run_sequential_plate_loop(
        big_plates,
        plate_templates,
        list(small_plates),
        plate_engine,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
        trace,
    )
    return finalize_metrics_and_refine(
        results,
        row_idx,
        pool,
        plate_templates,
        plate_engine,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
        trace,
    )


def _sort_results_by_layout_group(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将结果按照版型聚集（相同版型相邻），并按频次降序排列。"""
    if not results:
        return results
    groups = layout_groups(results)
    sorted_results = []
    for _, idxs in groups.items():
        for i in idxs:
            sorted_results.append(results[i])
    return sorted_results

def _run_consolidation_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    optim: int,
    stock_algorithm: str,
) -> List[Dict[str, Any]]:
    if len(results) <= 1 or not plate_templates:
        return results

    plate_engine = PlateOptimizer(config, resolve_packing_class("MaxRectsBaf"))
    stock_optimizer = StockOptimizer(config, stock_algorithm)

    consolidated = consolidate_layout_groups(
        results,
        plate_templates,
        plate_engine,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
    )
    return consolidated if consolidated is not None else results


def _run_cut_simplifier_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    optim: int,
    stock_algorithm: str,
) -> List[Dict[str, Any]]:
    if not results or not plate_templates:
        return results

    stock_optimizer = StockOptimizer(config, stock_algorithm)
    simplified = simplify_board_cuts(
        results,
        plate_templates,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
    )
    return simplified if simplified is not None else results


def _run_column_sort_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    optim: int,
    stock_algorithm: str,
) -> List[Dict[str, Any]]:
    if not results or not plate_templates:
        return results

    stock_optimizer = StockOptimizer(config, stock_algorithm)
    sorted_results = apply_column_sort_pass(
        results,
        plate_templates,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
    )
    return sorted_results if sorted_results is not None else results


def _run_row_sort_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    optim: int,
    stock_algorithm: str,
) -> List[Dict[str, Any]]:
    if not results or not plate_templates:
        return results

    stock_optimizer = StockOptimizer(config, stock_algorithm)
    sorted_results = apply_row_sort_pass(
        results,
        plate_templates,
        stock_optimizer,
        stock_plates,
        optim,
        config,
        converter,
    )
    return sorted_results if sorted_results is not None else results


def _run_post_pack_stock_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    optim: int,
    stock_effective: str,
    stock_plates: List[SmallPlate],
) -> List[Dict[str, Any]]:
    # Previously in cutting_service.py there wasn't a separate function for this, but I referenced it. Let's see if it existed.
    pass

def optimize_cutting(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]],
                     others: List[Dict[str, Any]] = None, optim: int = 0,
                     saw_blade: float = 4.0, algorithm: str = "auto",
                     stock_algorithm: str = "maxrects_baf",
                     enable_row_complementary: bool = True) -> List[Dict[str, Any]]:
    """
    主优化函数

    Args:
        plates: 大板信息列表
        orders: 订单信息列表
        others: 库存余料列表
        optim: 是否启用库存优化（仅影响库存板）
        saw_blade: 锯片厚度
        algorithm: 主算法选择
            - "MaxRectsBaf": MaxRects Best Area Fit
            - "GuillotineBafMinas": Guillotine Best Area Fit with Minimal Area Split
            - "SkylineMwfWm": Skyline Minimal Waste Fit with Merge
            - "ORToolsAssignMaxRects": OR-Tools 面积分板 + rectpack（内层 id 见 ORTOOLS_ASSIGN_INNER_PACKING_ID）
            - "ORToolsCP2D": 单张板内 CP-SAT 二维不重叠，件数过多时回退 rectpack
            - "auto": 自动优化模式（默认）- 尝试三种算法，选择最优
        stock_algorithm: 库存填充算法
            - "maxrects_baf": MaxRects Best Area Fit（默认）
            - "guillotine_bssf_llas": "Guillotine BSSF+LLAS"
        enable_row_complementary: 是否启用等高互补时的行式排布（默认 True）

    Returns:
        切割方案列表
    """
    settings = get_settings()
    stock_effective = resolve_stock_algorithm(
        stock_algorithm,
        normalize_enabled_stock_ids(settings.STOCK_ALGORITHMS_ENABLED),
    )
    stock_label = STOCK_ALGORITHM_LABELS.get(
        stock_effective, stock_effective)

    config = CuttingConfig(
        blade_thickness=saw_blade,
        enable_row_complementary=enable_row_complementary,
    )
    others_list = empty_stock_if_none(others)
    (
        converter,
        big_plates,
        plate_templates,
        small_plates,
        stock_plates,
    ) = load_converted_inputs(plates, orders, others_list, config)

    if not big_plates:
        return []

    if algorithm == "auto":
        enabled_ids = normalize_enabled_packing_ids(
            settings.CUTTING_ALGORITHMS_ENABLED)
        logger.info(
            "使用自动优化模式，启用的主算法: %s",
            ",".join(enabled_ids),
        )
        logger.info("库存填充策略: %s", stock_label)

        algorithm_results = []
        for algo_name, algo_class in iter_enabled_packing_algorithms(
                enabled_ids):
            logger.info("测试算法: %s", algo_name)
            results, metrics = _run_single_algorithm(
                converter,
                big_plates,
                plate_templates,
                list(small_plates),
                stock_plates,
                config,
                optim,
                algo_class,
                stock_effective,
            )
            algorithm_results.append((algo_name, results, metrics))
            log_candidate_metrics(algo_name, metrics)

        if not algorithm_results:
            logger.error(
                "没有可运行的主装箱算法，请检查 CUTTING_ALGORITHMS_ENABLED",
            )
            return []

        best_name, best_results, _ = select_best_solution(algorithm_results)
        if best_name:
            log_selection_rationale(best_name, algorithm_results)
        final = best_results if best_results is not None else []
    else:
        if is_registered_packing_id(algorithm):
            logger.info("使用算法: %s", algorithm)
            if algorithm == "ORToolsAssignMaxRects":
                logger.info(
                    "ORToolsAssign 内层 rectpack: %s",
                    settings.ORTOOLS_ASSIGN_INNER_PACKING_ID,
                )
            logger.info("库存填充策略: %s", stock_label)
            algo_spec: Any = (
                algorithm
                if algorithm in ORTOOLS_PACKING_IDS
                else resolve_packing_class(algorithm)
            )
        else:
            logger.warning(
                "未知主算法 id '%s'，使用回退 %s",
                algorithm,
                getattr(FALLBACK_PACKING_CLASS, "__name__", "fallback"),
            )
            algo_spec = FALLBACK_PACKING_CLASS

        final, metrics = _run_single_algorithm(
            converter,
            big_plates,
            plate_templates,
            list(small_plates),
            stock_plates,
            config,
            optim,
            algo_spec,
            stock_effective,
        )
        logger.info("完成切割:")
        logger.info("  - 使用板材: %s 块", metrics['used_plates'])
        logger.info("  - 平均利用率: %.2f%%", metrics['overall_rate'] * 100)
        logger.info(
            "  - 平均切割数: %.1f 次/板", metrics['avg_cuts_per_plate'])

    if final:
        final = _run_consolidation_pass(
            final, plate_templates, stock_plates, config, converter, optim, stock_effective
        )
        final = _run_cut_simplifier_pass(
            final, plate_templates, stock_plates, config, converter, optim, stock_effective
        )
        final = _run_column_sort_pass(
            final, plate_templates, stock_plates, config, converter, optim, stock_effective
        )
        final = _run_row_sort_pass(
            final, plate_templates, stock_plates, config, converter, optim, stock_effective
        )

    return _sort_results_by_layout_group(final)
def optimize_cutting_multistart(
    plates: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    others: Optional[List[Dict[str, Any]]] = None,
    optim: int = 0,
    saw_blade: float = 4.0,
    algorithm: str = "auto",
    stock_algorithm: str = "maxrects_baf",
    n_starts: int = 1,
    multistart_seed: Optional[int] = None,
    enable_row_complementary: bool = True,
) -> List[Dict[str, Any]]:
    """
    多起点优化：第 1 次保持订单行顺序；之后各次随机打乱订单行顺序再跑 optimize_cutting，
    用 compare_algorithms 在指标空间选最优方案（适用于进程池顶层调用，需可 pickle）。
    """
    if others is None:
        others = []

    if n_starts <= 1:
        return optimize_cutting(
            plates,
            orders,
            others,
            optim,
            saw_blade,
            algorithm,
            stock_algorithm,
            enable_row_complementary=enable_row_complementary,
        )

    orders_base = deepcopy(orders)
    rng = random.Random(
        multistart_seed if multistart_seed is not None else 42)
    best_plans: Optional[List[Dict[str, Any]]] = None
    best_metrics: Optional[Dict[str, Any]] = None

    for k in range(n_starts):
        if k == 0:
            od = orders_base
        else:
            od = deepcopy(orders_base)
            rng.shuffle(od)
        plans = optimize_cutting(
            plates,
            od,
            others,
            optim,
            saw_blade,
            algorithm,
            stock_algorithm,
            enable_row_complementary=enable_row_complementary,
        )
        if not plans:
            continue
        m = calculate_cutting_metrics(plans, 0)
        if best_metrics is None or compare_algorithms(m, best_metrics) < 0:
            best_metrics = m
            best_plans = deepcopy(plans)

    logger.info(
        "Multistart optimization: n_starts=%d, picked metrics used_plates=%s min_rate=%.4f",
        n_starts,
        best_metrics["used_plates"] if best_metrics else None,
        best_metrics["min_rate"] if best_metrics else 0.0,
    )
    if get_settings().CUTTING_TRACE_LOG_STAGES:
        logger.info(
            "cutting_trace stage=multistart_done n_starts=%s algorithm=%s best_used_plates=%s",
            n_starts,
            algorithm,
            best_metrics["used_plates"] if best_metrics else None,
        )
    return best_plans if best_plans is not None else []


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 示例数据
    plates = [
        {"length": 2440, "width": 1220, "quantity": 5}
    ]

    orders = [
        {"id": "A001", "length": 600, "width": 400, "quantity": 3},
        {"id": "A002", "length": 800, "width": 500, "quantity": 2},
        {"id": "A003", "length": 400, "width": 300, "quantity": 4},
    ]

    others = [
        {"id": "R001", "length": 200, "width": 150},
        {"id": "R002", "length": 300, "width": 200},
    ]

    print("=== 板材切割优化器演示 ===\n")

    # 1. 使用自动优化模式 + MaxRects BAF库存算法
    print("1. 自动优化模式 + MaxRects BAF库存算法:")
    results_auto = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="auto",
        stock_algorithm="maxrects_baf")
    print(f"   生成 {len(results_auto)} 个切割方案\n")

    # 2. 使用自动优化模式 + Guillotine BSSF + LLAS库存算法
    print("2. 自动优化模式 + Guillotine BSSF + LLAS库存算法:")
    results_guillotine = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="auto",
        stock_algorithm="guillotine_bssf_llas")
    print(f"   生成 {len(results_guillotine)} 个切割方案\n")

    # 3. 使用MaxRects BAF主算法 + MaxRects BAF库存算法
    print("3. MaxRects BAF主算法 + MaxRects BAF库存算法:")
    results_maxrects = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="MaxRectsBaf",
        stock_algorithm="maxrects_baf")
    print(f"   生成 {len(results_maxrects)} 个切割方案\n")

    # 4. 使用MaxRects BAF主算法 + Guillotine BSSF + LLAS库存算法
    print("4. MaxRects BAF主算法 + Guillotine BSSF + LLAS库存算法:")
    results_mixed = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="MaxRectsBaf",
        stock_algorithm="guillotine_bssf_llas")
    print(f"   生成 {len(results_mixed)} 个切割方案\n")

    # 显示库存算法说明
    print("=== 库存填充算法说明 ===")
    print("MaxRects BAF: 使用最大矩形算法，选择面积最小的可用区域")
    print("Guillotine BSSF + LLAS: 使用切割线算法，采用短边最佳适配和长剩余边分割策略")

def run_single_algorithm(
    plates: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    others: Optional[List[Dict[str, Any]]],
    optim: int,
    saw_blade: float,
    algorithm: Any,
    stock_algorithm: str = "maxrects_baf",
    enable_row_complementary: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Backward compatibility wrapper for tests."""
    config = CuttingConfig(
        blade_thickness=saw_blade,
        enable_row_complementary=enable_row_complementary,
    )
    others_list = empty_stock_if_none(others)
    (
        converter,
        big_plates,
        plate_templates,
        small_plates,
        stock_plates,
    ) = load_converted_inputs(plates, orders, others_list, config)
    
    return _run_single_algorithm(
        converter,
        big_plates,
        plate_templates,
        list(small_plates),
        stock_plates,
        config,
        optim,
        algorithm,
        stock_algorithm,
    )
