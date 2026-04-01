"""低利用率板拆件重排（依赖 PlatePackingEngine 协议）。"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.metrics import calculate_cutting_metrics, compare_algorithms
from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import StockOptimizer
from engine.plate_packing_protocol import PlatePackingEngine

from engine.pipeline.constants import REFINE_LOW_UTIL_THRESHOLD, REFINE_MAX_PASSES
from engine.pipeline.output import finalize_plate_output, orders_from_result_orders_only
from engine.pipeline.templates import clone_plate_template

logger = logging.getLogger("plate_cutting")


def single_repack_low_util_pass(
    results: List[Dict[str, Any]],
    row_template_idx: List[int],
    plate_templates: List[SmallPlate],
    plate_engine: PlatePackingEngine,
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[Tuple[List[Dict[str, Any]], List[int]]]:
    bad_positions = [
        pos for pos, r in enumerate(results) if r["rate"] < REFINE_LOW_UTIL_THRESHOLD
    ]
    if not bad_positions:
        return None

    sorted_bad = sorted(bad_positions, key=lambda pos: row_template_idx[pos])
    pool: List[SmallPlate] = []
    for pos in sorted_bad:
        pool.extend(orders_from_result_orders_only(results[pos]))
    bad_templates = [row_template_idx[pos] for pos in sorted_bad]

    kept: List[Tuple[int, Dict[str, Any]]] = [
        (row_template_idx[i], results[i])
        for i in range(len(results))
        if i not in bad_positions
    ]

    remaining = pool
    new_rows: List[Tuple[int, Dict[str, Any]]] = []
    for tpl_idx in bad_templates:
        if not remaining:
            break
        bp = clone_plate_template(plate_templates[tpl_idx])
        order_cuts, remaining = plate_engine.pack_orders(bp, remaining)
        if order_cuts:
            row = finalize_plate_output(
                bp,
                order_cuts,
                stock_plates,
                stock_optimizer,
                optim,
                config,
                converter,
            )
            new_rows.append((tpl_idx, row))

    if remaining:
        return None

    combined = kept + new_rows
    combined.sort(key=lambda t: t[0])
    return (
        [t[1] for t in combined],
        [t[0] for t in combined],
    )


def refine_low_utilization_plates(
    initial_results: List[Dict[str, Any]],
    initial_row_template_idx: List[int],
    plate_templates: List[SmallPlate],
    plate_engine: PlatePackingEngine,
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[List[Dict[str, Any]]]:
    m0 = calculate_cutting_metrics(initial_results, 0)
    working = initial_results
    working_idx = initial_row_template_idx

    for _ in range(REFINE_MAX_PASSES):
        nxt = single_repack_low_util_pass(
            working,
            working_idx,
            plate_templates,
            plate_engine,
            stock_optimizer,
            stock_plates,
            optim,
            config,
            converter,
        )
        if nxt is None:
            break
        cand, cand_idx = nxt
        m_w = calculate_cutting_metrics(working, 0)
        m_c = calculate_cutting_metrics(cand, 0)
        # 如果候选项不如当前(返回-1)或者实质相同(返回0，比如在重新塞了一次单板，方差完全相同)，则停止，避免无限循环
        if compare_algorithms(m_w, m_c) <= 0:
            break
        working, working_idx = cand, cand_idx
        if not any(r["rate"] < REFINE_LOW_UTIL_THRESHOLD for r in working):
            break

    if working is initial_results:
        return None
    m_final = calculate_cutting_metrics(working, 0)
    if compare_algorithms(m0, m_final) <= 0:
        return None
    return working
