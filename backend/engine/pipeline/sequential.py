"""顺序逐张大板装箱 + 可选低利用率 refine。"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.metrics import calculate_cutting_metrics
from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import StockOptimizer
from engine.plate_packing_protocol import PlatePackingEngine

from engine.pipeline.constants import REFINE_LOW_UTIL_THRESHOLD, REFINE_MAX_PASSES
from engine.pipeline.output import finalize_plate_output
from engine.pipeline.refine import refine_low_utilization_plates
from engine.pipeline.templates import clone_plate_template
from engine.pipeline.trace_context import CuttingTraceContext

logger = logging.getLogger("plate_cutting")


def run_sequential_plate_loop(
    big_plates: List[SmallPlate],
    plate_templates: List[SmallPlate],
    remaining_orders: List[SmallPlate],
    plate_engine: PlatePackingEngine,
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
    trace: Optional[CuttingTraceContext] = None,
) -> Tuple[List[Dict[str, Any]], List[int], List[SmallPlate]]:
    results: List[Dict[str, Any]] = []
    row_template_idx: List[int] = []
    pool = remaining_orders

    for i, _ in enumerate(big_plates):
        if not pool:
            break
        bp = clone_plate_template(plate_templates[i])
        order_cuts, pool = plate_engine.pack_orders(bp, pool)
        if trace:
            trace.stage(
                "plate_packed",
                plate_index=i,
                n_cuts=len(order_cuts),
                remaining_after=len(pool),
            )
        if order_cuts:
            result = finalize_plate_output(
                bp,
                order_cuts,
                stock_plates,
                stock_optimizer,
                optim,
                config,
                converter,
            )
            results.append(result)
            row_template_idx.append(i)

    return results, row_template_idx, pool


def finalize_metrics_and_refine(
    results: List[Dict[str, Any]],
    row_template_idx: List[int],
    remaining_orders: List[SmallPlate],
    plate_templates: List[SmallPlate],
    plate_engine: PlatePackingEngine,
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
    trace: Optional[CuttingTraceContext] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics = calculate_cutting_metrics(results, len(remaining_orders))

    if not remaining_orders and results:
        refined = refine_low_utilization_plates(
            results,
            row_template_idx,
            plate_templates,
            plate_engine,
            stock_optimizer,
            stock_plates,
            optim,
            config,
            converter,
        )
        if refined is not None:
            results = refined
            metrics = calculate_cutting_metrics(results, 0)
            if trace:
                trace.stage("refine_applied", n_plates=len(results))
            logger.info(
                "Applied low-utilization repack (threshold=%.0f%%, max_passes=%d)",
                REFINE_LOW_UTIL_THRESHOLD * 100,
                REFINE_MAX_PASSES,
            )
    elif trace:
        trace.stage(
            "skip_refine",
            reason="remaining_orders" if remaining_orders else "no_results",
            n_remaining=len(remaining_orders),
        )

    return results, metrics
