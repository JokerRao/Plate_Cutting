"""OR-Tools 方案 A：面积分板 + 每板 rectpack + 与顺序流一致的 refine。"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import get_settings
from core.metrics import calculate_cutting_metrics
from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import PlateOptimizer, StockOptimizer
from engine.ortools_packing import solve_area_assignment_to_bins
from engine.plate_packing_protocol import PlatePackingEngine

from engine.pipeline.output import finalize_plate_output
from engine.pipeline.sequential import finalize_metrics_and_refine
from engine.pipeline.templates import clone_plate_template
from engine.pipeline.trace_context import CuttingTraceContext

logger = logging.getLogger("plate_cutting")


def run_ortools_assign_then_rectpack(
    big_plates: List[SmallPlate],
    plate_templates: List[SmallPlate],
    small_plates: List[SmallPlate],
    stock_plates: List[SmallPlate],
    config: CuttingConfig,
    converter: DataConverter,
    inner_algo_class: Any,
    stock_algorithm: str,
    optim: int,
    fallback_rectpack: Callable[
        [],
        Tuple[List[Dict[str, Any]], Dict[str, Any]],
    ],
    trace: Optional[CuttingTraceContext] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    settings = get_settings()
    n_items = len(small_plates)
    if n_items > settings.ORTOOLS_ASSIGN_MAX_ITEMS:
        logger.info(
            "ORToolsAssign: 件数 %d 超过 ORTOOLS_ASSIGN_MAX_ITEMS，使用顺序 rectpack",
            n_items,
        )
        return fallback_rectpack()

    bt = int(round(float(config.blade_thickness)))
    areas = [
        (int(round(sp.length)) + bt) * (int(round(sp.width)) + bt)
        for sp in small_plates
    ]
    # Use effective plate area (net of one blade kerf on each edge) so the
    # area-based capacity matches the actual footprint budget available to pieces.
    caps = [
        max(1, int(round(float(pt.length))) - bt) * max(1, int(round(float(pt.width))) - bt)
        for pt in plate_templates
    ]
    assign = solve_area_assignment_to_bins(
        areas,
        caps,
        settings.ORTOOLS_ASSIGN_TIME_LIMIT_SEC,
    )
    if assign is None:
        return fallback_rectpack()

    if trace:
        trace.stage(
            "ortools_assign_solved",
            n_items=n_items,
            n_bins=len(plate_templates),
            assign_sample=assign[: min(20, len(assign))],
        )

    bins_content: List[List[SmallPlate]] = [[] for _ in plate_templates]
    for idx, b in enumerate(assign):
        bins_content[b].append(small_plates[idx])

    plate_engine: PlatePackingEngine = PlateOptimizer(config, inner_algo_class)
    stock_optimizer = StockOptimizer(config, stock_algorithm)
    results: List[Dict[str, Any]] = []
    row_template_idx: List[int] = []
    pool: List[SmallPlate] = []

    for i, _ in enumerate(big_plates):
        to_pack = bins_content[i] + pool
        if not to_pack:
            continue
        bp = clone_plate_template(plate_templates[i])
        order_cuts, pool = plate_engine.pack_orders(bp, to_pack)
        if trace:
            trace.stage(
                "ortools_plate_packed",
                plate_index=i,
                n_cuts=len(order_cuts),
                remaining_after=len(pool),
            )
        if order_cuts:
            results.append(
                finalize_plate_output(
                    bp,
                    order_cuts,
                    stock_plates,
                    stock_optimizer,
                    optim,
                    config,
                    converter,
                )
            )
            row_template_idx.append(i)

    return finalize_metrics_and_refine(
        results,
        row_template_idx,
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
