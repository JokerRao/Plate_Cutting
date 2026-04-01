from typing import Any, Dict, List, Optional, Tuple
import logging

from core.models import Cut, CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import StockOptimizer
from engine.pipeline.cut_simplifier import finalize_plate_output, count_cut_lines

logger = logging.getLogger("plate_cutting")

def row_sort_repack(
    order_cuts: List[List[Any]],
    board_l: float,
    bt: float,
) -> Optional[List[Cut]]:
    if not order_cuts:
        return []

    Y_TOL = 2.0
    sorted_cuts = sorted(order_cuts, key=lambda c: float(c[1]))
    rows: List[Tuple[float, List[Any]]] = []
    for c in sorted_cuts:
        y1 = float(c[1])
        if rows and abs(y1 - rows[-1][0]) < Y_TOL:
            rows[-1][1].append(c)
        else:
            rows.append((y1, [c]))

    new_cuts: List[Cut] = []
    for row_y_start, row_pieces in rows:
        row_pieces_sorted = sorted(row_pieces, key=lambda c: -float(c[2]))
        total_w = sum(float(c[2]) for c in row_pieces_sorted) + bt * (len(row_pieces_sorted) - 1)
        if total_w > board_l + 1e-6:
            logger.info("row_sort_repack: EXCEEDS limit")
            return None

        x_cur = 0.0
        for c in row_pieces_sorted:
            y1 = float(c[1])
            w = float(c[2])
            h = float(c[3])
            piece_id = c[5] if len(c) > 5 else ""
            sp = SmallPlate(length=int(round(w)), width=int(round(h)), plate_id=str(piece_id))
            nc = Cut(plate=sp, x1=x_cur, y1=y1, x2=x_cur + w, y2=y1 + h, is_stock=False)
            new_cuts.append(nc)
            x_cur += w + bt

    for i in range(len(new_cuts)):
        c1 = new_cuts[i]
        for j in range(i + 1, len(new_cuts)):
            c2 = new_cuts[j]
            if not (c1.x2 <= c2.x1 + 0.1 or c1.x1 >= c2.x2 - 0.1 or c1.y2 <= c2.y1 + 0.1 or c1.y1 >= c2.y2 - 0.1):
                logger.info("row_sort_repack: OVERLAP")
                return None

    return new_cuts

def apply_row_sort_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[List[Dict[str, Any]]]:
    if not results or not plate_templates:
        return None

    bt = float(config.blade_thickness)
    out = list(results)
    any_changed = False

    for idx, rep in enumerate(out):
        pl = rep.get("plate")
        if not pl or len(pl) < 2:
            continue
        board_l = float(pl[0])
        order_cuts = [c for c in rep["cutted"] if c[4] == 0]
        if len(order_cuts) <= 1:
            continue

        new_cuts = row_sort_repack(order_cuts, board_l, bt)
        if new_cuts is None:
            continue

        orig_positions = {(round(c[0], 1), round(c[1], 1)) for c in order_cuts}
        new_positions = {(round(nc.x1, 1), round(nc.y1, 1)) for nc in new_cuts}
        if orig_positions == new_positions:
            continue

        from engine.pipeline.cut_simplifier import _outer_plate_for_finalize
        bp = _outer_plate_for_finalize(rep, bt)
        new_rep = finalize_plate_output(
            bp,
            new_cuts,
            stock_plates,
            stock_optimizer,
            optim,
            config,
            converter,
        )
        # 对于行序重排，为了满足视觉规整（大件靠左），我们允许稍微增加切割线
        # 但如果增加得太离谱（比如 > 5），则拒绝
        lines_new = count_cut_lines(new_rep)
        lines_old = count_cut_lines(rep)
        if lines_new > lines_old + 5:
            logger.info("RowSort rejected due to significantly increased cut lines: %d -> %d", lines_old, lines_new)
            continue
        out[idx] = new_rep
        any_changed = True

    if not any_changed:
        return None
    logger.info("RowSort: %d 张板完成行序重排（大件靠左）。", sum(1 for a, b in zip(results, out) if a is not b))
    return out
