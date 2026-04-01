"""
切割简化（Cut Simplifier Pass）

执行时机：布局整合（Consolidation）之后、对外返回前；与 Consolidation 相同，
         先抽取订单件重排，再通过 finalize_plate_output 重新放置余板。

目标：在版型与件数不变的前提下，将每张板上的订单件重排为行式齐头布局
      （同行走水平方向、行间可视为 Guillotine 分割），使内部切割线数量
      （唯一内部水平线 + 唯一内部垂直线）尽可能减少，便于现场下料。

仅统计订单件（is_stock=0）的几何边界；余板不参与切割线统计。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.models import Cut, CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import StockOptimizer
from engine.pipeline.consolidate import layout_groups
from engine.pipeline.output import finalize_plate_output, orders_from_result_orders_only

logger = logging.getLogger("plate_cutting")


def count_cut_lines(result: Dict[str, Any]) -> int:
    """
    统计单张板上订单件产生的内部切割线数量（唯一内部水平线 + 唯一内部垂直线）。

    边界：排除贴大板四边的线（与 plate 内尺寸比较，容差 eps）。
    """
    order_cuts = [c for c in result["cutted"] if c[4] == 0]
    if not order_cuts:
        return 0
    plate = result.get("plate") or [0, 0]
    L = float(plate[0])
    W = float(plate[1])
    eps = 0.5
    y_int: set = set()
    x_int: set = set()
    for c in order_cuts:
        x0, y0 = float(c[0]), float(c[1])
        w, h = float(c[2]), float(c[3])
        x1, y1 = x0 + w, y0 + h
        for y in (round(y0, 1), round(y1, 1)):
            if eps < y < W - eps:
                y_int.add(y)
        for x in (round(x0, 1), round(x1, 1)):
            if eps < x < L - eps:
                x_int.add(x)
    return len(y_int) + len(x_int)


def _piece_spans(order: SmallPlate) -> Tuple[float, float]:
    """行向跨度（较大边沿 X）与行高（较小边沿 Y），均为几何尺寸不含锯缝。"""
    a, b = float(order.length), float(order.width)
    return max(a, b), min(a, b)


def _sort_pieces_deterministic(pieces: List[SmallPlate]) -> List[SmallPlate]:
    """行高降序、行宽降序、plate_id 保证稳定顺序。"""
    return sorted(
        pieces,
        key=lambda p: (
            -_piece_spans(p)[1],
            -_piece_spans(p)[0],
            str(p.plate_id),
        ),
    )

def _sort_row_pieces_deterministic(pieces: List[SmallPlate]) -> List[SmallPlate]:
    """同一行内按宽降序排列，使最长边靠左侧放置，保证空白留在右侧。"""
    return sorted(
        pieces,
        key=lambda p: (
            -_piece_spans(p)[0],
            str(p.plate_id),
        ),
    )


def guillotine_row_pack(
    pieces: List[SmallPlate],
    board_l: float,
    board_w: float,
    bt: float,
) -> Tuple[Optional[List[Cut]], List[SmallPlate]]:
    """
    行式齐头装箱：同行走同一行高（min 边），行内从左向右排较大边沿 X，行间加锯缝。

    Returns:
        (cuts, []) 成功；
        (None, pieces) 无法装入（含单件超限或总高超限）。
    """
    if not pieces:
        return [], []

    sorted_pieces = _sort_pieces_deterministic(pieces)
    for p in sorted_pieces:
        sx, sy = _piece_spans(p)
        if sx > board_l + 1e-6 or sy > board_w + 1e-6:
            return None, pieces

    rows: List[Tuple[float, List[SmallPlate]]] = []
    current: List[SmallPlate] = []
    row_h: Optional[float] = None
    row_w_acc = 0.0

    def flush() -> None:
        nonlocal current, row_h, row_w_acc
        if current and row_h is not None:
            rows.append((row_h, _sort_row_pieces_deterministic(current)))
        current = []
        row_h = None
        row_w_acc = 0.0

    for p in sorted_pieces:
        sx, sy = _piece_spans(p)
        if not current:
            row_h = sy
            current = [p]
            row_w_acc = sx
            continue

        assert row_h is not None
        same_h = abs(sy - row_h) <= 1e-6
        fits = row_w_acc + bt + sx <= board_l + 1e-6
        if same_h and fits:
            current.append(p)
            row_w_acc += bt + sx
        else:
            flush()
            row_h = sy
            current = [p]
            row_w_acc = sx

    flush()

    total_h = sum(rh for rh, _ in rows) + bt * max(0, len(rows) - 1)
    if total_h > board_w + 1e-6:
        return None, pieces

    def get_row_width(row_pieces: List[SmallPlate]) -> float:
        w = sum(_piece_spans(p)[0] + bt for p in row_pieces)
        return max(0.0, w - bt)

    # 排序：行高降序，行宽降序。
    # 优先将行高最高（即包含最大件）的行放在最底部 (y=0)，同高的行优先排宽的；
    # 行内已按零件宽度降序（最长件靠左）。
    # 这样确保大块板材被集中严格放置在左下角，而右上的剩余空间（空白区域）
    # 依然能形成相对连贯的大区域，便于后续放置整块的大余板。
    rows.sort(key=lambda r: (-r[0], -get_row_width(r[1])))

    cuts: List[Cut] = []
    y = 0.0
    for rh, row_pieces in rows:
        x = 0.0
        for p in row_pieces:
            sx, sy = _piece_spans(p)
            a, b = float(p.length), float(p.width)
            if a >= b:
                cuts.append(
                    Cut(
                        plate=p,
                        x1=x,
                        y1=y,
                        x2=x + a,
                        y2=y + b,
                        is_stock=False,
                    )
                )
            else:
                cuts.append(
                    Cut(
                        plate=p,
                        x1=x,
                        y1=y,
                        x2=x + b,
                        y2=y + a,
                        is_stock=False,
                    )
                )
            x += sx + bt
        y += rh + bt

    return cuts, []


def column_sort_repack(
    order_cuts: List[List[Any]],
    board_w: float,
    bt: float,
) -> Optional[List[Cut]]:
    """
    列序重排：保留现有列结构，仅将每列内的订单件按件高降序重新堆叠（最高件压底 y=0）。

    算法：
    1. 按 x1 值将件分组为列（x1 相差在 2mm 容差内的视为同一列）。
    2. 每列内按件高降序排序，从 y=0 向上累积放置。
    3. 若任一列堆叠后总高超过 board_w，整板跳过（返回 None）。
    4. 件的 x 坐标保持原值不变，只改变 y 坐标。

    Returns:
        新的 Cut 列表（成功）；None（任意列超限则整板跳过）。
    """
    if not order_cuts:
        return []

    # 按 x1 升序排列后，x1 相差 < X_TOL 的件归为同一列
    X_TOL = 2.0
    sorted_cuts = sorted(order_cuts, key=lambda c: float(c[0]))
    columns: List[Tuple[float, List[Any]]] = []  # (col_x_start, [cut_tuples])
    for c in sorted_cuts:
        x1 = float(c[0])
        if columns and abs(x1 - columns[-1][0]) < X_TOL:
            columns[-1][1].append(c)
        else:
            columns.append((x1, [c]))

    # 每列内按件高降序排序，然后重新堆叠
    new_cuts: List[Cut] = []
    for col_x_start, col_pieces in columns:
        col_pieces_sorted = sorted(col_pieces, key=lambda c: -float(c[3]))
        total_h = (
            sum(float(c[3]) for c in col_pieces_sorted)
            + bt * (len(col_pieces_sorted) - 1)
        )
        if total_h > board_w + 1e-6:
            return None  # 超限，整板跳过

        y_cur = 0.0
        for c in col_pieces_sorted:
            x1 = float(c[0])
            w = float(c[2])
            h = float(c[3])
            piece_id = c[5] if len(c) > 5 else ""
            sp = SmallPlate(length=int(round(w)), width=int(round(h)), plate_id=str(piece_id))
            new_cuts.append(Cut(plate=sp, x1=x1, y1=y_cur, x2=x1 + w, y2=y_cur + h, is_stock=False))
            y_cur += h + bt

    # 严格检测是否发生了重叠 (Overlap detection)
    for i in range(len(new_cuts)):
        c1 = new_cuts[i]
        for j in range(i + 1, len(new_cuts)):
            c2 = new_cuts[j]
            # 如果两个矩形相交 (由于有浮点误差，留 0.1 容差)
            if not (c1.x2 <= c2.x1 + 0.1 or c1.x1 >= c2.x2 - 0.1 or c1.y2 <= c2.y1 + 0.1 or c1.y1 >= c2.y2 - 0.1):
                return None  # 发生重叠，整板跳过

    return new_cuts


def apply_column_sort_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[List[Dict[str, Any]]]:
    """
    对所有板逐张做列序重排。

    若某张板的件序发生变化（即存在条料 → 大件之类的位置对调），则用
    finalize_plate_output 重新填充余板，再替换原结果。

    Returns:
        至少一张板发生改变时返回新 results，否则 None。
    """
    if not results or not plate_templates:
        return None

    bt = float(config.blade_thickness)
    out = list(results)
    any_changed = False

    for idx, rep in enumerate(out):
        pl = rep.get("plate")
        if not pl or len(pl) < 2:
            continue
        board_w = float(pl[1])
        order_cuts = [c for c in rep["cutted"] if c[4] == 0]
        if len(order_cuts) <= 1:
            continue

        new_cuts = column_sort_repack(order_cuts, board_w, bt)
        if new_cuts is None:
            continue  # 某列超限，跳过

        # 判断位置是否有变化（任意件的 y1 改变即视为改变）
        orig_positions = {(round(c[0], 1), round(c[1], 1)) for c in order_cuts}
        new_positions = {(round(nc.x1, 1), round(nc.y1, 1)) for nc in new_cuts}
        if orig_positions == new_positions:
            continue  # 无变化

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
        # 列序重排也是为了大件压底的规整，同样允许稍微增加切割线
        lines_new = count_cut_lines(new_rep)
        lines_old = count_cut_lines(rep)
        if lines_new > lines_old + 5:
            continue
        out[idx] = new_rep
        any_changed = True

    if not any_changed:
        return None
    logger.info("ColumnSort: %d 张板完成列序重排（大件压底）。", sum(1 for a, b in zip(results, out) if a is not b))
    return out


def _outer_plate_for_finalize(
    rep: Dict[str, Any],
    bt: float,
) -> SmallPlate:
    """由结果中的内尺寸 + 锯厚还原 finalize 所需的外尺寸大板。"""
    plate = rep["plate"]
    inner_l = float(plate[0])
    inner_w = float(plate[1])
    return SmallPlate(
        length=int(round(inner_l + bt)),
        width=int(round(inner_w + bt)),
        plate_id="",
        quantity=1,
    )


def simplify_board_cuts(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[List[Dict[str, Any]]]:
    """
    按版型分组：代表板试排若切割线严格减少，再对同组每张板单独验证；
    全部严格减少时才整组替换并 re-finalize。

    Returns:
        至少一版型有改进时的新 results，否则 None。
    """
    if not results or not plate_templates:
        return None

    bt = float(config.blade_thickness)
    out = list(results)
    any_improved = False

    for _fp, idxs in layout_groups(results).items():
        rep_idx = idxs[0]
        rep = results[rep_idx]
        pieces = orders_from_result_orders_only(rep)
        if len(pieces) <= 1:
            continue

        plate = rep.get("plate")
        if not plate or len(plate) < 2:
            continue
        inner_l, inner_w = float(plate[0]), float(plate[1])

        cuts_before_rep = count_cut_lines(rep)

        order_cuts_rep, rem = guillotine_row_pack(
            _sort_pieces_deterministic(pieces), inner_l, inner_w, bt,
        )
        if order_cuts_rep is None or rem:
            continue

        bp_try = _outer_plate_for_finalize(rep, bt)
        try_row = finalize_plate_output(
            bp_try,
            order_cuts_rep,
            stock_plates,
            stock_optimizer,
            optim,
            config,
            converter,
        )
        if count_cut_lines(try_row) > cuts_before_rep:
            continue

        new_for_group: Dict[int, Dict[str, Any]] = {}
        group_ok = True
        for j in idxs:
            rj = results[j]
            plj = rj.get("plate")
            if not plj or len(plj) < 2:
                group_ok = False
                break
            inner_lj, inner_wj = float(plj[0]), float(plj[1])
            pj = _sort_pieces_deterministic(orders_from_result_orders_only(rj))
            oc, r2 = guillotine_row_pack(pj, inner_lj, inner_wj, bt)
            if oc is None or r2:
                group_ok = False
                break
            bp_j = _outer_plate_for_finalize(rj, bt)
            cand = finalize_plate_output(
                bp_j,
                oc,
                stock_plates,
                stock_optimizer,
                optim,
                config,
                converter,
            )
            if count_cut_lines(cand) > count_cut_lines(rj):
                group_ok = False
                break
            new_for_group[j] = cand

        if not group_ok:
            continue

        logger.info(
            "CutSimplifier: 版型 rep_idx=%d 切割线 %d→%d，整组 %d 张应用行式重排。",
            rep_idx,
            cuts_before_rep,
            count_cut_lines(try_row),
            len(idxs),
        )
        for j, row in new_for_group.items():
            out[j] = row
        any_improved = True

    if not any_improved:
        return None
    return out
