"""
OR-Tools 辅助：方案 A（面积约束下的分板 CP-SAT + rectpack 落地）、
方案 B（单张板内 2D 可选矩形 NoOverlap，最大化放置件数）。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

from core.models import Cut, CuttingConfig, SmallPlate

logger = logging.getLogger("plate_cutting")


def _int_dim(x: float) -> int:
    return int(round(float(x)))


def piece_footprint(
    order: SmallPlate, blade: float, rotate: bool
) -> Tuple[int, int]:
    """与 rectpack 一致的占位尺寸（含锯缝）。"""
    bt = _int_dim(blade)
    L = _int_dim(order.length)
    W = _int_dim(order.width)
    if rotate:
        return W + bt, L + bt
    return L + bt, W + bt


def piece_fits_bin(
    order: SmallPlate, blade: float, bin_w: int, bin_h: int
) -> Tuple[bool, bool]:
    """(是否可放, 建议旋转) — 至少一种朝向可放入。"""
    for rot in (False, True):
        pw, ph = piece_footprint(order, blade, rot)
        if pw <= bin_w and ph <= bin_h:
            return True, rot
    return False, False


def solve_area_assignment_to_bins(
    order_areas: Sequence[int],
    bin_capacities: Sequence[int],
    time_limit_sec: float,
) -> Optional[List[int]]:
    """
    每件物品恰好放入一个桶；桶容量为面积上界。最小化「有货的桶数」。
    返回 assignment[i] = bin_index；无解时 None。
    """
    n = len(order_areas)
    m = len(bin_capacities)
    if n == 0:
        return []
    if m == 0:
        return None

    model = cp_model.CpModel()
    x: dict[Tuple[int, int], cp_model.IntVar] = {}
    for i in range(n):
        for b in range(m):
            x[i, b] = model.new_bool_var(f"a_{i}_{b}")
        model.add_exactly_one([x[i, b] for b in range(m)])

    y = [model.new_bool_var(f"yb_{b}") for b in range(m)]
    for b in range(m):
        model.add(
            sum(order_areas[i] * x[i, b] for i in range(n)) <= bin_capacities[b]
        )
        for i in range(n):
            model.add(y[b] >= x[i, b])

    model.minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info("OR-Tools 面积分板无可行解，将回退顺序装箱")
        return None

    assignment: List[int] = [0] * n
    for i in range(n):
        for b in range(m):
            if solver.value(x[i, b]):
                assignment[i] = b
                break
    return assignment


def pack_single_plate_cp2d(
    big_plate: SmallPlate,
    orders: List[SmallPlate],
    config: CuttingConfig,
    pick_rotation_for_index,
    time_limit_sec: float,
) -> Tuple[List[Cut], List[SmallPlate]]:
    """
    单张板：2D 可选矩形 NoOverlap，目标最大化放置件数，其次最大化已放面积。
    pick_rotation_for_index(i) -> bool
    """
    bt = config.blade_thickness
    W = _int_dim(big_plate.length)
    H = _int_dim(big_plate.width)
    n = len(orders)

    placeable: List[Tuple[int, bool, int, int]] = []
    impossible: List[int] = []
    for i in range(n):
        rot = pick_rotation_for_index(i)
        pw, ph = piece_footprint(orders[i], bt, rot)
        if pw > W or ph > H:
            rot = not rot
            pw, ph = piece_footprint(orders[i], bt, rot)
        if pw > W or ph > H:
            impossible.append(i)
        else:
            placeable.append((i, rot, pw, ph))

    if not placeable:
        return [], list(orders)

    model = cp_model.CpModel()
    placed_vars: List[cp_model.IntVar] = []
    ix_intervals: List = []
    iy_intervals: List = []
    sx_vars: List[cp_model.IntVar] = []
    sy_vars: List[cp_model.IntVar] = []
    rot_flags: List[bool] = []

    for k, (orig_i, rot, pw, ph) in enumerate(placeable):
        p = model.new_bool_var(f"cp2d_p_{orig_i}")
        placed_vars.append(p)
        rot_flags.append(rot)
        sx = model.new_int_var(0, max(0, W - pw), f"sx_{orig_i}")
        sy = model.new_int_var(0, max(0, H - ph), f"sy_{orig_i}")
        ex = model.new_int_var(pw, W, f"ex_{orig_i}")
        ey = model.new_int_var(ph, H, f"ey_{orig_i}")
        model.add(ex == sx + pw).only_enforce_if(p)
        model.add(ey == sy + ph).only_enforce_if(p)

        ix = model.new_optional_interval_var(sx, pw, ex, p, f"ix_{orig_i}")
        iy = model.new_optional_interval_var(sy, ph, ey, p, f"iy_{orig_i}")
        ix_intervals.append(ix)
        iy_intervals.append(iy)
        sx_vars.append(sx)
        sy_vars.append(sy)

    model.add_no_overlap_2d(ix_intervals, iy_intervals)

    area_terms = []
    for k, (orig_i, rot, pw, ph) in enumerate(placeable):
        L = _int_dim(orders[orig_i].length)
        Wd = _int_dim(orders[orig_i].width)
        area = L * Wd
        area_terms.append(placed_vars[k] * area)

    scale = W * H + 1
    model.maximize(scale * sum(placed_vars) + sum(area_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], list(orders)

    cuts: List[Cut] = []
    packed_orig: set = set()
    for k, (orig_i, rot, pw, ph) in enumerate(placeable):
        if not solver.value(placed_vars[k]):
            continue
        sx = solver.value(sx_vars[k])
        sy = solver.value(sy_vars[k])
        order = orders[orig_i]
        actual_length = order.width if rot else order.length
        actual_width = order.length if rot else order.width
        cuts.append(
            Cut(
                plate=order,
                x1=float(sx),
                y1=float(sy),
                x2=float(sx + actual_length),
                y2=float(sy + actual_width),
                is_stock=False,
            )
        )
        packed_orig.add(orig_i)

    remaining = [
        orders[i]
        for i in range(n)
        if i not in packed_orig and i not in impossible
    ]
    remaining.extend(orders[i] for i in impossible)
    return cuts, remaining
