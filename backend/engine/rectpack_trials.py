"""单张板 rectpack 多序试探（从 PlateOptimizer 抽出，由 PlateOptimizer 委托）。"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from core.models import Cut, SmallPlate

if TYPE_CHECKING:
    from engine.optimizers import PlateOptimizer

logger = logging.getLogger("plate_cutting")


class RectpackTrialRunner:
    def __init__(self, optimizer: PlateOptimizer):
        self._opt = optimizer

    def rectpack_with_sorted_orders(
        self,
        big_plate: SmallPlate,
        orders: List[SmallPlate],
        sorted_orders: List[Tuple[int, SmallPlate, bool]],
    ) -> Tuple[List[Cut], List[SmallPlate]]:
        bt = self._opt.config.blade_thickness
        packer = self._opt.create_packer(big_plate.length, big_plate.width)

        for orig_idx, order, should_rotate in sorted_orders:
            x1 = order.length + bt
            x2 = order.width + bt
            if should_rotate:
                packer.add_rect(x2, x1, rid=orig_idx)
            else:
                packer.add_rect(x1, x2, rid=orig_idx)

        packer.pack()

        cuts: List[Cut] = []
        packed_indices: set = set()

        for bin_data in packer:
            for rect in bin_data:
                try:
                    x = rect.x
                    y = rect.y
                    w = rect.width
                    h = rect.height
                    rid = None
                    for attr_name in ("rid", "id", "rect_id", "tag"):
                        if hasattr(rect, attr_name):
                            rid = getattr(rect, attr_name)
                            break
                    if rid is None:
                        rid = self._opt._find_matching_order_index(
                            orders,
                            w - self._opt.config.blade_thickness,
                            h - self._opt.config.blade_thickness,
                            packed_indices,
                        )
                except AttributeError as e:
                    logger.warning("Error accessing Rectangle attributes: %s", e)
                    continue

                if rid is None or rid in packed_indices:
                    continue

                order = orders[rid]
                rotated = (w - self._opt.config.blade_thickness != order.length)
                actual_length = order.width if rotated else order.length
                actual_width = order.length if rotated else order.width

                cuts.append(
                    Cut(
                        plate=order,
                        x1=x,
                        y1=y,
                        x2=x + actual_length,
                        y2=y + actual_width,
                        is_stock=False,
                    )
                )
                packed_indices.add(rid)

        remaining = [orders[i] for i in range(len(orders)) if i not in packed_indices]
        return cuts, remaining

    def score_rectpack_trial(
        self, cuts: List[Cut], big_plate: SmallPlate
    ) -> Tuple[int, float]:
        bt = self._opt.config.blade_thickness
        ln = big_plate.length - bt
        wd = big_plate.width - bt
        denom = float(ln * wd) if ln > 0 and wd > 0 else 1.0
        used = sum((c.x2 - c.x1) * (c.y2 - c.y1) for c in cuts)
        return (len(cuts), used / denom)

    def collect_rectpack_sort_variants(
        self,
        orders: List[SmallPlate],
        big_plate: SmallPlate,
        primary_sorted: List[Tuple[int, SmallPlate, bool]],
    ) -> List[List[Tuple[int, SmallPlate, bool]]]:
        seen: set = set()
        variants: List[List[Tuple[int, SmallPlate, bool]]] = []

        def add_variant(tuples: List[Tuple[int, SmallPlate, bool]]) -> None:
            sig = tuple(t[0] for t in tuples)
            if sig not in seen:
                seen.add(sig)
                variants.append(tuples)

        add_variant(primary_sorted)
        n = len(orders)
        if n <= 1:
            return variants

        idx_specs = [
            sorted(range(n), key=lambda i: -orders[i].area),
            sorted(range(n), key=lambda i: -max(orders[i].length, orders[i].width)),
            sorted(range(n), key=lambda i: -min(orders[i].length, orders[i].width)),
            sorted(range(n), key=lambda i: -(orders[i].length + orders[i].width)),
            sorted(range(n), key=lambda i: orders[i].area),
            list(reversed(range(n))),
        ]
        for idxs in idx_specs:
            add_variant(self._opt._indices_to_sorted_tuples(orders, idxs, big_plate))

        return variants

    def best_rectpack_for_bin(
        self,
        orders: List[SmallPlate],
        big_plate: SmallPlate,
        primary_sorted: List[Tuple[int, SmallPlate, bool]],
    ) -> Tuple[List[Cut], List[SmallPlate]]:
        variants = self.collect_rectpack_sort_variants(
            orders, big_plate, primary_sorted
        )
        best_cuts: Optional[List[Cut]] = None
        best_remaining: Optional[List[SmallPlate]] = None
        best_score: Tuple[int, float] = (-1, -1.0)

        for sorted_orders in variants:
            cuts, remaining = self.rectpack_with_sorted_orders(
                big_plate, orders, sorted_orders
            )
            score = self.score_rectpack_trial(cuts, big_plate)
            if score > best_score:
                best_score = score
                best_cuts = cuts
                best_remaining = remaining

        assert best_cuts is not None and best_remaining is not None
        if len(variants) > 1 and best_score[0] > 0:
            logger.debug(
                "Rectpack sort trials: %d variants, best pieces=%d util=%.4f",
                len(variants),
                best_score[0],
                best_score[1],
            )
        return best_cuts, best_remaining
