"""
OR-Tools 与现有切割流水线的衔接：B 方案引擎（单张板 CP-2D）；
A 方案在 services.cutting_service 中调用 ortools_packing 的面积分板 + PlateOptimizer。
"""
from __future__ import annotations

import logging
from typing import Any, List, Tuple

import rectpack

from core.models import Cut, CuttingConfig, SmallPlate
from engine.optimizers import PlateOptimizer
from engine.ortools_packing import pack_single_plate_cp2d

logger = logging.getLogger("plate_cutting")


class ORToolsCP2DEngine:
    """
    单张板内用 CP-SAT 2D NoOverlap 最大化放置件数；超过件数上限时回退 PlateOptimizer。
    不启用行式互补分支（避免与 rectpack 专用逻辑重复）。
    """

    def __init__(
        self,
        config: CuttingConfig,
        time_limit_sec: float,
        max_pieces: int,
        fallback_algorithm: Any = rectpack.MaxRectsBaf,
    ):
        self.config = config
        self.time_limit_sec = time_limit_sec
        self.max_pieces = max_pieces
        self._fallback = PlateOptimizer(config, fallback_algorithm)

    def pack_orders(
        self,
        big_plate: SmallPlate,
        orders: List[SmallPlate],
    ) -> Tuple[List[Cut], List[SmallPlate]]:
        if not orders:
            return [], []

        if len(orders) > self.max_pieces:
            logger.debug(
                "ORToolsCP2D: 件数 %d > 上限 %d，回退 rectpack",
                len(orders),
                self.max_pieces,
            )
            return self._fallback.pack_orders(big_plate, orders)

        def pick_rot(i: int) -> bool:
            return self._fallback._pick_rotation(orders[i], big_plate)

        cuts, rem = pack_single_plate_cp2d(
            big_plate,
            orders,
            self.config,
            pick_rot,
            self.time_limit_sec,
        )
        if not cuts and orders:
            logger.debug("ORToolsCP2D 无解，回退 rectpack")
            return self._fallback.pack_orders(big_plate, orders)
        return cuts, rem
