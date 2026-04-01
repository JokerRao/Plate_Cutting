"""大板订单装箱引擎协议：rectpack / OR-Tools 等实现统一接口，便于 refine 与流水线注入。"""
from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable

from core.models import Cut, SmallPlate


@runtime_checkable
class PlatePackingEngine(Protocol):
    """
    单张大板上的订单装箱（可含行式互补、rectpack 多序、CP-2D 等实现细节）。

    约定：
    - 可修改 big_plate 的展示尺寸（如 finalize 前扣锯片由 pipeline 统一处理）；
    - pack_orders 语义与 PlateOptimizer.pack_orders 一致。
    """

    def pack_orders(
        self,
        big_plate: SmallPlate,
        orders: List[SmallPlate],
    ) -> Tuple[List[Cut], List[SmallPlate]]:
        ...
