"""单张板输出：订单 cuts + 库存填充 + 转为 API dict。"""
from typing import Any, Dict, List

from core.models import Cut, CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import StockOptimizer


def orders_from_result_orders_only(result: Dict[str, Any]) -> List[SmallPlate]:
    """从切割结果中还原订单小件（忽略库存填充）。"""
    out: List[SmallPlate] = []
    for c in result["cutted"]:
        if c[4] != 0:
            continue
        out.append(
            SmallPlate(
                length=int(c[2]),
                width=int(c[3]),
                plate_id=str(c[5]),
            )
        )
    return out


def finalize_plate_output(
    big_plate_work: SmallPlate,
    order_cuts: List[Cut],
    stock_plates: List[SmallPlate],
    stock_optimizer: StockOptimizer,
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Dict[str, Any]:
    stock_cuts: List[Cut] = []
    if stock_plates:
        stock_cuts = stock_optimizer.fill_with_stock(
            big_plate_work.length,
            big_plate_work.width,
            order_cuts,
            stock_plates,
            optimize=bool(optim),
        )
    all_cuts = order_cuts + stock_cuts
    big_plate_work.length = big_plate_work.length - config.blade_thickness
    big_plate_work.width = big_plate_work.width - config.blade_thickness
    return converter.convert_cuts_to_output(big_plate_work, all_cuts)
