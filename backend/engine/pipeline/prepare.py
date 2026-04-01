"""将请求体转为大板/订单/库存 SmallPlate 列表（含锯片扩边）。"""
from typing import Any, Dict, List, Optional, Tuple

from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter

from engine.pipeline.normalize import apply_blade_margin_to_plate_dicts
from engine.pipeline.templates import clone_plate_template


def load_converted_inputs(
    plates: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    others: Optional[List[Dict[str, Any]]],
    config: CuttingConfig,
) -> Tuple[
    DataConverter,
    List[SmallPlate],
    List[SmallPlate],
    List[SmallPlate],
    List[SmallPlate],
]:
    plates0 = apply_blade_margin_to_plate_dicts(plates, config)
    converter = DataConverter()
    big_plates = converter.convert_plates(plates0)
    small_plates = converter.convert_orders(orders)
    stock_plates = converter.convert_stock(others) if others else []
    plate_templates = [clone_plate_template(p) for p in big_plates]
    return (
        converter,
        big_plates,
        plate_templates,
        small_plates,
        stock_plates,
    )


def empty_stock_if_none(
    others: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    return others if others is not None else []
