"""请求体大板尺寸：按锯片厚度扩展（与原有 cutting_service 一致）。"""
from typing import Any, Dict, List

from core.models import CuttingConfig


def apply_blade_margin_to_plate_dicts(
    plates: List[Dict[str, Any]],
    config: CuttingConfig,
) -> List[Dict[str, Any]]:
    plates0 = [{**plate} for plate in plates]
    for plate_data in plates0:
        quantity = plate_data.get("quantity", 0)
        if quantity > 0:
            plate_data["length"] = plate_data["length"] + config.blade_thickness
            plate_data["width"] = plate_data["width"] + config.blade_thickness
    return plates0
