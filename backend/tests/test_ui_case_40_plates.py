"""测试 7 个规格（UI Case）的通用 Level Packing（CommonDimStrip）是否能排进 40 张板。"""
from services.cutting_service import optimize_cutting

PLATES = [{"length": 2440, "width": 1220, "quantity": 200}]
ORDERS = [
    {"length": 1200, "width": 870, "quantity": 20},
    {"length": 1200, "width": 940, "quantity": 20},
    {"length": 1200, "width": 525, "quantity": 20},
    {"length": 1200, "width": 415, "quantity": 20},
    {"length": 842, "width": 940, "quantity": 40},
    {"length": 870, "width": 80, "quantity": 60},
    {"length": 1200, "width": 80, "quantity": 60},
]


def test_ui_case_40_plates_generic():
    """验证 CommonDimStrip 能够泛化地识别 1200mm 特征，将问题降维为 1D 装箱并达成专家级的 40 张板。"""
    results = optimize_cutting(
        PLATES,
        ORDERS,
        [],
        0,
        4.0,
        "CommonDimStrip",
        "maxrects_baf",
    )
    assert len(results) == 40
