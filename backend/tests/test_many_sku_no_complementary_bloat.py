"""多规格订单：互补排布只适用于少量尺寸组，否则板数会异常膨胀。"""
import rectpack

from services.cutting_service import run_single_algorithm

PLATES = [{"length": 2440, "width": 1220, "quantity": 200}]
# 7 种尺寸（与典型「整柜混单」类似）
ORDERS = [
    {"length": 1200, "width": 870, "quantity": 20},
    {"length": 1200, "width": 940, "quantity": 20},
    {"length": 1200, "width": 525, "quantity": 20},
    {"length": 1200, "width": 415, "quantity": 20},
    {"length": 842, "width": 940, "quantity": 40},
    {"length": 870, "width": 80, "quantity": 60},
    {"length": 1200, "width": 80, "quantity": 60},
]


def test_seven_skus_plate_count_reasonable_with_row_complementary():
    results, metrics = run_single_algorithm(
        PLATES,
        ORDERS,
        [],
        0,
        4.0,
        rectpack.MaxRectsBaf,
        "maxrects_baf",
        enable_row_complementary=True,
    )
    assert metrics["remaining_orders"] == 0
    # 曾错误地对「主尺寸对」做 composite-stack，其余规格堆在队尾，板数 ~70+；
    # 限制尺寸组数后应与关闭互补时同量级（约 40–50 张，留裕度防算法微调）
    assert metrics["used_plates"] < 55
    assert len(results) == metrics["used_plates"]
