"""专家 Case 2：三规格各 20 件 → 11 张 2440×1220；band-fill 应用前五张 2 大 + 4×806 类条。"""
import rectpack

from services.cutting_service import run_single_algorithm

PLATES = [{"length": 2440, "width": 1220, "quantity": 100}]
ORDERS = [
    {"id": 1, "length": 1014, "width": 814, "quantity": 20},
    {"id": 2, "length": 1006, "width": 350, "quantity": 20},
    {"id": 3, "length": 350, "width": 806, "quantity": 20},
]


def test_case2_eleven_plates_with_band_fill():
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
    assert metrics["used_plates"] == 11
    assert len(results) == 11
    # 前五张：band-fill，每板 6 件（2×1014×814 + 4×806×350 形态）
    for i in range(5):
        cuts = [c for c in results[i]["cutted"] if c[4] == 0]
        assert len(cuts) == 6
        assert results[i]["rate"] > 0.93
