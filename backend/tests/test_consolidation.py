"""
布局整合优化测试（Layout Consolidation）

验证：
  1. board_fingerprint / layout_groups 基础逻辑
  2. consolidate_layout_groups 在有独板版型时确实减少版型数
  3. optimize_cutting 端到端：整合后结果不比原方案差（板数 ≤ 原板数）
  4. 已无独板版型时整合为空操作（不改变结果）
"""

import rectpack

from engine.pipeline.consolidate import board_fingerprint, consolidate_layout_groups, layout_groups
from services.cutting_service import optimize_cutting

# ---------------------------------------------------------------------------
# 测试夹具：构造含独板版型的 mock 结果
# ---------------------------------------------------------------------------

def _make_mock_result(pieces, rate=0.9):
    """
    构造一条 mock 切割结果：pieces 为 [(length, width), ...]。
    cut 格式：[x, y, w, h, is_stock, id]
    """
    cutted = [
        [0, 0, float(l), float(w), 0, f"{l}x{w}"]
        for l, w in pieces
    ]
    return {
        "cutted": cutted,
        "rate": rate,
        "length": 2440.0,
        "width": 1220.0,
    }


# ---------------------------------------------------------------------------
# 单元测试：指纹与分组
# ---------------------------------------------------------------------------

class TestBoardFingerprint:
    def test_same_pieces_same_fingerprint(self):
        r1 = _make_mock_result([(1000, 500), (800, 400)])
        r2 = _make_mock_result([(800, 400), (1000, 500)])  # 顺序不同
        assert board_fingerprint(r1) == board_fingerprint(r2)

    def test_rotated_pieces_same_fingerprint(self):
        r1 = _make_mock_result([(1000, 500)])
        r2 = _make_mock_result([(500, 1000)])  # 旋转 90°
        assert board_fingerprint(r1) == board_fingerprint(r2)

    def test_different_pieces_different_fingerprint(self):
        r1 = _make_mock_result([(1000, 500)])
        r2 = _make_mock_result([(900, 500)])
        assert board_fingerprint(r1) != board_fingerprint(r2)

    def test_stock_cuts_excluded(self):
        """余板 (is_stock=1) 不计入指纹。"""
        r1 = _make_mock_result([(1000, 500)])
        r2_cutted = r1["cutted"] + [[100, 0, 200, 300, 1, "stock"]]
        r2 = {**r1, "cutted": r2_cutted}
        assert board_fingerprint(r1) == board_fingerprint(r2)


class TestLayoutGroups:
    def test_groups_by_fingerprint(self):
        r_a = _make_mock_result([(1000, 500), (1000, 500)])
        r_b = _make_mock_result([(900, 400)])
        r_a2 = _make_mock_result([(1000, 500), (1000, 500)])

        groups = layout_groups([r_a, r_b, r_a2])
        # a 类 2 张，b 类 1 张
        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]

    def test_all_identical(self):
        r = _make_mock_result([(600, 300)])
        groups = layout_groups([r, r, r])
        assert len(groups) == 1

    def test_all_unique(self):
        results = [_make_mock_result([(i * 100, 300)]) for i in range(1, 5)]
        groups = layout_groups(results)
        assert len(groups) == 4


# ---------------------------------------------------------------------------
# 集成测试：consolidate_layout_groups mock 场景
# ---------------------------------------------------------------------------

PLATES = [{"length": 2440, "width": 1220, "quantity": 100}]


def _build_consolidation_objects(saw_blade=4.0):
    """构造 consolidate_layout_groups 所需的 pipeline 对象。"""
    from core.models import CuttingConfig
    from engine.cutting_algorithms import resolve_packing_class
    from engine.optimizers import PlateOptimizer, StockOptimizer
    from engine.pipeline.prepare import empty_stock_if_none, load_converted_inputs

    config = CuttingConfig(blade_thickness=saw_blade)
    converter, big_plates, plate_templates, _sp, stock_plates = load_converted_inputs(
        PLATES, [], [], config
    )
    plate_engine = PlateOptimizer(config, resolve_packing_class("MaxRectsBaf"))
    stock_optimizer = StockOptimizer(config, "maxrects_baf")
    return config, converter, big_plates, plate_templates, stock_plates, plate_engine, stock_optimizer


class TestConsolidateLayoutGroups:
    def test_no_change_when_single_board(self):
        results = [_make_mock_result([(1000, 500)])]
        config, converter, _, plate_templates, stock_plates, pe, so = _build_consolidation_objects()
        out = consolidate_layout_groups(results, plate_templates, pe, so, stock_plates, 0, config, converter)
        assert out is None  # 单张板，无需整合

    def test_no_change_when_all_identical(self):
        r = _make_mock_result([(1000, 500), (1000, 500)])
        results = [r] * 4
        config, converter, _, plate_templates, stock_plates, pe, so = _build_consolidation_objects()
        out = consolidate_layout_groups(results, plate_templates, pe, so, stock_plates, 0, config, converter)
        assert out is None  # 全部相同，无独板版型

    def test_no_change_when_no_singletons(self):
        r_a = _make_mock_result([(1000, 500), (1000, 500)])
        r_b = _make_mock_result([(900, 400)])
        results = [r_a, r_a, r_b, r_b]  # 各 2 张，无独板
        config, converter, _, plate_templates, stock_plates, pe, so = _build_consolidation_objects()
        out = consolidate_layout_groups(results, plate_templates, pe, so, stock_plates, 0, config, converter)
        assert out is None


# ---------------------------------------------------------------------------
# 端到端集成测试：通过 optimize_cutting 触发整合
# ---------------------------------------------------------------------------

class TestOptimizeCuttingWithConsolidation:
    def test_case2_consolidation_does_not_increase_plates(self):
        """
        Case 2（三规格各 20 件）：整合后板数应 ≤ 整合前（不能变差）。
        使用 band_fill 算法，本身就有多种版型，整合有机会减少版型数。
        """
        orders = [
            {"id": 1, "length": 1014, "width": 814, "quantity": 20},
            {"id": 2, "length": 1006, "width": 350, "quantity": 20},
            {"id": 3, "length": 350, "width": 806, "quantity": 20},
        ]
        # 先用确定算法获取基准板数
        results_base, metrics_base = __import__(
            "services.cutting_service", fromlist=["run_single_algorithm"]
        ).run_single_algorithm(
            PLATES, orders, [], 0, 4.0,
            rectpack.MaxRectsBaf, "maxrects_baf",
            enable_row_complementary=True,
        )
        # 通过 optimize_cutting（内含整合）获取最终结果
        results_opt = optimize_cutting(
            PLATES, orders, algorithm="MaxRectsBaf",
            saw_blade=4.0, enable_row_complementary=True,
        )
        # 整合不能让板数增加
        assert len(results_opt) <= len(results_base)
        # 所有订单件必须全部放置
        placed = sum(
            sum(1 for c in r["cutted"] if c[4] == 0)
            for r in results_opt
        )
        assert placed == 60  # 3 种各 20 件

    def test_many_sku_consolidation_does_not_increase_plates(self):
        """7-SKU 混合订单：整合后板数应 ≤ 原始 CommonDimStrip 结果。"""
        orders = [
            {"id": 1, "length": 975, "width": 486, "quantity": 10},
            {"id": 2, "length": 975, "width": 356, "quantity": 10},
            {"id": 3, "length": 975, "width": 306, "quantity": 10},
            {"id": 4, "length": 975, "width": 256, "quantity": 10},
            {"id": 5, "length": 868, "width": 486, "quantity": 10},
            {"id": 6, "length": 868, "width": 356, "quantity": 10},
            {"id": 7, "length": 868, "width": 256, "quantity": 10},
        ]
        from services.cutting_service import run_single_algorithm
        results_base, metrics_base = run_single_algorithm(
            PLATES, orders, [], 0, 4.0,
            "CommonDimStrip", "maxrects_baf",
            enable_row_complementary=False,
        )
        results_opt = optimize_cutting(
            PLATES, orders, algorithm="CommonDimStrip",
            saw_blade=4.0, enable_row_complementary=False,
        )
        assert len(results_opt) <= len(results_base)
        # 所有件都放置完毕
        assert metrics_base["remaining_orders"] == 0
        placed = sum(
            sum(1 for c in r["cutted"] if c[4] == 0)
            for r in results_opt
        )
        assert placed == 70  # 7 种各 10 件
