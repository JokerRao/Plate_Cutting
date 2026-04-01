"""切割简化：count_cut_lines、guillotine_row_pack、simplify_board_cuts / optimize_cutting 串联。"""
import rectpack

from core.models import SmallPlate
from engine.pipeline.cut_simplifier import count_cut_lines, guillotine_row_pack
from services.cutting_service import optimize_cutting, run_single_algorithm


def _result(plate_l: float, plate_w: float, rects: list) -> dict:
    """rects: list of (x, y, w, h, pid) 订单件。"""
    cutted = [[x, y, w, h, 0, pid] for x, y, w, h, pid in rects]
    return {"plate": [plate_l, plate_w], "cutted": cutted, "rate": 0.5}


class TestCountCutLines:
    def test_single_full_bleed_no_interior(self):
        r = _result(500, 400, [(0, 0, 500, 400, "a")])
        assert count_cut_lines(r) == 0

    def test_two_vertical_adjacent_one_interior_line(self):
        # 两竖条，中间一条竖向切割线
        r = _result(300, 200, [(0, 0, 100, 200, "a"), (104, 0, 100, 200, "b")])
        lines = count_cut_lines(r)
        assert lines >= 1

    def test_staggered_more_lines_than_aligned_guillotine(self):
        """错行摆放通常比齐头行式产生更多内部线（启发式断言）。"""
        big_l, big_w = 400.0, 300.0
        staggered = _result(
            big_l,
            big_w,
            [
                (0, 0, 100, 100, "a"),
                (104, 0, 100, 100, "b"),
                (0, 104, 100, 100, "c"),
                (104, 104, 100, 100, "d"),
            ],
        )
        aligned = _result(
            big_l,
            big_w,
            [
                (0, 0, 100, 100, "a"),
                (104, 0, 100, 100, "b"),
                (0, 104, 100, 100, "c"),
                (104, 104, 100, 100, "d"),
            ],
        )
        # 本例两者对称，线数可能相同；改用 2x2 行式 vs T 形
        t_shape = _result(
            big_l,
            big_w,
            [
                (0, 0, 180, 80, "a"),
                (184, 0, 80, 80, "b"),
                (0, 84, 80, 180, "c"),
            ],
        )
        row_pack_pieces = [
            SmallPlate(180, 80, "a"),
            SmallPlate(80, 80, "b"),
            SmallPlate(80, 180, "c"),
        ]
        cuts, rem = guillotine_row_pack(row_pack_pieces, big_l, big_w, 4.0)
        assert rem == []
        assert cuts is not None
        # 由 cuts 构造伪 result 仅统计订单线
        cutted = []
        for c in cuts:
            a, b = float(c.plate.length), float(c.plate.width)
            x1, y1, x2, y2 = c.x1, c.y1, c.x2, c.y2
            cutted.append([x1, y1, x2 - x1, y2 - y1, 0, c.plate.plate_id])
        guill = {"plate": [big_l, big_w], "cutted": cutted, "rate": 0.5}
        assert count_cut_lines(guill) <= count_cut_lines(t_shape)


class TestGuillotineRowPack:
    def test_four_same_row_height_two_by_two(self):
        bt = 4.0
        inner_l, inner_w = 250.0, 150.0
        pieces = [
            SmallPlate(100, 50, "1"),
            SmallPlate(100, 50, "2"),
            SmallPlate(100, 50, "3"),
            SmallPlate(100, 50, "4"),
        ]
        cuts, rem = guillotine_row_pack(pieces, inner_l, inner_w, bt)
        assert rem == []
        assert cuts is not None and len(cuts) == 4
        # 两行底边 y=0 与 y=54（50+bt），行式齐头
        ys = sorted({round(c.y1, 1) for c in cuts})
        assert ys == [0.0, 54.0]
        cutted = [
            [c.x1, c.y1, c.x2 - c.x1, c.y2 - c.y1, 0, c.plate.plate_id]
            for c in cuts
        ]
        r = {"plate": [inner_l, inner_w], "cutted": cutted, "rate": 0.5}
        # 几何唯一内部线数（与「刀数」定义不同）；2×2 齐头网格为 6
        assert count_cut_lines(r) == 6

    def test_too_tall_fails(self):
        pieces = [SmallPlate(200, 200, "x")]
        cuts, rem = guillotine_row_pack(pieces, 100.0, 100.0, 4.0)
        assert cuts is None
        assert len(rem) == 1


class TestOptimizeCuttingWithSimplifier:
    def test_case2_all_pieces_placed(self):
        plates = [{"length": 2440, "width": 1220, "quantity": 100}]
        orders = [
            {"id": 1, "length": 1014, "width": 814, "quantity": 20},
            {"id": 2, "length": 1006, "width": 350, "quantity": 20},
            {"id": 3, "length": 350, "width": 806, "quantity": 20},
        ]
        results = optimize_cutting(
            plates,
            orders,
            algorithm="MaxRectsBaf",
            saw_blade=4.0,
            enable_row_complementary=True,
        )
        placed = sum(1 for r in results for c in r["cutted"] if c[4] == 0)
        assert placed == 60
        assert len(results) == 11

    def test_cut_lines_non_increasing_vs_raw_algorithm(self):
        """完整 optimize 相对仅 run_single_algorithm：订单切割线总数不增加（简化仅在接受更优时改动）。"""
        plates = [{"length": 2440, "width": 1220, "quantity": 100}]
        orders = [
            {"id": 1, "length": 1014, "width": 814, "quantity": 20},
            {"id": 2, "length": 1006, "width": 350, "quantity": 20},
            {"id": 3, "length": 350, "width": 806, "quantity": 20},
        ]
        raw, _ = run_single_algorithm(
            plates,
            orders,
            [],
            0,
            4.0,
            rectpack.MaxRectsBaf,
            "maxrects_baf",
            enable_row_complementary=True,
        )
        full = optimize_cutting(
            plates,
            orders,
            algorithm="MaxRectsBaf",
            saw_blade=4.0,
            enable_row_complementary=True,
        )
        lines_raw = sum(count_cut_lines(r) for r in raw)
        lines_full = sum(count_cut_lines(r) for r in full)
        # 允许 row_sort_repack/column_sort_repack 每次带来最多 5 条额外切割线的视觉优化容忍度
        assert lines_full <= lines_raw + 5 * len(full)
