"""互补尺寸对检测（从 PlateOptimizer 抽出，便于单测与阅读）。"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("plate_cutting")


def band_fill_details_if_applicable(
    pair: Tuple[int, int, int, int],
    L: int,
    W: int,
    bt: float,
) -> Optional[Dict]:
    """
    「双大件横排 + 底条(3) + 中缝(1)」几何：底部三条长条，上方两张大件夹一道竖向条料。

    仅当 (pair 中面积较大者为 big、较小者为 strip) 且尺寸满足板长板宽时返回 pattern dict。
    """
    w1, h1, w2, h2 = pair
    a1, a2 = w1 * h1, w2 * h2
    if a1 >= a2:
        big_k, strip_k = (w1, h1), (w2, h2)
    else:
        big_k, strip_k = (w2, h2), (w1, h1)

    bw, bh = big_k
    sw, sh = strip_k
    strip_long = max(sw, sh)
    strip_short = min(sw, sh)

    # 底部横条：每条占 strip_long 宽度，计算能放几条
    # n_bottom * strip_long - bt <= L
    n_bottom = int((L + bt) / strip_long)
    if n_bottom < 1 or n_bottom * strip_long - bt > L + 0.5:
        return None

    # 上半区：n_big 大件 + (n_big-1) 中缝条料，布局为 [大][条][大][条]...[大]
    # n_big * bw + (n_big-1) * strip_short - bt <= L
    # => n_big * (bw + strip_short) <= L + bt + strip_short
    if bw + strip_short <= 0:
        return None
    n_big = int((L + bt + strip_short) / (bw + strip_short))
    # Ensure the formula holds exactly
    while n_big > 0 and n_big * bw + max(0, n_big - 1) * strip_short - bt > L + 0.5:
        n_big -= 1
    if n_big < 1:
        return None
    n_mid = n_big - 1  # middle strips between big pieces

    # 纵：底条高度方向为 strip_short，再上 max(大高, 中缝竖向长边)
    if strip_short + max(bh, strip_long) - bt > W + 0.5:
        return None

    raw_big_area = (bw - bt) * (bh - bt)
    raw_strip_area = (strip_long - bt) * (strip_short - bt)
    placed_area = n_big * raw_big_area + (n_bottom + n_mid) * raw_strip_area
    util = placed_area / (L * W) if L * W > 0 else 0.0

    return {
        "type": "band_fill",
        "big_key": big_k,
        "strip_key": strip_k,
        "n_big": n_big,
        "n_bottom": n_bottom,
        "n_mid": n_mid,
        "util": util,
    }


def find_complementary_pairs(
    size_groups: Dict,
    L: int,
    W: int,
) -> Tuple[Dict, Dict]:
    """
    找到能够更好组合的尺寸对（包含自旋转优化），解决混合组合优于单一排列的问题。

    Returns:
        (complementary_dict, pattern_details_dict)
    """
    sizes = list(size_groups.keys())
    complementary: Dict = {}
    pattern_details: Dict = {}

    for i, (w1, h1) in enumerate(sizes):
        single_count = (L // w1) * (W // h1)
        single_util = single_count * w1 * h1 / (L * W) if L * W > 0 else 0

        # Construct targets avoiding redundant checks, plus self rotation
        targets = []
        for w2, h2 in sizes[i:]:
            if (w1, h1) != (w2, h2):
                targets.append((w2, h2))
                
        # Inject self-rotation to test perfect fits like 4 rows horizontal + 1 row vertical
        if w1 != h1:
            targets.append((h1, w1))

        for w2, h2 in targets:
            best_mixed = 0
            best_strategy = None
            best_pattern = None

            # 1. Row-based (interleave in the same row if heights match)
            if abs(h1 - h2) < 1:
                num_rows = int(W // h1)
                if num_rows > 0:
                    max_count1 = int(L // w1) + 1
                    for count1 in range(max_count1):
                        remaining_width = L - count1 * w1
                        count2 = int(remaining_width // w2)
                        area_per_row = count1 * w1 * h1 + count2 * w2 * h2
                        total_area = area_per_row * num_rows
                        row_util = total_area / (L * W) if L * W > 0 else 0
                        if row_util > best_mixed:
                            best_mixed = row_util
                            best_strategy = (
                                f"row-based: {count1}×size1 + {count2}×size2 per row, "
                                f"{num_rows} rows"
                            )
                            best_pattern = {
                                "type": "row",
                                "count1": count1,
                                "count2": count2,
                                "rows": num_rows,
                            }

            # 2. Column-based (split the width into columns of w1 and columns of w2)
            max_n1 = max(1, int(L // w1))
            for n1 in range(1, max_n1):
                used_w = n1 * w1
                remaining = L - used_w
                n2 = int(remaining // w2)
                if n2 == 0:
                    continue
                area1 = n1 * w1 * int(W // h1) * h1
                area2 = n2 * w2 * int(W // h2) * h2
                col_util = (area1 + area2) / (L * W) if L * W > 0 else 0
                if col_util > best_mixed:
                    best_mixed = col_util
                    best_strategy = (
                        f"column-based: {n1} cols size1 + {n2} cols size2"
                    )
                    best_pattern = {
                        "type": "column",
                        "count1": n1,
                        "count2": n2,
                    }

            # 3. Layer-based (split the height into rows of h1 and rows of h2)
            max_r1 = max(1, int(W // h1))
            for r1 in range(1, max_r1):
                used_h = r1 * h1
                remaining_h = W - used_h
                r2 = int(remaining_h // h2)
                if r2 == 0:
                    continue
                c1 = int(L // w1)
                c2 = int(L // w2)
                area1 = r1 * c1 * w1 * h1
                area2 = r2 * c2 * w2 * h2
                layer_util = (area1 + area2) / (L * W) if L * W > 0 else 0
                if layer_util > best_mixed:
                    best_mixed = layer_util
                    best_strategy = (
                        f"layer-based: {r1} rows of size1 + {r2} rows of size2"
                    )
                    best_pattern = {
                        "type": "layer",
                        "r1": r1,
                        "r2": r2,
                        "c1": c1,
                        "c2": c2,
                    }

            # 4. Composite-stack (stack 1 size1 and 1 size2 vertically into a composite column)
            combinations = [
                (w1, h1, w2, h2, "N", "N"),
                (w1, h1, h2, w2, "N", "R"),
                (h1, w1, w2, h2, "R", "N"),
                (h1, w1, h2, w2, "R", "R")
            ]
            for c_w1, c_h1, c_w2, c_h2, rot1, rot2 in combinations:
                col_h = c_h1 + c_h2
                if col_h <= W: # Can stack vertically
                    col_w = max(c_w1, c_w2)
                    num_cols = int(L // col_w)
                    if num_cols > 0:
                        tail_gap_w = L - num_cols * col_w
                        bottom_strip_h = W - col_h

                        # Count size2 pieces that fit:
                        # (a) In the right tail gap (full board height)
                        fit2_tail_a = max(
                            int(tail_gap_w // w2) * int(W // h2),
                            int(tail_gap_w // h2) * int(W // w2)
                        ) if tail_gap_w > 0 else 0

                        # (b) In the bottom strip below all columns (full board width)
                        fit2_bottom = max(
                            int(L // w2) * int(bottom_strip_h // h2),
                            int(L // h2) * int(bottom_strip_h // w2)
                        ) if bottom_strip_h > 0 else 0

                        # (c) In the small-side gap within each column above c_w2
                        # Each column has width col_w but size2 only uses c_w2.
                        # The gap per col is col_w - c_w2, height is c_h2.
                        side_gap_per_col = col_w - c_w2
                        fit2_side = (
                            int(side_gap_per_col // w2) * int(c_h2 // h2) +
                            int(side_gap_per_col // h2) * int(c_h2 // w2)
                        ) * num_cols if side_gap_per_col > 0 else 0

                        total_fit2_extra = max(fit2_tail_a, fit2_bottom) + fit2_side
                        area1 = num_cols * w1 * h1
                        area2 = (num_cols + total_fit2_extra) * w2 * h2
                        comp_util = (area1 + area2) / (L * W) if L * W > 0 else 0
                        if comp_util > best_mixed:
                            best_mixed = comp_util
                            best_strategy = (
                                f"composite-stack: {num_cols} cols of ({c_w1}x{c_h1} + {c_w2}x{c_h2}) "
                                f"with bottom_strip={fit2_bottom} tail={fit2_tail_a} side={fit2_side} extra"
                            )
                            best_pattern = {
                                "type": "composite",
                                "col_w": col_w,
                                "c_w1": c_w1,
                                "c_h1": c_h1,
                                "c_w2": c_w2,
                                "c_h2": c_h2,
                                "num_cols": num_cols,
                                "rot1": rot1,
                                "rot2": rot2,
                                "util": comp_util,
                            }

            # Commit the best mixed pattern if it outperforms pure stacking
            if best_mixed > single_util + 0.02:
                gain = best_mixed - single_util
                key = (w1, h1, w2, h2)
                complementary[key] = gain
                pattern_details[key] = best_pattern
                logger.info(
                    "Found complementary pair: (%sx%s, %sx%s) with %.2f%% gain using %s",
                    w1,
                    h1,
                    w2,
                    h2,
                    gain * 100,
                    best_strategy,
                )

    return complementary, pattern_details
