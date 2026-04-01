"""行式互补排布（从 PlateOptimizer 抽出）。"""
import logging
from typing import Any, Dict, List, Tuple

from core.models import Cut, CuttingConfig, SmallPlate

logger = logging.getLogger("plate_cutting")


def pack_orders_row_based(
    config: CuttingConfig,
    big_plate: SmallPlate,
    orders: List[SmallPlate],
    size1_key: Tuple,
    size2_key: Tuple,
    count1_per_row: int,
    count2_per_row: int,
) -> Tuple[List[Cut], List[SmallPlate]]:
    bt = config.blade_thickness
    w1, h1 = size1_key
    w2, h2 = size2_key
    num_rows = int(big_plate.width // h1)

    size1_orders = []
    size2_orders = []
    for i, order in enumerate(orders):
        w_with_blade = order.length + bt
        h_with_blade = order.width + bt
        if abs(w_with_blade - w1) < 1 and abs(h_with_blade - h1) < 1:
            size1_orders.append((i, order))
        elif abs(w_with_blade - w2) < 1 and abs(h_with_blade - h2) < 1:
            size2_orders.append((i, order))

    cuts: List[Cut] = []
    packed_indices = set()

    current_y = 0.0
    for row_idx in range(num_rows):
        current_x = 0.0
        for _ in range(count1_per_row):
            if size1_orders:
                idx, order = size1_orders.pop(0)
                cuts.append(
                    Cut(
                        plate=order,
                        x1=current_x,
                        y1=current_y,
                        x2=current_x + order.length,
                        y2=current_y + order.width,
                        is_stock=False,
                    )
                )
                packed_indices.add(idx)
                current_x += order.length + bt
        for _ in range(count2_per_row):
            if size2_orders:
                idx, order = size2_orders.pop(0)
                cuts.append(
                    Cut(
                        plate=order,
                        x1=current_x,
                        y1=current_y,
                        x2=current_x + order.length,
                        y2=current_y + order.width,
                        is_stock=False,
                    )
                )
                packed_indices.add(idx)
                current_x += order.length + bt
                
        # 兜底贪心填充：如果标准 pattern 的数量耗尽，但行内还有物理空间，直接用剩余零件塞满此行
        while True:
            added = False
            if size1_orders and current_x + w1 <= big_plate.length + 0.001:
                idx, order = size1_orders.pop(0)
                cuts.append(Cut(plate=order, x1=current_x, y1=current_y, x2=current_x + order.length, y2=current_y + order.width, is_stock=False))
                packed_indices.add(idx)
                current_x += w1
                added = True
            elif size2_orders and current_x + w2 <= big_plate.length + 0.001:
                idx, order = size2_orders.pop(0)
                cuts.append(Cut(plate=order, x1=current_x, y1=current_y, x2=current_x + order.length, y2=current_y + order.width, is_stock=False))
                packed_indices.add(idx)
                current_x += w2
                added = True
            if not added:
                break
                
        current_y += float(h1)
        if not size1_orders and not size2_orders:
            break

    remaining = [order for i, order in enumerate(orders) if i not in packed_indices]
    logger.info(
        "Row-based packing: placed %d pieces in %d rows",
        len(cuts),
        row_idx + 1,
    )
    return cuts, remaining


def pack_orders_layer_based(
    config: CuttingConfig,
    big_plate: SmallPlate,
    orders: List[SmallPlate],
    size1_key: Tuple,
    size2_key: Tuple,
    r1: int,
    r2: int,
    c1: int,
    c2: int,
) -> Tuple[List[Cut], List[SmallPlate]]:
    """以层叠的方式进行混合排配（例如：底部 r1 行每行高度 h1 放 c1 个，顶部 r2 行每行高度 h2 放 c2 个）。
    支持同一个零件自身的任意旋转形态装箱。"""
    bt = config.blade_thickness
    w1, h1 = size1_key
    w2, h2 = size2_key
    
    # 统一提取符合尺寸及旋转要求的零件池
    pool = []
    for i, order in enumerate(orders):
        wb = order.length + bt
        hb = order.width + bt
        # 是否满足 size1（原向或翻转）
        match1 = (abs(wb - w1) < 1 and abs(hb - h1) < 1) or (abs(wb - h1) < 1 and abs(hb - w1) < 1)
        # 是否满足 size2（原向或翻转）
        match2 = (abs(wb - w2) < 1 and abs(hb - h2) < 1) or (abs(wb - h2) < 1 and abs(hb - w2) < 1)
        
        if match1 or match2:
            pool.append((i, order))
            
    cuts: List[Cut] = []
    packed_indices = set()
    
    current_y = 0.0
    
    # 浇筑第一群层 (r1 行)
    for _ in range(r1):
        current_x = 0.0
        row_h = float(h1)
        for _ in range(c1):
            if pool:
                idx, order = pool.pop(0)
                # 检查此零件当前姿态是否天然匹配 w1，否则必须旋转
                actual_w = float(order.length)
                actual_h = float(order.width)
                if abs((order.length + bt) - w1) >= 1: # 宽度不符，说明必须旋转才适配
                    actual_w = float(order.width)
                    actual_h = float(order.length)
                    
                cuts.append(Cut(plate=order, x1=current_x, y1=current_y, x2=current_x+actual_w, y2=current_y+actual_h, is_stock=False))
                packed_indices.add(idx)
                current_x += w1
        current_y += row_h
        
    # 浇筑第二群层 (r2 行)
    for _ in range(r2):
        current_x = 0.0
        row_h = float(h2)
        for _ in range(c2):
            if pool:
                idx, order = pool.pop(0)
                # 同理：检查此零件对于第二种楼层需求是否匹配 w2，否则必须旋转
                actual_w = float(order.length)
                actual_h = float(order.width)
                if abs((order.length + bt) - w2) >= 1:
                    actual_w = float(order.width)
                    actual_h = float(order.length)
                    
                cuts.append(Cut(plate=order, x1=current_x, y1=current_y, x2=current_x+actual_w, y2=current_y+actual_h, is_stock=False))
                packed_indices.add(idx)
                current_x += w2
        current_y += row_h
        
    remaining = [order for i, order in enumerate(orders) if i not in packed_indices]
    logger.info(
        "Layer-based packing: placed %d pieces in %d + %d layers",
        len(cuts),
        r1,
        r2,
    )
    return cuts, remaining

def pack_orders_composite_stack(
    big_plate: Any,
    orders: List[Any],
    pattern_details: Dict[str, Any],
    blade_thickness: float,
) -> Tuple[List[Cut], List[Any]]:
    """
    Physically builds the 'composite-stack' macro pattern.
    Places interlocking columns of size1 and size2, and greedily fills the tail gap.
    """
    cuts: List[Cut] = []
    bt = float(blade_thickness)
    
    col_w = float(pattern_details['col_w'])
    c_w1, c_h1 = float(pattern_details['c_w1']), float(pattern_details['c_h1'])
    c_w2, c_h2 = float(pattern_details['c_w2']), float(pattern_details['c_h2'])
    num_cols = int(pattern_details['num_cols'])
    
    # Pre-filter orders into groups
    # We must match the raw dimension (without blade) by subtracting bt first
    w1, h1 = c_w1, c_h1
    w2, h2 = c_w2, c_h2
    
    size1_orders, size2_orders = [], []
    packed_indices = set()
    
    for i, o in enumerate(orders):
        ow, oh = float(o.length) + bt, float(o.width) + bt
        # Because complementary_pairs checks matching precisely:
        if (abs(ow - w1) < 1 and abs(oh - h1) < 1) or (abs(ow - h1) < 1 and abs(oh - w1) < 1):
            size1_orders.append((i, o))
        elif (abs(ow - w2) < 1 and abs(oh - h2) < 1) or (abs(ow - h2) < 1 and abs(oh - w2) < 1):
            size2_orders.append((i, o))
            
    current_x = 0.0
    for _ in range(num_cols):
        # We need 1 size1 and 1 size2
        if size1_orders and size2_orders:
            idx1, order1 = size1_orders.pop(0)
            idx2, order2 = size2_orders.pop(0)
            
            # Place size1 at the top or bottom? The complementary_pairs assumes they stack.
            # Let's cleanly stack them. size1 at Y=0, size2 at Y=h1.
            act_w1, act_h1 = float(order1.length), float(order1.width)
            if abs((act_w1 + bt) - c_w1) >= 1:
                act_w1, act_h1 = act_h1, act_w1
                
            cuts.append(Cut(plate=order1, x1=current_x, y1=0, x2=current_x+act_w1, y2=act_h1, is_stock=False))
            packed_indices.add(idx1)
            
            act_w2, act_h2 = float(order2.length), float(order2.width)
            if abs((act_w2 + bt) - c_w2) >= 1:
                act_w2, act_h2 = act_h2, act_w2
                
            cuts.append(Cut(plate=order2, x1=current_x, y1=c_h1, x2=current_x+act_w2, y2=c_h1+act_h2, is_stock=False))
            packed_indices.add(idx2)
            
            current_x += col_w
        else:
            break
            
    # Try fitting anything into the tail gap
    # Start greedily fitting into the remaining full-height columns
    # Right gap starts at current_x
    while size1_orders or size2_orders:
        added = False
        remaining_x = big_plate.length + 0.001 - current_x
        if remaining_x <= 0:
            break
            
        # Try fitting size1 column
        if size1_orders and h1 <= big_plate.width + 0.001 and w1 <= remaining_x:
            idx, order = size1_orders.pop(0)
            act_w, act_h = float(order.length), float(order.width)
            if abs((act_w + bt) - w1) >= 1:
                act_w, act_h = act_h, act_w
            cuts.append(Cut(plate=order, x1=current_x, y1=0, x2=current_x+act_w, y2=act_h, is_stock=False))
            packed_indices.add(idx)
            current_x += w1
            added = True
        elif size1_orders and w1 <= big_plate.width + 0.001 and h1 <= remaining_x: # rotated size1
            idx, order = size1_orders.pop(0)
            act_w, act_h = float(order.length), float(order.width)
            if abs((act_w + bt) - h1) >= 1:
                act_w, act_h = act_h, act_w
            cuts.append(Cut(plate=order, x1=current_x, y1=0, x2=current_x+act_w, y2=act_h, is_stock=False))
            packed_indices.add(idx)
            current_x += h1
            added = True
        # Try fitting size2 column
        elif size2_orders and h2 <= big_plate.width + 0.001 and w2 <= remaining_x:
            idx, order = size2_orders.pop(0)
            act_w, act_h = float(order.length), float(order.width)
            if abs((act_w + bt) - w2) >= 1:
                act_w, act_h = act_h, act_w
            cuts.append(Cut(plate=order, x1=current_x, y1=0, x2=current_x+act_w, y2=act_h, is_stock=False))
            packed_indices.add(idx)
            current_x += w2
            added = True
        elif size2_orders and w2 <= big_plate.width + 0.001 and h2 <= remaining_x: # rotated size2
            idx, order = size2_orders.pop(0)
            act_w, act_h = float(order.length), float(order.width)
            if abs((act_w + bt) - h2) >= 1:
                act_w, act_h = act_h, act_w
            cuts.append(Cut(plate=order, x1=current_x, y1=0, x2=current_x+act_w, y2=act_h, is_stock=False))
            packed_indices.add(idx)
            current_x += h2
            added = True
            
        if not added:
            break
            
    remaining = [order for i, order in enumerate(orders) if i not in packed_indices]
    return cuts, remaining


def _match_eff_key(
    order: SmallPlate, bt: float, kw: float, kh: float
) -> bool:
    ow = float(order.length) + bt
    oh = float(order.width) + bt
    return (abs(ow - kw) < 1 and abs(oh - kh) < 1) or (
        abs(ow - kh) < 1 and abs(oh - kw) < 1
    )


def pack_orders_band_fill(
    big_plate: SmallPlate,
    orders: List[SmallPlate],
    pattern_details: Dict[str, Any],
    blade_thickness: float,
) -> Tuple[List[Cut], List[SmallPlate]]:
    """
    双大件横排 + 底部三条同宽条料 + 中间竖向条料（中缝）。
    适用于专家案：2×大件 + 4×长条（3 底 + 1 中缝）。
    """
    bt = float(blade_thickness)
    big_k = pattern_details["big_key"]
    strip_k = pattern_details["strip_key"]
    bkw, bkh = float(big_k[0]), float(big_k[1])
    skw, skh = float(strip_k[0]), float(strip_k[1])
    strip_long = max(skw, skh)
    strip_short = min(skw, skh)

    # Dynamic counts from pattern detection; fall back to original defaults
    n_big = int(pattern_details.get("n_big", 2))
    n_bottom = int(pattern_details.get("n_bottom", 3))
    n_mid = int(pattern_details.get("n_mid", 1))  # strips between big pieces
    n_strips_needed = n_bottom + n_mid

    big_pool: List[Tuple[int, SmallPlate]] = []
    strip_pool: List[Tuple[int, SmallPlate]] = []
    packed_indices: set = set()

    for i, o in enumerate(orders):
        if _match_eff_key(o, bt, bkw, bkh):
            big_pool.append((i, o))
        elif _match_eff_key(o, bt, skw, skh):
            strip_pool.append((i, o))

    cuts: List[Cut] = []

    def orient_for_eff(
        order: SmallPlate, want_w: float, want_h: float
    ) -> Tuple[float, float]:
        aw = float(order.length) + bt
        ah = float(order.width) + bt
        if abs(aw - want_w) < 1 and abs(ah - want_h) < 1:
            return float(order.length), float(order.width)
        return float(order.width), float(order.length)

    # Need exactly n_big bigs + n_strips_needed strips; skip if insufficient
    if len(big_pool) < n_big or len(strip_pool) < n_strips_needed:
        return cuts, list(orders)

    # Bottom band: n_bottom strips placed with long edge horizontal
    x = 0.0
    y_bot = 0.0
    bot_h = 0.0
    for _ in range(n_bottom):
        idx, order = strip_pool.pop(0)
        act_w, act_h = orient_for_eff(order, strip_long, strip_short)
        cuts.append(
            Cut(
                plate=order,
                x1=x,
                y1=y_bot,
                x2=x + act_w,
                y2=y_bot + act_h,
                is_stock=False,
            )
        )
        packed_indices.add(idx)
        x += act_w + bt
        bot_h = max(bot_h, act_h)

    y_top = bot_h + bt

    # Upper band: [big][mid_strip][big][mid_strip]...[big]
    # n_big big pieces interleaved with n_mid = n_big-1 middle strips
    x = 0.0
    for bi in range(n_big):
        idx, order = big_pool.pop(0)
        act_w, act_h = orient_for_eff(order, bkw, bkh)
        cuts.append(
            Cut(
                plate=order,
                x1=x,
                y1=y_top,
                x2=x + act_w,
                y2=y_top + act_h,
                is_stock=False,
            )
        )
        packed_indices.add(idx)
        x += act_w + bt

        # Place middle strip between big pieces (not after the last one)
        if bi < n_big - 1 and strip_pool:
            idx, order = strip_pool.pop(0)
            act_w, act_h = orient_for_eff(order, strip_short, strip_long)
            cuts.append(
                Cut(
                    plate=order,
                    x1=x,
                    y1=y_top,
                    x2=x + act_w,
                    y2=y_top + act_h,
                    is_stock=False,
                )
            )
            packed_indices.add(idx)
            x += act_w + bt

    remaining = [o for i, o in enumerate(orders) if i not in packed_indices]
    logger.info(
        "Band-fill packing: placed %d pieces (%d big + %d bottom + %d mid strips)",
        len(cuts),
        n_big,
        n_bottom,
        n_mid,
    )
    return cuts, remaining
