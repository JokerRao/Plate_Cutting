from collections import Counter
import logging
import argparse

# 配置日志系统 (建议在生产环境使用 INFO 级别)
def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger('plate_cutting')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', default='INFO', help='Set logging level: DEBUG / INFO / WARNING / ERROR')
    return parser.parse_args()

# --- 全局设置 ---
args = parse_args()
logger = setup_logger(getattr(logging, args.log_level.upper()))


def should_rotate(x1, x2, length0, width0, ratio_threshold=0.56):
    """
    判断是否应该旋转小板以获得更好的排布效率。(预处理阶段使用)
    返回 True 表示推荐旋转。
    这是一个基于网格布局的简单启发式，可能不适用于所有情况。
    """
    if x1 <= 0 or x2 <= 0:
        return False

    direct_fit = (length0 // x1) * (width0 // x2)
    rotated_fit = (length0 // x2) * (width0 // x1)
    aspect_ratio_close = abs(x1 - x2) / max(x1, x2) < ratio_threshold
    return (x1 < x2 and rotated_fit >= direct_fit) or (aspect_ratio_close and rotated_fit > direct_fit)


# 定义一个板材类来表示大板
class Plate:
    # [REFACTOR] 将硬编码的默认值移出类定义，它们将在 __init__ 中被设定
    # --- 配置参数 ---
    
    def __init__(self, length, width, blade_thick=4, 
                 gap_merge_tolerance=20, 
                 square_like_ratio=0.4, 
                 small_area_threshold=1.2e5,
                 min_width_for_sorting=400):
        self.length = length
        self.width = width
        self.used_area = 0
        self.cuts = []
        self.current_x = 0
        self.current_y = 0
        self.row_width = 0
        self.available_gaps = []
        self.used_perc = 0
        self.blade_thick = blade_thick
        self.last_cut = (0, 0)
        
        # [REFACTOR] 将配置参数作为实例属性，使其可配置
        self.tolerant = gap_merge_tolerance
        self.ratio = square_like_ratio
        self.area_threshold = small_area_threshold
        # 这个参数主要在外部排序时使用，但作为板材配置的一部分是合理的
        self.min_width_for_sorting = min_width_for_sorting

        logger.debug(f"Created new plate: {length}x{width}, blade thickness: {blade_thick}")

    def can_fit(self, small_plate):
        """检查小板是否能放入当前行或开始新行（纯检查，无副作用）"""
        sp_length, sp_width, _, _ = small_plate # 增加了 is_stock 标志位
        if self.current_x + sp_length <= self.length and self.current_y + max(self.row_width, sp_width) <= self.width:
            return True
        elif self.current_y + self.row_width + self.blade_thick + sp_width <= self.width:
            return True
        else:
            return False
    
    def can_fit_in_gap(self, small_plate, gap):
        """检查小板是否能放入指定缝隙"""
        sp_length, sp_width, _, _ = small_plate
        start_x, start_y, end_x, end_y = gap
        fits = start_x + sp_length <= end_x and start_y + sp_width <= end_y
        if fits:
            logger.debug(f"Plate {sp_length}x{sp_width} can fit in gap at ({start_x}, {start_y})")
        return fits
    
    def add_cut_in_gap(self, small_plate):
        """在缝隙中切割小板并添加到切割列表"""
        sp_length, sp_width, _, _ = small_plate
        for i, gap in enumerate(self.available_gaps):
            if self.can_fit_in_gap(small_plate, gap):
                start_x, start_y, end_x, end_y = gap
                self.cuts.append((small_plate, start_x, start_y, start_x + sp_length, start_y + sp_width))
                self.used_area += sp_length * sp_width
                
                # 从列表中移除被占用的缝隙
                del self.available_gaps[i]
                
                logger.debug(f"Cut plate {sp_length}x{sp_width} in gap at ({start_x}, {start_y})")
                
                # 新缝隙1：位于被放置小板的右侧，高度与原缝隙相同
                if start_x + sp_length + self.blade_thick < end_x:
                    right_gap = (start_x + sp_length + self.blade_thick, start_y, end_x, end_y)
                    self._add_gap(right_gap)

                # 新缝隙2：位于被放置小板的下方，宽度仅与被放置小板相同
                # （因为它右边的空间已经被 right_gap 覆盖了）
                if start_y + sp_width + self.blade_thick < end_y:
                    bottom_gap = (start_x, start_y + sp_width + self.blade_thick, start_x + sp_length, end_y)
                    self._add_gap(bottom_gap)
                
                return True
        return False

    def add_cut(self, small_plate):
        """切割小板并添加到切割列表（主区域）"""
        x1, x2 = small_plate[:2]
        should_rotate_here = (
            (x2, x1) == self.last_cut[:2] and 
            x1 and x2 and 
            (x1 * x2 < self.area_threshold or abs(x1 - x2) / max(x1, x2) < self.ratio)
        )
        if should_rotate_here:
            small_plate0 = (x2, x1, small_plate[2], small_plate[3])
        else:
            small_plate0 = small_plate

        if not self.can_fit(small_plate0):
            return False

        sp_length, sp_width, _, _ = small_plate0
        
        if self.current_x + sp_length <= self.length and self.current_y + max(self.row_width, sp_width) <= self.width:
            start_x, start_y = self.current_x, self.current_y
            
            if sp_width < self.row_width:
                if start_y + sp_width + self.blade_thick < start_y + self.row_width:
                    gap_below = (start_x, start_y + sp_width + self.blade_thick, start_x + sp_length, start_y + self.row_width)
                    self._add_gap(gap_below)

            self.current_x += sp_length + self.blade_thick
            self.row_width = max(self.row_width, sp_width)
            
        else:
            if self.row_width > 0:
                gap_at_row_end = (self.current_x, self.current_y, self.length, self.current_y + self.row_width)
                self._add_gap(gap_at_row_end)
            
            self.current_x = 0
            self.current_y += self.row_width + self.blade_thick
            
            start_x, start_y = self.current_x, self.current_y
            self.current_x += sp_length + self.blade_thick
            self.row_width = sp_width
        
        end_x, end_y = start_x + sp_length, start_y + sp_width
        self.cuts.append((small_plate0, start_x, start_y, end_x, end_y))
        self.used_area += sp_length * sp_width
        self.last_cut = (sp_length, sp_width)
        logger.debug(f"Cut plate {sp_length}x{sp_width} at ({start_x}, {start_y})")
        return True

    # [FIX] 重构 _add_gap 方法以避免无限循环
    def _add_gap(self, new_gap):
        """
        添加新的缝隙到可用缝隙列表中，并迭代式地与现有缝隙合并，直到无法再合并为止。
        这种方法比之前的递归添加方式更稳定，可以防止无限循环。
        """
        x1, y1, x2, y2 = new_gap
        # 基础合法性检查
        if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1 or (x2 - x1) < self.blade_thick or (y2 - y1) < self.blade_thick:
            logger.debug(f"Skipped invalid or too small gap: {new_gap}")
            return

        current_gap = new_gap
        
        # 持续尝试合并，直到一轮完整的检查后没有任何合并发生
        while True:
            merged_in_pass = False
            i = 0
            while i < len(self.available_gaps):
                existing_gap = self.available_gaps[i]
                
                # 尝试将 current_gap 和 existing_gap 合并
                # 水平合并检查
                can_merge_horizontally = (
                    abs(current_gap[1] - existing_gap[1]) < self.tolerant and
                    abs(current_gap[3] - existing_gap[3]) < self.tolerant
                )
                if can_merge_horizontally:
                    if abs(current_gap[0] - existing_gap[2]) < self.tolerant: # current 在 existing 右边
                        merged_gap = (existing_gap[0], min(current_gap[1], existing_gap[1]), current_gap[2], max(current_gap[3], existing_gap[3]))
                        current_gap = merged_gap
                        del self.available_gaps[i]
                        merged_in_pass = True
                        break # 合并后，从头开始新一轮检查
                    elif abs(existing_gap[0] - current_gap[2]) < self.tolerant: # current 在 existing 左边
                        merged_gap = (current_gap[0], min(current_gap[1], existing_gap[1]), existing_gap[2], max(current_gap[3], existing_gap[3]))
                        current_gap = merged_gap
                        del self.available_gaps[i]
                        merged_in_pass = True
                        break

                # 垂直合并检查
                can_merge_vertically = (
                    abs(current_gap[0] - existing_gap[0]) < self.tolerant and
                    abs(current_gap[2] - existing_gap[2]) < self.tolerant
                )
                if can_merge_vertically:
                    if abs(current_gap[1] - existing_gap[3]) < self.tolerant: # current 在 existing 下边
                        merged_gap = (min(current_gap[0], existing_gap[0]), existing_gap[1], max(current_gap[2], existing_gap[2]), current_gap[3])
                        current_gap = merged_gap
                        del self.available_gaps[i]
                        merged_in_pass = True
                        break
                    elif abs(existing_gap[1] - current_gap[3]) < self.tolerant: # current 在 existing 上边
                        merged_gap = (min(current_gap[0], existing_gap[0]), current_gap[1], max(current_gap[2], existing_gap[2]), existing_gap[3])
                        current_gap = merged_gap
                        del self.available_gaps[i]
                        merged_in_pass = True
                        break
                
                i += 1 # 如果没有合并，继续检查下一个
            
            # 如果完整遍历了一遍列表都没有发生合并，则结束循环
            if not merged_in_pass:
                break
        
        # 将最终（可能经过多次合并）的缝隙添加到列表中
        self.available_gaps.append(current_gap)


    def utilization(self):
        """计算板材利用率"""
        if self.length * self.width == 0:
            self.used_perc = 0
        else:
            self.used_perc = (self.used_area / (self.length * self.width)) * 100
        logger.debug(f"Plate utilization: {self.used_perc:.2f}%")
        return self.used_perc


# [REFACTOR] 让主函数可以接收配置参数
def optimize_cutting(plates, orders, others, optim=0, n_plate=None, saw_blade=4, plate_config=None):
    """
    输入参数:
    ...
    plate_config: 一个字典，包含传递给Plate类的配置参数
    """
    logger.info("Starting optimization process")
    
    if plate_config is None:
        plate_config = {}

    big_plate_list = []
    for p in plates:
        if isinstance(p, dict) and p.get('quantity', 0) > 0 and p.get('length', 0) > 0 and p.get('width', 0) > 0:
            for _ in range(p['quantity']):
                big_plate_list.append((p['length'], p['width']))
    if not big_plate_list:
        logger.error("No valid plates found. Aborting.")
        return []

    small_plates_list = []
    for p in orders:
        if isinstance(p, dict) and p.get('quantity', 0) > 0 and p.get('length', 0) > 0 and p.get('width', 0) > 0:
            small_plates_list.append((p['length'], p['width'], p.get('id', ''), p['quantity']))

    stock_plates_list = []
    if others:
        for p in others:
            if isinstance(p, dict) and p.get('length', 0) > 0 and p.get('width', 0) > 0:
                stock_id = p.get('id') or f"stock_{p['length']}x{p['width']}"
                stock_plates_list.append((p['length'], p['width'], stock_id))

    # [REFACTOR] 在创建大板对象时，传入配置参数
    big_plate_objects = [Plate(length, width, saw_blade, **plate_config) for length, width in big_plate_list]
    if n_plate is None:
        n_plate = len(big_plate_list)
    
    length0, width0 = big_plate_list[0]

    small_plates2 = []
    for x1, x2, x3, nums in small_plates_list:
        if should_rotate(x1, x2, length0, width0):
            small_plates2.append((x2, x1, x3, nums))
        else:
            small_plates2.append((x1, x2, x3, nums))
    
    # [REFACTOR] 从板材对象实例中获取排序宽度阈值
    min_width_for_sorting = big_plate_objects[0].min_width_for_sorting if big_plate_objects else 400
    width_counts = Counter(plate[1] for plate in small_plates2)
    for i in list(width_counts.keys()):
        if i < min_width_for_sorting:
            width_counts[i] = 0
            
    small_plates2.sort(key=lambda x: (width_counts.get(x[1], 0), x[1], x[0]), reverse=True)
    
    small_plate_objects = [(x1, x2, x3, False) for x1, x2, x3, nums in small_plates2 for _ in range(nums)]
    logger.info(f"Processed {len(small_plate_objects)} total order pieces to cut")

    used_plates = []
    remaining_pieces = small_plate_objects.copy()

    for i, big_plate in enumerate(big_plate_objects):
        if i >= n_plate or not remaining_pieces:
            break
        logger.info(f"Processing plate {i+1}/{len(big_plate_objects)}")
        
        plate_was_used = False
        while True:
            placed_piece_index = -1
            # 优先在缝隙中放置
            if big_plate.available_gaps:
                for j, small_plate in enumerate(remaining_pieces):
                    sp_rotated = (small_plate[1], small_plate[0], small_plate[2], small_plate[3])
                    if big_plate.add_cut_in_gap(small_plate) or \
                       (small_plate[0] != small_plate[1] and big_plate.add_cut_in_gap(sp_rotated)):
                        placed_piece_index = j
                        break
                if placed_piece_index != -1:
                    del remaining_pieces[placed_piece_index]
                    plate_was_used = True
                    continue # 继续尝试在缝隙中放置下一个

            # 在主区域放置
            placed_in_main_area = False
            for j, small_plate in enumerate(remaining_pieces):
                if big_plate.add_cut(small_plate):
                    placed_piece_index = j
                    placed_in_main_area = True
                    break
            
            if placed_in_main_area:
                del remaining_pieces[placed_piece_index]
                plate_was_used = True
            else:
                logger.debug(f"Plate {i+1} is now full or no remaining pieces fit.")
                break
        
        if plate_was_used:
            used_plates.append(big_plate)
            if not remaining_pieces:
                logger.info("All order pieces have been cut.")
                break

    big_plate_objects = used_plates

    if stock_plates_list:
        logger.info("Processing stock plates to fill remaining gaps.")
        stock_plate_objects = [(max(x1, x2), min(x1, x2), x3, True) for x1, x2, x3 in stock_plates_list]
        
        for big_plate in big_plate_objects:
            if not optim:
                _process_stock_plates(big_plate, stock_plate_objects)
            else:
                _optimize_stock_placement(big_plate, stock_plate_objects)

    for big_plate in big_plate_objects:
        big_plate.utilization()

    cutted = []
    for big_plate in big_plate_objects:
        if big_plate.cuts:
            plate_cuts = []
            for cut in big_plate.cuts:
                is_stock_flag = cut[0][3]
                plate_id = cut[0][2]
                
                plate_cuts.append([
                    cut[1],      # start_x
                    cut[2],      # start_y
                    cut[0][0],   # length
                    cut[0][1],   # width
                    1 if is_stock_flag else 0,
                    plate_id
                ])
            cutted.append({
                'rate': big_plate.used_perc / 100,
                'plate': [big_plate.length, big_plate.width],
                'cutted': plate_cuts
            })
            
    logger.info(f"Optimization complete. Generated {len(cutted)} cutting plans.")
    return cutted


def _process_stock_plates(big_plate, stock_plate_objects):
    """
    使用无限供应的库存板，贪婪地填充大板的剩余空间。
    """
    logger.debug("Greedily filling remaining space with stock plates.")
    max_iterations = 100 # 防止意外的无限循环
    for iter_num in range(max_iterations):
        a_piece_was_placed_this_round = False
        
        for stock_plate in stock_plate_objects:
            stock_plate_rotated = (stock_plate[1], stock_plate[0], stock_plate[2], stock_plate[3])
            
            while True:
                if big_plate.add_cut_in_gap(stock_plate) or \
                   (stock_plate[0] != stock_plate[1] and big_plate.add_cut_in_gap(stock_plate_rotated)):
                    a_piece_was_placed_this_round = True
                    continue
                
                if big_plate.add_cut(stock_plate):
                    a_piece_was_placed_this_round = True
                    continue
                
                break
                
        if not a_piece_was_placed_this_round:
            logger.debug(f"No more stock plates can be placed. Finished in {iter_num+1} iterations.")
            break
    else:
        logger.warning("Reached maximum iterations in stock plate processing.")


def _optimize_stock_placement(big_plate, stock_plate_objects):
    """ 
    [MODIFIED] 通过测试不同排序策略（包括优先测试每一种库存板）优化库存板放置。
    """
    if not stock_plate_objects:
        return

    logger.debug("Optimizing stock plate placement with multiple strategies, including prioritizing each stock type.")

    original_state = {
        'cuts': [c for c in big_plate.cuts],
        'available_gaps': [g for g in big_plate.available_gaps],
        'used_area': big_plate.used_area,
        'current_x': big_plate.current_x,
        'current_y': big_plate.current_y,
        'row_width': big_plate.row_width,
        'last_cut': big_plate.last_cut
    }

    candidate_orderings = {
        "largest_area_first": sorted(stock_plate_objects, key=lambda p: p[0] * p[1], reverse=True),
        "smallest_area_first": sorted(stock_plate_objects, key=lambda p: p[0] * p[1]),
        "largest_perimeter_first": sorted(stock_plate_objects, key=lambda p: 2 * (p[0] + p[1]), reverse=True),
        "original": stock_plate_objects,
    }

    for plate_to_prioritize in stock_plate_objects:
        strategy_name = f"prioritize_{plate_to_prioritize[2]}_{plate_to_prioritize[0]}x{plate_to_prioritize[1]}"
        other_plates = [p for p in stock_plate_objects if p != plate_to_prioritize]
        new_ordering = [plate_to_prioritize] + other_plates
        candidate_orderings[strategy_name] = new_ordering

    best_utilization = -1
    best_ordering_name = None
    best_final_state = None

    for name, ordering in candidate_orderings.items():
        # 恢复状态
        big_plate.cuts = [c for c in original_state['cuts']]
        big_plate.available_gaps = [g for g in original_state['available_gaps']]
        big_plate.used_area = original_state['used_area']
        big_plate.current_x = original_state['current_x']
        big_plate.current_y = original_state['current_y']
        big_plate.row_width = original_state['row_width']
        big_plate.last_cut = original_state['last_cut']
        
        _process_stock_plates(big_plate, ordering)
        current_utilization = big_plate.utilization()
        
        logger.debug(f"Testing strategy '{name}': Utilization = {current_utilization:.2f}%")

        if current_utilization > best_utilization + 0.01:
            best_utilization = current_utilization
            best_ordering_name = name
            best_final_state = {
                'cuts': [c for c in big_plate.cuts],
                'available_gaps': [g for g in big_plate.available_gaps],
                'used_area': big_plate.used_area,
                'used_perc': big_plate.used_perc,
                'current_x': big_plate.current_x,
                'current_y': big_plate.current_y,
                'row_width': big_plate.row_width,
                'last_cut': big_plate.last_cut
            }

    if best_final_state:
        logger.info(f"Applying best strategy '{best_ordering_name}' with utilization {best_utilization:.2f}%")
        big_plate.cuts = best_final_state['cuts']
        big_plate.available_gaps = best_final_state['available_gaps']
        big_plate.used_area = best_final_state['used_area']
        big_plate.used_perc = best_final_state['used_perc']
        big_plate.current_x = best_final_state['current_x']
        big_plate.current_y = best_final_state['current_y']
        big_plate.row_width = best_final_state['row_width']
        big_plate.last_cut = best_final_state['last_cut']
    else:
        logger.info("Stock optimization did not yield any valid placement.")