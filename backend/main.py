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
args = parse_args()
logger = setup_logger(getattr(logging, args.log_level.upper()))

def should_rotate(x1, x2, length0, width0, ratio_threshold=0.56):
    """
    判断是否应该旋转小板以获得更好的排布效率。
    返回 True 表示推荐旋转。
    """
    if x1 <= 0 or x2 <= 0:
        return False

    # 不旋转：x1 是长边，x2 是短边
    direct_fit = (length0 // x1) * (width0 // x2)
    # 旋转：x2 是长边，x1 是短边
    rotated_fit = (length0 // x2) * (width0 // x1)

    # 比例判断：近似正方形时也尝试旋转
    aspect_ratio_close = abs(x1 - x2) / max(x1, x2) < ratio_threshold

    # 当原始是长条且旋转后更优，或者接近正方形且旋转更优
    return (x1 < x2 and rotated_fit >= direct_fit) or (aspect_ratio_close and rotated_fit > direct_fit)

# 定义一个板材类来表示大板
class Plate:
    # --- 配置参数 ---
    # 合并缝隙时，坐标差异的容忍度（单位：mm），经验值，小于此值的缝隙可视为连续
    GAP_MERGE_TOLERANCE = 30 
    # 当一行宽度接近总宽度时，自动延伸以简化余料的阈值（单位：mm）
    ROW_ATTACH_THRESHOLD = 100 
    # 判断小板是否接近正方形的比例阈值，用于旋转决策
    SQUARE_LIKE_RATIO = 0.4
    # 在特定启发式旋转中，用于判断是否为“小”板的面积阈值
    SMALL_AREA_THRESHOLD = 1.2e5 
    # 过滤掉宽度小于此值的小板，不参与宽度频率排序，以优先处理大尺寸板
    MIN_WIDTH_FOR_SORTING = 400
    
    def __init__(self, length, width, blade_thick=4):
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
        self.tolerant = self.GAP_MERGE_TOLERANCE
        self.attach = self.ROW_ATTACH_THRESHOLD
        self.last_cut = (0, 0)
        self.ratio = self.SQUARE_LIKE_RATIO
        self.area_threshold = self.SMALL_AREA_THRESHOLD
        logger.debug(f"Created new plate: {length}x{width}, blade thickness: {blade_thick}")

    def can_fit(self, small_plate):
        """检查小板是否能放入当前行或开始新行（纯检查，无副作用）"""
        sp_length, sp_width, _ = small_plate
        if self.current_x + sp_length <= self.length and self.current_y + max(self.row_width, sp_width) <= self.width:
            return True
        elif self.current_y + self.row_width + self.blade_thick + sp_width <= self.width:
            return True
        else:
            return False
    
    def can_fit_in_gap(self, small_plate, gap):
        """检查小板是否能放入指定缝隙"""
        sp_length, sp_width, _ = small_plate
        start_x, start_y, end_x, end_y = gap
        fits = start_x + sp_length <= end_x and start_y + sp_width <= end_y
        if fits:
            logger.debug(f"Plate {sp_length}x{sp_width} can fit in gap at ({start_x}, {start_y})")
        return fits
    
    def add_cut_in_gap(self, small_plate):
        """在缝隙中切割小板并添加到切割列表"""
        sp_length, sp_width, _ = small_plate
        for gap in self.available_gaps[:]:
            if self.can_fit_in_gap(small_plate, gap):
                start_x, start_y, end_x, end_y = gap
                self.cuts.append((small_plate, start_x, start_y, start_x + sp_length, start_y + sp_width))
                self.used_area += sp_length * sp_width
                self.available_gaps.remove(gap)
                logger.debug(f"Cut plate {sp_length}x{sp_width} in gap at ({start_x}, {start_y})")
                
                right_gap = (start_x + sp_length + self.blade_thick, start_y, end_x, end_y)
                self._add_gap(right_gap)

                bottom_gap = (start_x, start_y + sp_width + self.blade_thick, start_x + sp_length, end_y)
                self._add_gap(bottom_gap)
                
                return True
        return False

    def add_cut(self, small_plate):
        """切割小板并添加到切割列表（主区域）"""
        x1, x2 = small_plate[:2]
        # 简化旋转逻辑，只在特定条件下旋转
        should_rotate_here = (
            (x2, x1) == self.last_cut[:2] and 
            x1 and x2 and 
            (x1 * x2 < self.area_threshold or abs(x1 - x2) / max(x1, x2) < self.ratio)
        )
        if should_rotate_here:
            small_plate0 = (x2, x1, small_plate[2])
        else:
            small_plate0 = small_plate

        if not self.can_fit(small_plate0):
            return False

        sp_length, sp_width, _ = small_plate0
        
        if self.current_x + sp_length <= self.length and self.current_y + max(self.row_width, sp_width) <= self.width:
            start_x, start_y = self.current_x, self.current_y
            
            if sp_width < self.row_width:
                 gap_below = (start_x, start_y + sp_width + self.blade_thick, start_x + sp_length, start_y + self.row_width)
                 self._add_gap(gap_below)

            self.current_x += sp_length + self.blade_thick
            self.row_width = max(self.row_width, sp_width)
            
            if start_x == 0 and start_y == 0 and self.row_width <= self.width and self.row_width > self.width - self.attach:
                 self.row_width = self.width - start_y
                 new_gap = (0, start_y + sp_width + self.blade_thick, self.current_x, self.width)
                 self._add_gap(new_gap)
        else:
            if self.row_width > 0:
                gap_at_row_end = (self.current_x, self.current_y, self.length, self.current_y + self.row_width)
                self._add_gap(gap_at_row_end)
            
            self.current_x = 0
            self.current_y += self.row_width
            
            start_x, start_y = self.current_x, self.current_y
            self.current_x += sp_length + self.blade_thick
            self.row_width = sp_width
            
            if start_y + self.row_width <= self.width and start_y + self.row_width > self.width - self.attach:
                new_gap = (self.current_x, start_y + self.row_width, self.length, self.width)
                self._add_gap(new_gap)
                self.row_width = self.width - start_y
        
        end_x, end_y = start_x + sp_length, start_y + sp_width
        self.cuts.append((small_plate, start_x, start_y, end_x, end_y))
        self.used_area += sp_length * sp_width
        self.last_cut = (sp_length, sp_width)
        logger.debug(f"Cut plate {sp_length}x{sp_width} at ({start_x}, {start_y})")
        return True

    def _add_gap(self, new_gap):
        """
        添加新的缝隙到可用缝隙列表中，并尝试与现有缝隙合并
        
        该方法实现了智能的缝隙管理算法：
        1. 验证新缝隙的合法性
        2. 尝试与现有缝隙进行水平或垂直合并
        3. 使用容忍度参数来处理浮点精度问题
        4. 递归处理合并后的缝隙，确保完全合并
        
        Args:
            new_gap: 新缝隙的坐标元组 (x1, y1, x2, y2)
                    x1,y1: 左上角坐标
                    x2,y2: 右下角坐标
        
        Returns:
            None (直接修改 self.available_gaps)
        """
        x1, y1, x2, y2 = new_gap

        # 坐标合法性检查 - 确保所有坐标都是非负数
        if min(x1, y1, x2, y2) < 0:
            logger.warning(f"Skipped invalid gap with negative coordinates: {new_gap}")
            return
        
        # 尺寸合法性检查 - 确保缝隙有正的长度和宽度
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Skipped invalid gap with non-positive dimensions: {new_gap}")
            return

        # 最小尺寸检查 - 过滤掉小于锯片厚度的缝隙，避免无效切割
        if (x2 - x1) < self.blade_thick or (y2 - y1) < self.blade_thick:
            logger.debug(f"Skipped too small gap: {new_gap}")
            return

        # 待处理的缝隙队列，用于递归合并
        gaps_to_process = [new_gap]
        
        # 处理所有待合并的缝隙
        while gaps_to_process:
            current_gap = gaps_to_process.pop(0)
            merged_with_existing = False
            i = 0
            
            # 遍历现有缝隙，寻找可合并的缝隙
            while i < len(self.available_gaps):
                existing_gap = self.available_gaps[i]
                
                # 水平合并检查：两个缝隙在Y轴方向对齐（考虑容忍度）
                # 条件：Y坐标差异小于容忍度，且X轴方向相邻
                if (abs(current_gap[1] - existing_gap[1]) < self.tolerant and 
                    abs(current_gap[3] - existing_gap[3]) < self.tolerant):
                    
                    # 情况1：当前缝隙在现有缝隙右侧
                    if abs(current_gap[2] - existing_gap[0]) < self.tolerant:
                        merged_gap = (current_gap[0], min(current_gap[1], existing_gap[1]), 
                                    existing_gap[2], max(current_gap[3], existing_gap[3]))
                        del self.available_gaps[i]
                        gaps_to_process.append(merged_gap)
                        merged_with_existing = True
                        break
                    # 情况2：当前缝隙在现有缝隙左侧
                    elif abs(existing_gap[2] - current_gap[0]) < self.tolerant:
                        merged_gap = (existing_gap[0], min(current_gap[1], existing_gap[1]), 
                                    current_gap[2], max(current_gap[3], existing_gap[3]))
                        del self.available_gaps[i]
                        gaps_to_process.append(merged_gap)
                        merged_with_existing = True
                        break

                # 垂直合并检查：两个缝隙在X轴方向对齐（考虑容忍度）
                # 条件：X坐标差异小于容忍度，且Y轴方向相邻
                if (abs(current_gap[0] - existing_gap[0]) < self.tolerant and 
                    abs(current_gap[2] - existing_gap[2]) < self.tolerant):
                    
                    # 情况1：当前缝隙在现有缝隙下方
                    if abs(current_gap[3] - existing_gap[1]) < self.tolerant:
                        merged_gap = (min(current_gap[0], existing_gap[0]), current_gap[1], 
                                    max(current_gap[2], existing_gap[2]), existing_gap[3])
                        del self.available_gaps[i]
                        gaps_to_process.append(merged_gap)
                        merged_with_existing = True
                        break
                    # 情况2：当前缝隙在现有缝隙上方
                    elif abs(existing_gap[3] - current_gap[1]) < self.tolerant:
                        merged_gap = (min(current_gap[0], existing_gap[0]), existing_gap[1], 
                                    max(current_gap[2], existing_gap[2]), current_gap[3])
                        del self.available_gaps[i]
                        gaps_to_process.append(merged_gap)
                        merged_with_existing = True
                        break
                i += 1
            
            # 如果没有找到可合并的缝隙，将当前缝隙添加到可用缝隙列表
            if not merged_with_existing:
                self.available_gaps.append(current_gap)

    def utilization(self):
        """计算板材利用率"""
        if self.length * self.width == 0:
            self.used_perc = 0
        else:
            self.used_perc = (self.used_area / (self.length * self.width)) * 100
        logger.debug(f"Plate utilization: {self.used_perc:.2f}%")
        return self.used_perc

# 主函数:优化大板切割方案
def optimize_cutting(plates, orders, others, optim = 0, n_plate = None, saw_blade=4):
    """
    输入参数:
    plates: 大板列表，格式为[{"id":1, "length":2440, "width":1220, "quantity":100}, ...]
    orders: 订单列表，格式为[{"id":1, "length":1217, "width":607, "quantity":10}, ...]
    others: 库存板列表，格式为[{"id":1, "length":1217, "width":607, "client":"bbb"}, ...]
    optim: 是否优化库存板排列，默认为0
    n_plate: 使用大板数量，默认为None（使用所有大板）
    saw_blade: 锯片厚度，默认为4
    """
    logger.info("Starting optimization process")
    logger.info(f"Input parameters: {len(plates)} plates, {len(orders)} orders, {len(others) if others else 0} stock items")
    logger.debug(f"Optimization mode: {optim}, n_plate: {n_plate}, saw_blade: {saw_blade}")

    # 转换大板格式
    big_plate_list = []
    for p in plates:
        if not isinstance(p, dict):
            logger.warning(f"Invalid plate format: {p}, skipping")
            continue
        if isinstance(p.get('quantity'), int) and p['quantity'] > 0:
            if p.get('length', 0) <= 0 or p.get('width', 0) <= 0:
                logger.warning(f"Invalid plate dimensions: {p}, skipping")
                continue
            for _ in range(p['quantity']):
                big_plate_list.append((p['length'], p['width']))
    if not big_plate_list:
        logger.warning("No valid plates found in input")
        return []
    
    # 转换小板格式 (orders)
    small_plates_list = []
    for p in orders:
        if not isinstance(p, dict):
            logger.warning(f"Invalid order format: {p}, skipping")
            continue
        if isinstance(p.get('quantity'), int) and p['quantity'] > 0:
            if p.get('length', 0) <= 0 or p.get('width', 0) <= 0:
                logger.warning(f"Invalid order dimensions: {p}, skipping")
                continue
            small_plates_list.append((p['length'], p['width'], p.get('id', ''), p['quantity']))
    
    # 转换库存板格式 (others)
    stock_plates_list = []
    if others:
        for p in others:
            if not isinstance(p, dict):
                logger.warning(f"Invalid stock format: {p}, skipping")
                continue
            if p.get('length', 0) > 0 and p.get('width', 0) > 0:
                stock_plates_list.append((p['length'], p['width'], f"R{p.get('id', '')}"))
            else:
                logger.warning(f"Invalid stock dimensions: {p}, skipping")

    logger.info(f"Processed input: {len(big_plate_list)} plates, {len(small_plates_list)} orders, {len(stock_plates_list)} stock items")

    # 创建大板对象列表
    big_plate_objects = [Plate(length, width, saw_blade) for length, width in big_plate_list]
    small_plates1 = small_plates_list.copy()
    
    if n_plate is None:
        n_plate = len(big_plate_list)
    
    # 处理库存板材,确保长边在前
    stock_plate_objects = [(max(x1, x2), min(x1, x2), x3) for x1, x2, x3 in stock_plates_list] if stock_plates_list else []
    
    if not big_plate_list:
        logger.error("Big plate list is empty, cannot proceed with optimization.")
        return []
    length0 = big_plate_list[0][0]
    width0 = big_plate_list[0][1]
    
    # 根据切割效率对小板进行排序和旋转
    small_plates2 = []
    for plate_data in small_plates1:
        # 安全地解包数据，防止索引错误
        if len(plate_data) != 4:
            logger.warning(f"Invalid plate data format: {plate_data}, skipping")
            continue
        x1, x2, x3, nums = plate_data
        # 判断是否需要旋转小板材以获得更好的切割效率
        if should_rotate(x1, x2, length0, width0):
            small_plates2.append((x2, x1, x3, nums))
            logger.debug(f"Rotated order plate: {x1}x{x2} -> {x2}x{x1}")
        else:
            small_plates2.append((x1, x2, x3, nums))
    # 统计宽度出现次数并排序
    width_counts = Counter(plate[1] for plate in small_plates2)
    for i in list(width_counts.keys()): # 使用 list 包装以允许在迭代中修改
        if i < Plate.MIN_WIDTH_FOR_SORTING:
            width_counts[i] = 0
    small_plates2.sort(key=lambda x: (width_counts.get(x[1], 0), x[1], x[0]), reverse=True)
    small_plate_objects = [(x1, x2, x3) for x1, x2, x3, nums in small_plates2 for _ in range(nums)]
    logger.info(f"Processed {len(small_plate_objects)} total pieces to cut")

    # 主切割循环
    used_plates = []
    remaining_pieces_count = len(small_plate_objects)

    for i, big_plate in enumerate(big_plate_objects):
        if i >= n_plate:
            break
        logger.info(f"Processing plate {i+1}/{len(big_plate_objects)}")
        if remaining_pieces_count == 0:
            logger.info("All pieces have been cut.")
            break

        plate_was_used_in_this_round = False
        while True:
            placed_a_piece_this_pass = False
            placed_piece_index = -1

            for j, small_plate in enumerate(small_plate_objects):
                if small_plate is None:
                    continue

                placed_successfully = False
                
                if big_plate.add_cut_in_gap(small_plate):
                    placed_successfully = True
                elif small_plate[0] != small_plate[1]:
                    small_plate_rotated = (small_plate[1], small_plate[0], small_plate[2])
                    if big_plate.add_cut_in_gap(small_plate_rotated):
                        placed_successfully = True

                if not placed_successfully and big_plate.add_cut(small_plate):
                    placed_successfully = True

                if placed_successfully:
                    placed_piece_index = j
                    placed_a_piece_this_pass = True
                    plate_was_used_in_this_round = True
                    break

            if placed_a_piece_this_pass:
                small_plate_objects[placed_piece_index] = None
                remaining_pieces_count -= 1
            else:
                logger.debug(f"Plate {i+1} is now full or no remaining pieces fit.")
                break

        if plate_was_used_in_this_round:
            used_plates.append(big_plate)

    big_plate_objects = used_plates

    if stock_plates_list:
        logger.info("Processing stock plates")
        for i, big_plate in enumerate(big_plate_objects):
            logger.debug(f"Processing stock plates for plate {i+1}/{len(big_plate_objects)}")
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
                is_stock = str(cut[0][2]).startswith('R')
                plate_cuts.append([
                    cut[1],
                    cut[2],  
                    cut[0][0],
                    cut[0][1],
                    1 if is_stock else 0,
                    cut[0][2][1:] if is_stock else cut[0][2]
                ])
            cutted.append({
                'rate': big_plate.used_perc/100,
                'plate': [big_plate.length, big_plate.width],
                'cutted': plate_cuts
            })
            
    logger.info(f"Optimization complete. Generated {len(cutted)} cutting plans")
    return cutted

def _process_stock_plates(big_plate, stock_plate_objects):
    """使用更智能的贪婪算法，反复尝试将库存板填充到大板的剩余空间中。"""
    logger.debug("Processing stock plates by greedily filling remaining space.")
    max_iterations = 1000  # 防止无限循环
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        a_piece_was_placed_in_this_pass = False
        
        for stock_plate in stock_plate_objects:
            stock_plate_rotated = (stock_plate[1], stock_plate[0], stock_plate[2])
            while True:
                placed_this_iteration = False
                if big_plate.add_cut_in_gap(stock_plate):
                    placed_this_iteration = True
                elif stock_plate[0] != stock_plate[1] and big_plate.add_cut_in_gap(stock_plate_rotated):
                    placed_this_iteration = True
                elif big_plate.add_cut(stock_plate):
                    placed_this_iteration = True

                if placed_this_iteration:
                    a_piece_was_placed_in_this_pass = True
                    continue
                else:
                    break
                    
        if not a_piece_was_placed_in_this_pass:
            break
    
    if iteration_count >= max_iterations:
        logger.warning("Reached maximum iterations in stock plate processing")

def _optimize_stock_placement(big_plate, stock_plate_objects):
    """通过测试几种不同的智能排序策略来优化库存板的放置，并使用“保存/恢复状态”代替 deepcopy 以提高性能。"""
    if not stock_plate_objects:
        return

    logger.debug("Optimizing stock plate placement with faster, smarter strategy.")

    original_state = {
        'cuts': big_plate.cuts.copy(),
        'available_gaps': big_plate.available_gaps.copy(),
        'used_area': big_plate.used_area,
        'current_x': big_plate.current_x,
        'current_y': big_plate.current_y,
        'row_width': big_plate.row_width
    }

    candidate_orderings = {
        "original": stock_plate_objects,
        "largest_area_first": sorted(stock_plate_objects, key=lambda p: p[0] * p[1], reverse=True),
        "smallest_area_first": sorted(stock_plate_objects, key=lambda p: p[0] * p[1]),
        "largest_perimeter_first": sorted(stock_plate_objects, key=lambda p: 2 * (p[0] + p[1]), reverse=True)
    }

    best_utilization = -1
    best_ordering_name = None
    best_final_state = None

    for name, ordering in candidate_orderings.items():
        big_plate.cuts = original_state['cuts'].copy()
        big_plate.available_gaps = original_state['available_gaps'].copy()
        big_plate.used_area = original_state['used_area']
        big_plate.current_x = original_state['current_x']
        big_plate.current_y = original_state['current_y']
        big_plate.row_width = original_state['row_width']

        _process_stock_plates(big_plate, ordering)
        current_utilization = big_plate.utilization()
        
        logger.debug(f"Testing strategy '{name}': Utilization = {current_utilization:.2f}%")

        if current_utilization > best_utilization + 0.01:
            best_utilization = current_utilization
            best_ordering_name = name
            best_final_state = {
                'cuts': big_plate.cuts.copy(),
                'available_gaps': big_plate.available_gaps.copy(),
                'used_area': big_plate.used_area,
                'used_perc': big_plate.used_perc
            }

    if best_final_state:
        logger.info(f"Applying best strategy '{best_ordering_name}' with utilization {best_utilization:.2f}%")
        big_plate.cuts = best_final_state['cuts']
        big_plate.available_gaps = best_final_state['available_gaps']
        big_plate.used_area = best_final_state['used_area']
        big_plate.used_perc = best_final_state['used_perc']
    else:
        logger.info("Stock optimization did not yield any valid placement.")