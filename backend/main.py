import copy
from collections import Counter
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,  # 设置为 INFO 以减少不必要的输出，调试时可改为 DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('plate_cutting')

# 定义一个板材类来表示大板
class Plate:
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
        self.tolerant = 30
        self.attach = 100
        self.stuck = 0
        self.last_cut = (0, 0)
        self.ratio = 0.4
        self.show = 1.2 * 10**5
        logger.debug(f"Created new plate: {length}x{width}, blade thickness: {blade_thick}")

    def can_fit(self, small_plate):
        sp_length, sp_width, _ = small_plate
        if self.current_x + sp_length <= self.length and self.current_y + sp_width <= self.width:
            logger.debug(f"Plate {sp_length}x{sp_width} can fit at current position ({self.current_x}, {self.current_y})")
            return True
        elif self.current_y + self.row_width + sp_width <= self.width:
            if not self.stuck:
                self.update_gap()
            self.stuck = 0
            logger.debug(f"Plate {sp_length}x{sp_width} can fit in new row")
            return True
        else:
            if not self.stuck:
                self.update_gap()
                self.current_x = 0
                self.current_y += self.row_width
                self.row_width = 0
                self.stuck = 0
            logger.debug(f"Plate {sp_length}x{sp_width} cannot fit")
            return False

    def can_fit_in_gap(self, small_plate, gap):
        sp_length, sp_width, _ = small_plate
        start_x, start_y, end_x, end_y = gap
        fits = start_x + sp_length <= end_x and start_y + sp_width <= end_y
        if fits:
            logger.debug(f"Plate {sp_length}x{sp_width} can fit in gap at ({start_x}, {start_y})")
        return fits

    def add_cut_in_gap(self, small_plate):
        if not self.available_gaps:
            return False
        
        sp_length, sp_width, _ = small_plate
        for gap in self.available_gaps[:]:
            if self.can_fit_in_gap(small_plate, gap):
                start_x, start_y, _, _ = gap
                self.cuts.append((small_plate, start_x, start_y, start_x + sp_length, start_y + sp_width))
                self.used_area += sp_length * sp_width
                self.available_gaps.remove(gap)
                logger.info(f"Cut plate {sp_length}x{sp_width} in gap at ({start_x}, {start_y})")

                # 更新缝隙逻辑
                if gap[0] + sp_length + self.blade_thick < gap[2]:
                    new_gap = (gap[0] + sp_length + self.blade_thick, gap[1], gap[2], gap[1] + sp_width + self.blade_thick)
                    self.available_gaps.append(new_gap)
                    logger.debug(f"Created new horizontal gap: {new_gap}")
                
                if gap[1] + sp_width + self.blade_thick < gap[3]:
                    # (此处的复杂缝隙合并逻辑保持不变, 但已属于优化范畴)
                    new_gap = (gap[0], gap[1] + sp_width + self.blade_thick, gap[2], gap[3])
                    self.available_gaps.append(new_gap)
                    logger.debug(f"Created new vertical gap: {new_gap}")

                return True
        logger.debug(f"Could not fit plate {sp_length}x{sp_width} in any available gap")
        return False

    def update_gap(self):
        gap0 = (self.current_x, self.current_y, self.length, self.current_y + self.row_width)
        if self.available_gaps and gap0 not in self.available_gaps:
             for gap in self.available_gaps[:]:
                if (self.current_x <= gap[0] + self.tolerant and 
                    self.current_x > gap[0] - self.tolerant and 
                    self.current_y == gap[3]):
                    self.available_gaps.remove(gap)
                    new_gap = (max(self.current_x, gap[0]), gap[1], self.length, self.current_y + self.row_width)
                    self.available_gaps.append(new_gap)
                    logger.debug(f"Updated gap: {new_gap}")
                    return
        self.available_gaps.append(gap0)
        logger.debug(f"Added new gap: {gap0}")
    
    def add_cut(self, small_plate):
        x1, x2 = small_plate[:2]
        if (x2, x1) == self.last_cut[:2] and x1 and x2 and (x1*x2 < self.show or abs(x1- x2)/max(x1,x2) < self.ratio):
            small_plate0 = (x2, x1, small_plate[2])
        else:
            small_plate0 = small_plate

        if not self.can_fit(small_plate0):
            return False

        sp_length, sp_width, _ = small_plate0
        if self.current_x + sp_length <= self.length:
            if (small_plate0[:2] != self.last_cut and 
                (self.length//sp_width)*((self.tolerant + self.row_width)//sp_length) > 
                (self.length//sp_length)*((self.tolerant + self.row_width)//sp_width) and 
                (sp_length*sp_width < self.show or abs(sp_length - sp_width)/max(sp_length, sp_width) < self.ratio)):
                sp_length, sp_width = sp_width, sp_length
                logger.debug("Rotated plate for better fit")
                
            start_x, start_y = self.current_x, self.current_y
            self.current_x += sp_length + self.blade_thick
            
            if sp_width + self.blade_thick < self.row_width - self.tolerant:
                gap = (start_x, start_y + sp_width + self.blade_thick, self.current_x, start_y + self.row_width)
                self._update_gap_after_cut(gap)
            elif sp_width + self.blade_thick > self.row_width + self.tolerant:
                new_gap = (0, start_y + self.row_width, start_x, start_y + sp_width + self.blade_thick)
                self.available_gaps.append(new_gap)
                logger.debug(f"Added gap for excess height: {new_gap}")
                
            self.row_width = max(self.row_width, sp_width + self.blade_thick)
            if start_x == 0 and start_y == 0 and start_y + self.row_width <= self.width and start_y + self.row_width > self.width - self.attach:
                self.row_width = self.width - start_y
                new_gap = (start_x, start_y + sp_width + self.blade_thick, self.current_x, self.width)
                self.available_gaps.append(new_gap)
                logger.debug(f"Added edge gap: {new_gap}")
        else:
            if ((self.length//sp_width)*((self.width - self.row_width)//sp_length) > 
                (self.length//sp_length)*((self.width - self.row_width)//sp_width) and 
                abs(sp_length - sp_width)/max(sp_length, sp_width) < self.ratio):
                sp_length, sp_width = sp_width, sp_length
                logger.debug("Rotated plate for new row")
                
            self.current_x = 0
            self.current_y += self.row_width
            start_x, start_y = self.current_x, self.current_y
            self.current_x += sp_length + self.blade_thick
            self.row_width = sp_width + self.blade_thick

            if start_y + self.row_width <= self.width and start_y + self.row_width > self.width - self.attach:
                new_gap = (start_x, start_y + self.row_width, self.current_x, self.width)
                self.available_gaps.append(new_gap)
                self.row_width = self.width - start_y
                logger.debug(f"Added new row edge gap: {new_gap}")

        end_x, end_y = start_x + sp_length, start_y + sp_width
        if self.available_gaps:
            self.available_gaps = [gap for gap in self.available_gaps[:] if not (start_x == gap[0] and start_y == gap[1])]

        self.cuts.append((small_plate, start_x, start_y, end_x, end_y))
        self.used_area += sp_length * sp_width
        self.last_cut = (sp_length, sp_width)
        logger.info(f"Cut plate {sp_length}x{sp_width} at ({start_x}, {start_y})")
        return True

    def _update_gap_after_cut(self, gap):
        for gap0 in self.available_gaps:
            if (gap0[3] == gap[3] and gap0[2] == gap[0] and 
                abs(gap0[1] - gap[1]) < self.tolerant):
                new_gap = (gap0[0], max(gap0[1], gap[1]), gap[2], gap[3])
                self.available_gaps.append(new_gap)
                self.available_gaps.remove(gap0)
                logger.debug(f"Updated gap after cut: {new_gap}")
                return
        self.available_gaps.append(gap)
        logger.debug(f"Added new gap after cut: {gap}")
    
    def utilization(self):
        if self.length * self.width == 0:
            return 0.0
        self.used_perc = (self.used_area / (self.length * self.width)) * 100
        logger.info(f"Plate utilization: {self.used_perc:.2f}%")
        return self.used_perc

def optimize_cutting(plates, orders, others, optim=0, n_plate=None, saw_blade=4):
    logger.info("Starting optimization process")
    
    # --- 数据准备部分 (与原代码相同) ---
    big_plate_list = []
    for p in plates:
        for _ in range(p.get('quantity', 0)):
            big_plate_list.append((p['length'], p['width']))
    if not big_plate_list:
        logger.warning("No valid plates found in input")
        return []

    small_plates_list = [(p['length'], p['width'], p.get('id', ''), p['quantity']) 
                        for p in orders if p.get('quantity', 0) > 0]
    
    stock_plates_list = [(p['length'], p['width'], f"R{p.get('id', '')}")
                        for p in others if p.get('length', 0) > 0 and p.get('width', 0) > 0] if others else []

    logger.info(f"Processed input: {len(big_plate_list)} plates, {len(small_plates_list)} orders, {len(stock_plates_list)} stock items")
    
    big_plate_objects = [Plate(length, width, saw_blade) for length, width in big_plate_list]
    
    # 根据切割效率对小板进行排序和旋转
    small_plates2 = []
    length0, width0 = big_plate_list[0]
    for x1, x2, x3, nums in small_plates_list:
        if (x1 < x2 and (length0//x2)*(width0//x1) >= (length0//x1)*(width0//x2)) or (abs(x1- x2)/max(x1,x2) < 0.56 and (length0//x2)*(width0//x1) > (length0//x1)*(width0//x2)):
            small_plates2.append((x2, x1, x3, nums))
            logger.debug(f"Rotated order plate: {x1}x{x2} -> {x2}x{x1}")
        else:
            small_plates2.append((x1, x2, x3, nums))
    
    width_counts = Counter(plate[1] for plate in small_plates2)
    for i in width_counts.keys():
        if i < 400:
            width_counts[i] = 0
    small_plates2.sort(key=lambda x: (width_counts[x[1]], x[1], x[0]), reverse=True)
    
    # 将订单展开成单个板材列表
    small_plate_objects = [(x1, x2, x3) for x1, x2, x3, nums in small_plates2 for _ in range(nums)]
    logger.info(f"Processed {len(small_plate_objects)} total pieces to cut")

    # 主切割循环
    used_plates = []
    remaining_pieces = small_plate_objects.copy()

    for i, big_plate in enumerate(big_plate_objects):
        logger.info(f"Processing plate {i+1}/{len(big_plate_objects)}")
        if not remaining_pieces:
            logger.info("All pieces have been cut")
            break
        
        pieces_to_place_this_round = remaining_pieces.copy()
        placed_indices = []

        for j, small_plate in enumerate(pieces_to_place_this_round):
            logger.debug(f"Attempting to cut piece {j+1}/{len(pieces_to_place_this_round)}")
            placed = False
            small_plate1 = (small_plate[1], small_plate[0], small_plate[2])

            # 优先在缝隙中放置
            if big_plate.available_gaps:
                if big_plate.add_cut_in_gap(small_plate):
                    placed = True
                elif big_plate.add_cut_in_gap(small_plate1): # 尝试旋转后放入缝隙
                    placed = True
            
            # 如果缝隙中放不下，则在主区域放置
            if not placed:
                if big_plate.add_cut(small_plate):
                    placed = True
            
            if placed:
                placed_indices.append(j)
                logger.debug(f"Piece {small_plate} placed.")

        # 如果当前大板有任何切割操作，则将其加入已用列表
        if placed_indices:
            used_plates.append(big_plate)
            # 从后往前删除，避免索引变化导致错误
            for index in sorted(placed_indices, reverse=True):
                del remaining_pieces[index]
    
    big_plate_objects = used_plates

    if stock_plates_list:
        logger.info("Processing stock plates")
        stock_plate_objects = [(max(x1, x2), min(x1, x2), x3) for x1, x2, x3 in stock_plates_list]
        for big_plate in big_plate_objects:
            if not optim:
                _process_stock_plates(big_plate, stock_plate_objects)
            else:
                _optimize_stock_placement(big_plate, stock_plate_objects)

    for big_plate in big_plate_objects:
        big_plate.utilization()

    # 转换为cutted格式输出
    cutted = []
    for big_plate in big_plate_objects:
        if big_plate.cuts:
            plate_cuts = []
            for cut in big_plate.cuts:
                is_stock = str(cut[0][2]).startswith('R')
                plate_cuts.append([
                    cut[1], # start_x
                    cut[2], # start_y  
                    cut[0][0], # length
                    cut[0][1], # width
                    1 if is_stock else 0,
                    cut[0][2][1:] if is_stock else cut[0][2]
                ])
            cutted.append({
                'rate': big_plate.used_perc / 100,
                'plate': [big_plate.length, big_plate.width],
                'cutted': plate_cuts
            })
            
    logger.info(f"Optimization complete. Generated {len(cutted)} cutting plans")
    return cutted

def _process_stock_plates(big_plate, stock_plate_objects):
    logger.debug("Processing stock plates without optimization")
    for stock_plate in stock_plate_objects:
        try:
            n = int((100 - big_plate.utilization())/100*(big_plate.length * big_plate.width)/(stock_plate[0]*stock_plate[1]))
        except ZeroDivisionError:
            n = 0
        logger.debug(f"Attempting to place {n} copies of stock plate {stock_plate[0]}x{stock_plate[1]}")
        for _ in range(n):
            if not big_plate.add_cut_in_gap(stock_plate):
                stock_plate1 = (stock_plate[1], stock_plate[0], stock_plate[2])
                if not big_plate.add_cut_in_gap(stock_plate1):
                    if not big_plate.add_cut(stock_plate):
                        big_plate.add_cut(stock_plate1)

def _optimize_stock_placement(big_plate, stock_plate_objects):
    # 此函数逻辑较为复杂且依赖于 _process_stock_plates 的正确性
    # 由于 _process_stock_plates 已恢复到有问题的版本，此处的优化效果可能不符合预期
    logger.debug("Optimizing stock plate placement")
    base_util = big_plate.utilization()
    n_optim = -1 # 使用-1表示尚未找到优化方案
    
    # 尝试将每个库存板作为第一个填充的板，看哪个效果最好
    for i in range(min(10, len(stock_plate_objects))):
        # 创建一个深拷贝的板材对象用于模拟
        big_plate_sim = copy.deepcopy(big_plate)
        
        # 创建一个重新排序的库存列表
        stock_order_sim = stock_plate_objects.copy()
        first_stock = stock_order_sim.pop(i)
        stock_order_sim.insert(0, first_stock)
        
        _process_stock_plates(big_plate_sim, stock_order_sim)
        
        sim_util = big_plate_sim.utilization()
        if sim_util > base_util:
            base_util = sim_util
            n_optim = i
            logger.debug(f"Found better utilization: {base_util:.2f}% with stock plate {i} first")

    # 如果找到了更优的顺序，则按该顺序在原始板材上进行实际填充
    if n_optim != -1:
        final_stock_order = stock_plate_objects.copy()
        first_stock = final_stock_order.pop(n_optim)
        final_stock_order.insert(0, first_stock)
        _process_stock_plates(big_plate, final_stock_order)
        logger.info(f"Final utilization after stock optimization: {big_plate.utilization():.2f}%")
    else:
        # 如果没有找到更优解，则按原顺序填充
        _process_stock_plates(big_plate, stock_plate_objects)