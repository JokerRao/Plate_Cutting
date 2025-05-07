import copy
from collections import Counter
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('plate_cutting')

# 定义一个板材类来表示大板
class Plate:
    def __init__(self, length, width, blade_thick=4):
        self.length = length  # 板材长度
        self.width = width    # 板材宽度
        self.used_area = 0    # 已使用面积
        self.cuts = []        # 存储切割信息,格式为(小板尺寸,起点x,起点y,终点x,终点y)
        self.current_x = 0    # 当前切割位置的x坐标
        self.current_y = 0    # 当前切割位置的y坐标
        self.row_width = 0    # 当前行的宽度
        self.current_x1 = 0   # 缝隙切割时的x坐标
        self.current_y1 = 0   # 缝隙切割时的y坐标
        self.row_width1 = 0   # 缝隙切割时的行宽
        self.available_gaps = [] # 可用缝隙列表,格式为(起点x,起点y,终点x,终点y)
        self.used_perc = 0    # 利用率
        self.blade_thick = blade_thick
        self.tolerant = 30    # 容差值
        self.attach = 100     # 边缘附着宽度
        self.stuck = 0        # 卡住标记
        self.last_cut = (0, 0) # 上一次切割尺寸
        self.ratio = 0.4      # 长宽比阈值
        self.show = 1.2*10**5 # 面积阈值
        logger.debug(f"Created new plate: {length}x{width}, blade thickness: {blade_thick}")

    def can_fit(self, small_plate):
        """检查小板是否能放入当前行或开始新行"""
        sp_length, sp_width, _ = small_plate

        # 检查是否能放入当前行
        if self.current_x + sp_length <= self.length and self.current_y + sp_width <= self.width:
            logger.debug(f"Plate {sp_length}x{sp_width} can fit at current position ({self.current_x}, {self.current_y})")
            return True
        
        # 如果不能放入当前行,检查是否可以开始新行
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
        """检查小板是否能放入指定缝隙"""
        sp_length, sp_width, _ = small_plate
        start_x, start_y, end_x, end_y = gap
        self.current_x1 = start_x
        self.current_y1 = start_y
        # 检查尺寸是否合适
        fits = start_x + sp_length <= end_x and start_y + sp_width <= end_y
        if fits:
            logger.debug(f"Plate {sp_length}x{sp_width} can fit in gap at ({start_x}, {start_y})")
        return fits
    
    def add_cut_in_gap(self, small_plate):
        """在缝隙中切割小板并添加到切割列表"""
        if not self.available_gaps:
            logger.debug("No available gaps for cutting")
            return False
            
        sp_length, sp_width, _ = small_plate
        for gap in self.available_gaps:
            start_x, start_y, end_x, end_y = gap
            if self.can_fit_in_gap(small_plate, gap):
                self.cuts.append((small_plate, start_x, start_y, start_x + sp_length, start_y + sp_width))
                self.used_area += sp_length * sp_width
                self.available_gaps.remove(gap)
                logger.info(f"Cut plate {sp_length}x{sp_width} in gap at ({start_x}, {start_y})")
                
                if gap[0] + sp_length + self.blade_thick < gap[2]:
                    new_gap = (gap[0] + sp_length + self.blade_thick, gap[1], gap[2], gap[1] + sp_width + self.blade_thick)
                    self.available_gaps.append(new_gap)
                    logger.debug(f"Created new horizontal gap: {new_gap}")
                    
                if gap[1] + sp_width + self.blade_thick < gap[3]:
                    if gap[3] == self.width and gap[0] != 0:
                        for gap0 in self.available_gaps:
                            if (gap0[3] == self.width and gap0[2] == gap[0] and 
                                abs(gap0[1] - (gap[1] + sp_width + self.blade_thick)) < self.tolerant):
                                new_gap = (gap0[0], max(gap0[1], gap[1] + sp_width + self.blade_thick), gap[2], gap[3])
                                self.available_gaps.append(new_gap)
                                self.available_gaps.remove(gap0)
                                logger.debug(f"Updated existing gap: {new_gap}")
                                break
                        else:
                            new_gap = (gap[0], gap[1] + sp_width + self.blade_thick, gap[2], gap[3])
                            self.available_gaps.append(new_gap)
                            logger.debug(f"Created new vertical gap: {new_gap}")
                    else:
                        new_gap = (gap[0], gap[1] + sp_width + self.blade_thick, gap[2], gap[3])
                        self.available_gaps.append(new_gap)
                        logger.debug(f"Created new vertical gap: {new_gap}")
                return True
        logger.debug(f"Could not fit plate {sp_length}x{sp_width} in any available gap")
        return False

    def update_gap(self):
        """更新可用缝隙信息"""
        gap0 = (self.current_x, self.current_y, self.length, self.current_y + self.row_width)
        if self.available_gaps and gap0 not in self.available_gaps:
            for gap in self.available_gaps:
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
        """切割小板并添加到切割列表"""
        x1, x2 = small_plate[:2]
        # 根据长宽比和面积判断是否需要旋转
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
            self.available_gaps = [gap for gap in self.available_gaps if not (start_x == gap[0] and start_y == gap[1])]

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
        """计算板材利用率"""
        self.used_perc =  (self.used_area / (self.length * self.width)) * 100
        logger.info(f"Plate utilization: {self.used_perc:.2f}%")
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
        quantity = p.get('quantity', 0)
        if quantity > 0:
            for _ in range(quantity):
                big_plate_list.append((p['length'], p['width']))
    if not big_plate_list:
        logger.warning("No valid plates found in input")
        return []
    
    # 转换小板格式 (orders)
    small_plates_list = [(p['length'], p['width'], p.get('id', ''), p['quantity']) 
                        for p in orders if p.get('quantity', 0) > 0]
    
    # 转换库存板格式 (others)
    stock_plates_list = [(p['length'], p['width'], p.get('id', ''))
                        for p in others if p.get('length', 0) > 0 and p.get('width', 0) > 0] if others else []

    logger.info(f"Processed input: {len(big_plate_list)} plates, {len(small_plates_list)} orders, {len(stock_plates_list)} stock items")

    # 创建大板对象列表
    big_plate_objects = [Plate(length, width, saw_blade) for length, width in big_plate_list]
    small_plates1 = small_plates_list.copy()
    
    if n_plate is None:
        n_plate = len(big_plate_list)
    
    # 处理库存板材,确保长边在前
    stock_plate_objects = [(max(x1, x2), min(x1, x2), x3) for x1, x2, x3 in stock_plates_list] if stock_plates_list else []
    length0 = big_plate_list[0][0]
    width0 = big_plate_list[0][1]
    
    # 根据切割效率对小板进行排序和旋转
    small_plates2 = []
    for x1, x2, x3, nums in small_plates1:
        # 判断是否需要旋转小板材以获得更好的切割效率
        if (x1 < x2 and (length0//x2)*(width0//x1) >= (length0//x1)*(width0//x2)) or (abs(x1- x2)/max(x1,x2) < 0.56 and (length0//x2)*(width0//x1) > (length0//x1)*(width0//x2)):
            # 交换长宽并添加到small_plates2
            small_plates2.append((x2, x1, x3, nums))
            logger.debug(f"Rotated order plate: {x1}x{x2} -> {x2}x{x1}")
        else:
            # 保持原有尺寸添加到small_plates2
            small_plates2.append((x1, x2, x3, nums))
            
    # 统计宽度出现次数并排序
    width_counts = Counter(plate[1] for plate in small_plates2)
    for i in width_counts.keys():
        if i < 400:
            width_counts[i] = 0
    small_plates2.sort(key=lambda x: (width_counts[x[1]], x[1], x[0]), reverse=True)
    small_plate_objects = [(x1, x2, x3) for x1, x2, x3, nums in small_plates2 for _ in range(nums)]
    logger.info(f"Processed {len(small_plate_objects)} total pieces to cut")

    # 主切割循环
    used_plates = []
    for i, big_plate in enumerate(big_plate_objects):
        logger.info(f"Processing plate {i+1}/{len(big_plate_objects)}")
        if not small_plate_objects:
            logger.info("All pieces have been cut")
            break
        small_plate_objects0 = copy.deepcopy(small_plate_objects)
        plate_used = False
        for j, small_plate in enumerate(small_plate_objects0):
            logger.debug(f"Attempting to cut piece {j+1}/{len(small_plate_objects0)}")
            placed = False
            small_plate1 = (small_plate[1],small_plate[0],small_plate[2])

            if big_plate.available_gaps:
                if not placed and big_plate.add_cut_in_gap(small_plate):
                    placed = True
                    small_plate_objects.remove(small_plate)
                    plate_used = True
                    logger.debug("Piece placed in gap")

                small_plates3 = [(x1, x2, x3) for x1, x2, x3, nums in small_plates2]
                idx0 = small_plates3.index(small_plate)

                left0 = any((gap[2]-gap[0])*(gap[3]-gap[1]) > big_plate.show and 
                        any(sp[0]*sp[1] > big_plate.show and big_plate.can_fit_in_gap(sp, gap)
                            for sp in small_plates3[idx0+1:])
                        for gap in big_plate.available_gaps)

                if not placed and not left0 and big_plate.add_cut_in_gap(small_plate1):
                    placed = True
                    small_plate_objects.remove(small_plate)
                    plate_used = True
                    logger.debug("Rotated piece placed in gap")
                
            if not placed and big_plate.add_cut(small_plate):
                placed = True
                small_plate_objects.remove(small_plate)
                plate_used = True
                logger.debug("Piece placed in main area")
        
        if plate_used:
            used_plates.append(big_plate)
    
    big_plate_objects = used_plates

    # 只有当stock_plates存在且有数据时才处理库存板材
    if stock_plates_list:
        logger.info("Processing stock plates")
        for i, big_plate in enumerate(big_plate_objects):
            logger.debug(f"Processing stock plates for plate {i+1}/{len(big_plate_objects)}")
            if not optim:
                _process_stock_plates(big_plate, stock_plate_objects)
            else:
                _optimize_stock_placement(big_plate, stock_plate_objects)

    # 计算最终利用率
    for big_plate in big_plate_objects:
        big_plate.utilization()

    # 转换为cutted格式输出
    cutted = []
    for big_plate in big_plate_objects:
        if big_plate.cuts:  # 只添加有切割记录的板
            plate_cuts = []
            for cut in big_plate.cuts:
                # 检查切割板材的ID是否以R开头来判断是否为库存板
                is_stock = str(cut[0][2]).startswith('R')
                plate_cuts.append([
                    cut[1], # start_x
                    cut[2], # start_y  
                    cut[0][0], # length
                    cut[0][1], # width
                    1 if is_stock else 0,  # is_stock
                    cut[0][2][1:] if is_stock else cut[0][2] # id
                ])
            cutted.append({
                'rate': big_plate.used_perc/100,
                'plate': [big_plate.length, big_plate.width],
                'cutted': plate_cuts
            })
            
    logger.info(f"Optimization complete. Generated {len(cutted)} cutting plans")
    return cutted

def _process_stock_plates(big_plate, stock_plate_objects):
    logger.debug("Processing stock plates without optimization")
    for stock_plate in stock_plate_objects:
        n = int((100 - big_plate.utilization())/100*(big_plate.length * big_plate.width)/(stock_plate[0]*stock_plate[1]))
        logger.debug(f"Attempting to place {n} copies of stock plate {stock_plate[0]}x{stock_plate[1]}")
        for _ in range(n):
            if not big_plate.add_cut(stock_plate):
                stock_plate1 = (stock_plate[1], stock_plate[0], stock_plate[2])
                if not big_plate.add_cut_in_gap(stock_plate1):
                    big_plate.add_cut_in_gap(stock_plate)

def _optimize_stock_placement(big_plate, stock_plate_objects):
    logger.debug("Optimizing stock plate placement")
    base_util = big_plate.used_perc
    n_optim = 0
    
    for i in range(min(10, len(stock_plate_objects))):
        stock_plate_objects0 = stock_plate_objects.copy()
        tcp0 = stock_plate_objects0.pop(i)
        stock_plate_objects0 = [tcp0] + stock_plate_objects0
        big_plate0 = copy.deepcopy(big_plate)
        
        _process_stock_plates(big_plate0, stock_plate_objects0)
        
        if big_plate0.utilization() > base_util:
            base_util = big_plate0.used_perc
            n_optim = i
            logger.debug(f"Found better utilization: {base_util:.2f}% with stock plate {i}")

    stock_plate_objects0 = stock_plate_objects.copy()
    tcp0 = stock_plate_objects0.pop(n_optim)
    stock_plate_objects0 = [tcp0] + stock_plate_objects0
    _process_stock_plates(big_plate, stock_plate_objects0)
    logger.info(f"Final utilization after stock optimization: {big_plate.used_perc:.2f}%")