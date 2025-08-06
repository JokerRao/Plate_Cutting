# 导入所需的库
import copy  # 用于创建对象的深拷贝，特别是在模拟优化时，避免修改原始对象
from collections import Counter  # 用于统计列表中元素的频率，如此处用来统计小板宽度
import logging  # 用于记录程序运行过程中的信息、调试信息和错误

# --- 配置日志系统 ---
# logging.basicConfig 用于为日志系统进行一次性配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志记录的最低级别。INFO表示只记录一般信息、警告和错误，DEBUG会记录更详细的调试信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 定义日志输出的格式，包含时间、日志名称、级别和消息内容
)
# 获取一个名为 'plate_cutting' 的日志记录器实例
logger = logging.getLogger('plate_cutting')

# --- 板材类定义 ---
# 定义一个类来表示每一块大的原材料板材及其状态
class Plate:
    # 初始化方法，当创建一个新的 Plate 对象时被调用
    def __init__(self, length, width, blade_thick=4):
        self.length = length  # 大板的长度
        self.width = width  # 大板的宽度
        self.used_area = 0  # 已切割使用的面积，用于计算利用率
        self.cuts = []  # 一个列表，用于存储所有已切割的小板信息 (小板尺寸, x1, y1, x2, y2)
        self.current_x = 0  # 当前切割行的光标 x 坐标，表示下一块板可以开始放置的 x 位置
        self.current_y = 0  # 当前切割行的 y 坐标，表示当前行的起始 y 位置
        self.row_width = 0  # 当前切割行的最大宽度（高度）
        self.available_gaps = []  # 可用的余料区域列表，每个元素是一个元组 (x1, y1, x2, y2) 表示一个矩形空隙
        self.used_perc = 0  # 板材利用率的百分比
        self.blade_thick = blade_thick  # 锯片厚度，每次切割都会损耗这个宽度
        self.tolerant = 30  # 一个容差值，用于判断缝隙是否可以合并等
        self.attach = 100 # 一个边界吸附距离，当剩余宽度小于此值时，可能会尝试填满整行
        self.stuck = 0 # 一个状态标志，用于避免在某些情况下陷入死循环或重复计算
        self.last_cut = (0, 0) # 记录上一次切割的小板尺寸 (长, 宽)，用于启发式决策
        self.ratio = 0.4 # 一个比例阈值，用于判断是否值得旋转小板
        self.show = 1.2 * 10**5 # 一个面积阈值，用于某些启发式决策
        logger.debug(f"创建新板材: {length}x{width}, 锯片厚度: {blade_thick}")

    # 判断一个小板是否能放在当前行的剩余空间或新的一行
    def can_fit(self, small_plate):
        sp_length, sp_width, _ = small_plate # 解包获取小板的长和宽

        # 检查是否能放在当前行的当前位置
        if self.current_x + sp_length <= self.length and self.current_y + sp_width <= self.width:
            logger.debug(f"小板 {sp_length}x{sp_width} 可以放入当前位置 ({self.current_x}, {self.current_y})")
            return True
        # 如果当前行放不下，检查是否能开启新的一行来放置
        elif self.current_y + self.row_width + sp_width <= self.width:
            # 如果可以开启新行，先更新上一行的余料区域
            if not self.stuck:
                self.update_gap()
            self.stuck = 0 # 重置标志
            logger.debug(f"小板 {sp_length}x{sp_width} 可以放入新的一行")
            return True
        # 如果新的一行也放不下，说明板材剩余空间不足
        else:
            # 这是一个回退或更新逻辑，确保在判断失败后，状态被正确更新
            if not self.stuck:
                self.update_gap()
                self.current_x = 0
                self.current_y += self.row_width
                self.row_width = 0
                self.stuck = 0
            logger.debug(f"小板 {sp_length}x{sp_width} 无法放入")
            return False

    # 检查一个小板是否能放入一个指定的缝隙（余料区域）
    def can_fit_in_gap(self, small_plate, gap):
        sp_length, sp_width, _ = small_plate # 获取小板尺寸
        start_x, start_y, end_x, end_y = gap # 获取缝隙的坐标范围
        # 判断小板的长宽是否都小于等于缝隙的长宽
        fits = start_x + sp_length <= end_x and start_y + sp_width <= end_y
        if fits:
            logger.debug(f"小板 {sp_length}x{sp_width} 可以放入缝隙 ({start_x}, {start_y})")
        return fits

    # 尝试将一个小板放入一个可用的缝隙中
    def add_cut_in_gap(self, small_plate):
        if not self.available_gaps: # 如果没有可用缝隙，直接返回失败
            return False
        
        sp_length, sp_width, _ = small_plate
        # 遍历所有可用缝隙（使用切片[:]是为了安全地在循环中修改列表）
        for gap in self.available_gaps[:]:
            if self.can_fit_in_gap(small_plate, gap):
                start_x, start_y, _, _ = gap # 获取缝隙的起始坐标
                # 记录这次切割
                self.cuts.append((small_plate, start_x, start_y, start_x + sp_length, start_y + sp_width))
                self.used_area += sp_length * sp_width # 更新已用面积
                self.available_gaps.remove(gap) # 移除被占用的整个缝隙
                logger.info(f"在缝隙 ({start_x}, {start_y}) 中切割了小板 {sp_length}x{sp_width}")

                # --- 更新缝隙逻辑 ---
                # 在被占用的缝隙中，可能会产生新的更小的缝隙
                # 1. 右侧产生新缝隙
                if gap[0] + sp_length + self.blade_thick < gap[2]:
                    new_gap = (gap[0] + sp_length + self.blade_thick, gap[1], gap[2], gap[1] + sp_width + self.blade_thick)
                    self.available_gaps.append(new_gap)
                    logger.debug(f"产生新的水平缝隙: {new_gap}")
                
                # 2. 下方产生新缝隙
                if gap[1] + sp_width + self.blade_thick < gap[3]:
                    # (此处的复杂缝隙合并逻辑保持不变, 但已属于优化范畴)
                    new_gap = (gap[0], gap[1] + sp_width + self.blade_thick, gap[2], gap[3])
                    self.available_gaps.append(new_gap)
                    logger.debug(f"产生新的垂直缝隙: {new_gap}")

                return True # 成功放入，结束函数
        logger.debug(f"无法在任何可用缝隙中放入小板 {sp_length}x{sp_width}")
        return False # 遍历完所有缝隙都放不下，返回失败

    # 当一行填满后，将该行剩余的水平空间作为一个新的可用缝隙记录下来
    def update_gap(self):
        # 定义当前行在x方向上剩余的矩形区域
        gap0 = (self.current_x, self.current_y, self.length, self.current_y + self.row_width)
        # 检查是否可以与已有的缝隙合并（一个复杂的启发式合并逻辑）
        if self.available_gaps and gap0 not in self.available_gaps:
             for gap in self.available_gaps[:]:
                # 如果新缝隙的左上角与某个旧缝隙的左下角在垂直方向上对齐
                if (self.current_x <= gap[0] + self.tolerant and 
                    self.current_x > gap[0] - self.tolerant and 
                    self.current_y == gap[3]):
                    self.available_gaps.remove(gap) # 移除旧缝隙
                    # 创建一个合并后的新缝隙
                    new_gap = (max(self.current_x, gap[0]), gap[1], self.length, self.current_y + self.row_width)
                    self.available_gaps.append(new_gap)
                    logger.debug(f"更新并合并缝隙: {new_gap}")
                    return
        # 如果不能合并，则直接添加为新缝隙
        self.available_gaps.append(gap0)
        logger.debug(f"添加新缝隙: {gap0}")
    
    # 在主区域（非缝隙）添加一次切割
    def add_cut(self, small_plate):
        x1, x2 = small_plate[:2] # 获取小板长宽
        # 启发式逻辑：如果当前小板的尺寸与上一次切割的小板尺寸正好相反，并且满足某些条件（面积小或形状接近正方形），则使用旋转后的尺寸
        if (x2, x1) == self.last_cut[:2] and x1 and x2 and (x1*x2 < self.show or abs(x1- x2)/max(x1,x2) < self.ratio):
            small_plate0 = (x2, x1, small_plate[2])
        else:
            small_plate0 = small_plate

        # 首先检查是否能放下
        if not self.can_fit(small_plate0):
            return False

        sp_length, sp_width, _ = small_plate0
        # 如果当前行在x方向上还有足够空间
        if self.current_x + sp_length <= self.length:
            # 启发式旋转：如果旋转后，预估能放下更多的小板，则进行旋转
            if (small_plate0[:2] != self.last_cut and 
                (self.length//sp_width)*((self.tolerant + self.row_width)//sp_length) > 
                (self.length//sp_length)*((self.tolerant + self.row_width)//sp_width) and 
                (sp_length*sp_width < self.show or abs(sp_length - sp_width)/max(sp_length, sp_width) < self.ratio)):
                sp_length, sp_width = sp_width, sp_length
                logger.debug("为获得更好拟合效果而旋转板材")
                
            start_x, start_y = self.current_x, self.current_y # 记录切割起始点
            self.current_x += sp_length + self.blade_thick # 光标向右移动
            
            # 如果放置后，在当前行内，该小板的下方留出了可观的空隙
            if sp_width + self.blade_thick < self.row_width - self.tolerant:
                gap = (start_x, start_y + sp_width + self.blade_thick, self.current_x, start_y + self.row_width)
                self._update_gap_after_cut(gap) # 尝试将此空隙与现有缝隙合并
            # 如果放置后，该小板超出了当前行的高度
            elif sp_width + self.blade_thick > self.row_width + self.tolerant:
                # 在行的左侧，旧行和新行高度之间产生一个缝隙
                new_gap = (0, start_y + self.row_width, start_x, start_y + sp_width + self.blade_thick)
                self.available_gaps.append(new_gap)
                logger.debug(f"为超出行高的部分添加缝隙: {new_gap}")
                
            self.row_width = max(self.row_width, sp_width + self.blade_thick) # 更新行宽为当前行所有板材宽度的最大值
            # 启发式吸附：如果当前行接近大板的边缘，则将行宽扩展到边缘，以避免产生无法利用的窄条
            if start_x == 0 and start_y == 0 and start_y + self.row_width <= self.width and start_y + self.row_width > self.width - self.attach:
                self.row_width = self.width - start_y
                new_gap = (start_x, start_y + sp_width + self.blade_thick, self.current_x, self.width)
                self.available_gaps.append(new_gap)
                logger.debug(f"添加边缘缝隙: {new_gap}")
        # 如果当前行放不下，需要换行
        else:
            # 换行前，同样进行启发式旋转判断
            if ((self.length//sp_width)*((self.width - self.row_width)//sp_length) > 
                (self.length//sp_length)*((self.width - self.row_width)//sp_width) and 
                abs(sp_length - sp_width)/max(sp_length, sp_width) < self.ratio):
                sp_length, sp_width = sp_width, sp_length
                logger.debug("为放入新行而旋转板材")
                
            self.current_x = 0 # x光标回到最左边
            self.current_y += self.row_width # y光标移动到新行的起始位置
            start_x, start_y = self.current_x, self.current_y # 记录新行的第一个切割的起始点
            self.current_x += sp_length + self.blade_thick # 移动x光标
            self.row_width = sp_width + self.blade_thick # 设置新行的宽度

            # 同样，对新行进行边缘吸附判断
            if start_y + self.row_width <= self.width and start_y + self.row_width > self.width - self.attach:
                new_gap = (start_x, start_y + self.row_width, self.current_x, self.width)
                self.available_gaps.append(new_gap)
                self.row_width = self.width - start_y
                logger.debug(f"为新行添加边缘缝隙: {new_gap}")

        end_x, end_y = start_x + sp_length, start_y + sp_width # 计算切割的结束点
        # 清理可能由于放置操作而已被完全填充的旧缝隙
        if self.available_gaps:
            self.available_gaps = [gap for gap in self.available_gaps[:] if not (start_x == gap[0] and start_y == gap[1])]

        self.cuts.append((small_plate, start_x, start_y, end_x, end_y)) # 记录本次切割
        self.used_area += sp_length * sp_width # 更新已用面积
        self.last_cut = (sp_length, sp_width) # 记录最后一次切割的尺寸
        logger.info(f"在 ({start_x}, {start_y}) 位置切割了小板 {sp_length}x{sp_width}")
        return True

    # 辅助函数，在一次切割后，尝试将新产生的缝隙与已有缝隙合并
    def _update_gap_after_cut(self, gap):
        for gap0 in self.available_gaps:
            # 如果新缝隙gap和某个旧缝隙gap0在y方向上连续且在x方向上相邻
            if (gap0[3] == gap[3] and gap0[2] == gap[0] and 
                abs(gap0[1] - gap[1]) < self.tolerant):
                # 合并成一个更大的缝隙
                new_gap = (gap0[0], max(gap0[1], gap[1]), gap[2], gap[3])
                self.available_gaps.append(new_gap)
                self.available_gaps.remove(gap0)
                logger.debug(f"切割后更新缝隙: {new_gap}")
                return
        # 如果不能合并，则直接添加新缝隙
        self.available_gaps.append(gap)
        logger.debug(f"切割后添加新缝隙: {gap}")
    
    # 计算并返回板材的利用率
    def utilization(self):
        if self.length * self.width == 0: # 避免除以零
            return 0.0
        # 利用率 = (已用面积 / 总面积) * 100
        self.used_perc = (self.used_area / (self.length * self.width)) * 100
        logger.info(f"板材利用率: {self.used_perc:.2f}%")
        return self.used_perc

# --- 主优化函数 ---
# 接收大板列表、订单列表、余料列表等作为输入，返回切割方案
def optimize_cutting(plates, orders, others, optim=0, n_plate=None, saw_blade=4):
    logger.info("开始排版优化过程")
    
    # --- 1. 数据准备 ---
    # 将输入的字典列表转换为更易于处理的元组列表
    big_plate_list = []
    for p in plates:
        # 根据 'quantity' 字段，将每种大板重复添加到列表中
        for _ in range(p.get('quantity', 0)):
            big_plate_list.append((p['length'], p['width']))
    if not big_plate_list: # 如果没有可用的大板，直接返回空
        logger.warning("输入中没有有效的大板信息")
        return []

    # 转换订单（需要切割的小板）
    small_plates_list = [(p['length'], p['width'], p.get('id', ''), p['quantity']) 
                        for p in orders if p.get('quantity', 0) > 0]
    
    # 转换余料（可以用来填充空隙的库存板）
    stock_plates_list = [(p['length'], p['width'], f"R{p.get('id', '')}")
                        for p in others if p.get('length', 0) > 0 and p.get('width', 0) > 0] if others else []

    logger.info(f"已处理输入: {len(big_plate_list)} 块大板, {len(small_plates_list)} 种订单, {len(stock_plates_list)} 种余料")
    
    # 将大板数据实例化为 Plate 对象
    big_plate_objects = [Plate(length, width, saw_blade) for length, width in big_plate_list]
    
    # --- 2. 排序与预处理 ---
    # 根据切割效率对小板进行排序和预旋转，这是核心启发式策略之一
    small_plates2 = []
    length0, width0 = big_plate_list[0] # 以第一块大板的尺寸作为参考
    for x1, x2, x3, nums in small_plates_list:
        # 启发式旋转：如果旋转后（x2作为长），预估能在大板上放下更多块，则进行旋转
        if (x1 < x2 and (length0//x2)*(width0//x1) >= (length0//x1)*(width0//x2)) or (abs(x1- x2)/max(x1,x2) < 0.56 and (length0//x2)*(width0//x1) > (length0//x1)*(width0//x2)):
            small_plates2.append((x2, x1, x3, nums)) # 存储旋转后的尺寸
            logger.debug(f"旋转订单板材: {x1}x{x2} -> {x2}x{x1}")
        else:
            small_plates2.append((x1, x2, x3, nums)) # 存储原始尺寸
    
    # 使用 Counter 统计每种宽度的出现次数
    width_counts = Counter(plate[1] for plate in small_plates2)
    # 忽略非常窄的板材的宽度计数，这可能是一个特定的业务需求
    for i in width_counts.keys():
        if i < 400:
            width_counts[i] = 0
    # 核心排序逻辑：
    # 1. 按宽度出现次数降序排（优先处理具有相同宽度的小板，利于整行切割）
    # 2. 按宽度本身降序排（先处理宽的）
    # 3. 按长度降序排（先处理长的）
    small_plates2.sort(key=lambda x: (width_counts[x[1]], x[1], x[0]), reverse=True)
    
    # 将带数量的订单列表展开成单个小板对象的列表
    small_plate_objects = [(x1, x2, x3) for x1, x2, x3, nums in small_plates2 for _ in range(nums)]
    logger.info(f"总计处理 {len(small_plate_objects)} 个待切割的小板")

    # --- 3. 主切割循环 ---
    used_plates = [] # 存储实际使用过的大板对象
    remaining_pieces = small_plate_objects.copy() # 复制一份待切割列表，因为我们会从中删除元素

    # 遍历每一块大板
    for i, big_plate in enumerate(big_plate_objects):
        logger.info(f"正在处理第 {i+1}/{len(big_plate_objects)} 块大板")
        if not remaining_pieces: # 如果所有小板都已切割完毕
            logger.info("所有小板已切割完毕")
            break
        
        pieces_to_place_this_round = remaining_pieces.copy() # 当前轮次要尝试放置的板
        placed_indices = [] # 记录在这一轮中成功放置的板的索引

        # 遍历所有待切割的小板
        for j, small_plate in enumerate(pieces_to_place_this_round):
            logger.debug(f"尝试切割第 {j+1}/{len(pieces_to_place_this_round)} 号小板")
            placed = False # 标记当前小板是否被成功放置
            small_plate1 = (small_plate[1], small_plate[0], small_plate[2]) # 准备一个旋转后的版本

            # 策略：优先在缝隙中放置，以提高利用率
            if big_plate.available_gaps:
                if big_plate.add_cut_in_gap(small_plate): # 尝试用原始方向放入缝隙
                    placed = True
                elif big_plate.add_cut_in_gap(small_plate1): # 尝试用旋转方向放入缝隙
                    placed = True
            
            # 如果缝隙中放不下，则在主区域放置
            if not placed:
                if big_plate.add_cut(small_plate): # 调用主切割方法
                    placed = True
            
            # 如果成功放置了
            if placed:
                placed_indices.append(j) # 记录其索引
                logger.debug(f"小板 {small_plate} 已放置.")

        # 如果当前大板上有任何切割操作
        if placed_indices:
            used_plates.append(big_plate) # 将此大板标记为已使用
            # 从 remaining_pieces 列表中移除已放置的小板
            # 从后往前删除，避免因索引变化导致错误
            for index in sorted(placed_indices, reverse=True):
                del remaining_pieces[index]
    
    # 循环结束后，big_plate_objects 更新为实际使用了的板
    big_plate_objects = used_plates

    # --- 4. 余料填充 ---
    # 如果有库存余料，则尝试用它们填充剩余的空隙
    if stock_plates_list:
        logger.info("开始处理库存余料")
        # 预处理余料，统一为 (长, 宽, ID) 格式，长 > 宽
        stock_plate_objects = [(max(x1, x2), min(x1, x2), x3) for x1, x2, x3 in stock_plates_list]
        for big_plate in big_plate_objects:
            if not optim: # 如果优化标志为0，使用普通填充
                _process_stock_plates(big_plate, stock_plate_objects)
            else: # 否则，使用带顺序优化的填充
                _optimize_stock_placement(big_plate, stock_plate_objects)

    # --- 5. 计算最终利用率并格式化输出 ---
    for big_plate in big_plate_objects:
        big_plate.utilization() # 计算每块板的最终利用率

    # 转换为指定的 cutted 格式输出
    cutted = []
    for big_plate in big_plate_objects:
        if big_plate.cuts: # 只输出有切割操作的板
            plate_cuts = []
            for cut in big_plate.cuts:
                # 判断是否是余料（通过ID是否以'R'开头）
                is_stock = str(cut[0][2]).startswith('R')
                plate_cuts.append([
                    cut[1], # start_x
                    cut[2], # start_y  
                    cut[0][0], # length
                    cut[0][1], # width
                    1 if is_stock else 0, # is_stock 标志
                    cut[0][2][1:] if is_stock else cut[0][2] # ID, 如果是余料则去掉'R'
                ])
            cutted.append({
                'rate': big_plate.used_perc / 100, # 利用率
                'plate': [big_plate.length, big_plate.width], # 大板尺寸
                'cutted': plate_cuts # 切割详情列表
            })
            
    logger.info(f"优化完成. 生成了 {len(cutted)} 个切割方案")
    return cutted

# --- 辅助函数：普通余料填充 ---
def _process_stock_plates(big_plate, stock_plate_objects):
    logger.debug("正在进行无优化的余料填充")
    for stock_plate in stock_plate_objects:
        try:
            # 粗略估计还能放下多少块该余料
            n = int((100 - big_plate.utilization())/100*(big_plate.length * big_plate.width)/(stock_plate[0]*stock_plate[1]))
        except ZeroDivisionError:
            n = 0
        logger.debug(f"尝试放置 {n} 块余料 {stock_plate[0]}x{stock_plate[1]}")
        # 循环尝试放置n次
        for _ in range(n):
            # 填充策略：优先放缝隙，再尝试旋转放缝隙，再尝试主区域，最后尝试旋转放主区域
            if not big_plate.add_cut_in_gap(stock_plate):
                stock_plate1 = (stock_plate[1], stock_plate[0], stock_plate[2])
                if not big_plate.add_cut_in_gap(stock_plate1):
                    if not big_plate.add_cut(stock_plate):
                        big_plate.add_cut(stock_plate1)

# --- 辅助函数：优化余料填充 ---
def _optimize_stock_placement(big_plate, stock_plate_objects):
    # 此函数逻辑较为复杂，它尝试寻找一个更优的余料填充顺序
    logger.debug("正在优化余料放置顺序")
    base_util = big_plate.utilization() # 记录填充前的利用率
    n_optim = -1 # 用于记录最佳的起始余料索引，-1表示还没找到更优方案
    
    # 遍历前10个或所有余料，尝试将每一个作为第一个填充的余料，看哪个效果最好
    for i in range(min(10, len(stock_plate_objects))):
        # 创建一个大板的深拷贝用于模拟，避免影响原始大板对象
        big_plate_sim = copy.deepcopy(big_plate)
        
        # 创建一个重新排序的余料列表，将第i个余料放到最前面
        stock_order_sim = stock_plate_objects.copy()
        first_stock = stock_order_sim.pop(i)
        stock_order_sim.insert(0, first_stock)
        
        # 在模拟板上执行普通填充
        _process_stock_plates(big_plate_sim, stock_order_sim)
        
        # 获取模拟后的利用率
        sim_util = big_plate_sim.utilization()
        # 如果模拟结果比当前最优结果好
        if sim_util > base_util:
            base_util = sim_util # 更新最优利用率
            n_optim = i # 记录下这个起始余料的索引
            logger.debug(f"找到更优利用率: {base_util:.2f}% (当余料 {i} 第一个被放置时)")

    # 如果找到了更优的顺序 (n_optim != -1)
    if n_optim != -1:
        # 按照找到的最优顺序创建最终的余料列表
        final_stock_order = stock_plate_objects.copy()
        first_stock = final_stock_order.pop(n_optim)
        final_stock_order.insert(0, first_stock)
        # 在真实的大板上执行最优顺序的填充
        _process_stock_plates(big_plate, final_stock_order)
        logger.info(f"余料优化后最终利用率: {big_plate.utilization():.2f}%")
    else:
        # 如果没有找到更优解，则按原顺序填充
        _process_stock_plates(big_plate, stock_plate_objects)