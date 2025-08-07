import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import rectpack
import itertools
import collections
import operator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('plate_cutting')


@dataclass
class CuttingConfig:
    """切割配置参数"""
    blade_thickness: int = 4  # 锯片厚度


@dataclass
class SmallPlate:
    """小板材信息"""
    length: int
    width: int
    plate_id: str = ""
    quantity: int = 1
    
    @property
    def area(self) -> int:
        return self.length * self.width


@dataclass
class Cut:
    """切割记录"""
    plate: SmallPlate
    x1: int
    y1: int
    x2: int
    y2: int
    is_stock: bool = False


class Rectangle:
    """矩形类"""
    
    def __init__(self, x: int, y: int, width: int, height: int, rid=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rid = rid
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        return self.y
    
    @property
    def top(self) -> int:
        return self.y + self.height
    
    def area(self) -> int:
        """返回矩形面积"""
        return self.width * self.height
    
    def intersects(self, other: 'Rectangle') -> bool:
        """检查两个矩形是否相交"""
        return not (self.right <= other.left or 
                   other.right <= self.left or
                   self.top <= other.bottom or 
                   other.top <= self.bottom)
    
    def contains(self, other: 'Rectangle') -> bool:
        """检查是否完全包含另一个矩形"""
        return (self.left <= other.left and 
                self.bottom <= other.bottom and
                self.right >= other.right and 
                self.top >= other.top)
    
    def join(self, other: 'Rectangle') -> bool:
        """尝试合并两个相邻的矩形"""
        # 检查是否可以水平合并
        if (self.y == other.y and self.height == other.height):
            if self.right == other.left:
                self.width += other.width
                return True
            elif other.right == self.left:
                self.x = other.x
                self.width += other.width
                return True
        
        # 检查是否可以垂直合并
        if (self.x == other.x and self.width == other.width):
            if self.top == other.bottom:
                self.height += other.height
                return True
            elif other.top == self.bottom:
                self.y = other.y
                self.height += other.height
                return True
        
        return False
    
    def __repr__(self) -> str:
        return f"Rectangle({self.x}, {self.y}, {self.width}, {self.height})"


class CustomStockPacker:
    """自定义库存板装箱器 - 基于标准 MaxRects BAF 算法实现"""
    
    def __init__(self, width: int, height: int, config: CuttingConfig):
        self.width = width
        self.height = height
        self.config = config
        self.rot = True  # 允许旋转
        self.rectangles = []  # 已放置的矩形
        self.cuts: List[Cut] = []
        self.reset()
    
    def reset(self):
        """重置装箱器"""
        self._max_rects = [Rectangle(0, 0, self.width, self.height)]
        self.rectangles = []
        self.cuts = []
    
    def _rect_fitness(self, max_rect: Rectangle, width: int, height: int) -> Optional[float]:
        """
        计算矩形适配度 - 使用 Best Area Fit 策略
        
        Args:
            max_rect: 目标最大矩形
            width: 待放置矩形宽度
            height: 待放置矩形高度
            
        Returns:
            适配度值（越小越好），如果无法放置则返回None
        """
        if width > max_rect.width or height > max_rect.height:
            return None
        
        # Best Area Fit: 选择面积最小的能容纳的最大矩形
        return (max_rect.width * max_rect.height) - (width * height)
    
    def _select_position(self, w: int, h: int) -> Tuple[Optional[Rectangle], Optional[Rectangle]]:
        """
        选择最佳放置位置
        
        Args:
            w: 矩形宽度
            h: 矩形高度
            
        Returns:
            (放置矩形, 选中的最大矩形)，如果无法放置则返回(None, None)
        """
        if not self._max_rects:
            return None, None
        
        first_item = operator.itemgetter(0)
        
        # 正常方向的矩形
        fitn = ((self._rect_fitness(m, w, h), w, h, m) for m in self._max_rects 
                if self._rect_fitness(m, w, h) is not None)
        
        # 旋转后的矩形
        fitr = ((self._rect_fitness(m, h, w), h, w, m) for m in self._max_rects 
                if self._rect_fitness(m, h, w) is not None)
        
        if not self.rot:
            fitr = []
        
        fit = itertools.chain(fitn, fitr)
        
        try:
            _, w, h, m = min(fit, key=first_item)
        except ValueError:
            return None, None
        
        return Rectangle(m.x, m.y, w, h), m
    
    def _generate_splits(self, m: Rectangle, r: Rectangle) -> List[Rectangle]:
        """
        当一个矩形被放置在最大矩形内时，可能产生最多4个新的最大矩形
        
        Args:
            m: 原最大矩形
            r: 被放置的矩形
            
        Returns:
            新产生的最大矩形列表
        """
        new_rects = []
        
        # 左侧剩余
        if r.left > m.left:
            new_rects.append(Rectangle(m.left, m.bottom, r.left - m.left, m.height))
        
        # 右侧剩余
        if r.right < m.right:
            new_rects.append(Rectangle(r.right, m.bottom, m.right - r.right, m.height))
        
        # 上方剩余
        if r.top < m.top:
            new_rects.append(Rectangle(m.left, r.top, m.width, m.top - r.top))
        
        # 下方剩余
        if r.bottom > m.bottom:
            new_rects.append(Rectangle(m.left, m.bottom, m.width, r.bottom - m.bottom))
        
        return new_rects
    
    def _split(self, rect: Rectangle):
        """
        分割所有与给定矩形相交的最大矩形
        
        Args:
            rect: 新放置的矩形
        """
        max_rects = collections.deque()
        
        for r in self._max_rects:
            if r.intersects(rect):
                max_rects.extend(self._generate_splits(r, rect))
            else:
                max_rects.append(r)
        
        self._max_rects = list(max_rects)
    
    def _remove_duplicates(self):
        """移除被其他矩形包含的最大矩形"""
        contained = set()
        for m1, m2 in itertools.combinations(self._max_rects, 2):
            if m1.contains(m2):
                contained.add(m2)
            elif m2.contains(m1):
                contained.add(m1)
        
        self._max_rects = [m for m in self._max_rects if m not in contained]
    
    def add_rect(self, plate: SmallPlate) -> bool:
        """
        添加矩形到装箱器中
        
        Args:
            plate: 要添加的板材
            
        Returns:
            是否成功添加
        """
        # 考虑锯片厚度
        needed_width = plate.length + self.config.blade_thickness
        needed_height = plate.width + self.config.blade_thickness
        
        # 寻找最佳位置
        rect, _ = self._select_position(needed_width, needed_height)
        if not rect:
            return False
        
        # 分割相交的最大矩形
        self._split(rect)
        
        # 移除重复的最大矩形
        self._remove_duplicates()
        
        # 创建实际放置的矩形（不包含锯片厚度）
        actual_width = plate.length
        actual_height = plate.width
        
        # 检查是否旋转了
        rotated = (rect.width - self.config.blade_thickness != plate.length)
        if rotated:
            actual_width, actual_height = actual_height, actual_width
        
        # 记录切割
        cut = Cut(
            plate=plate,
            x1=rect.x,
            y1=rect.y,
            x2=rect.x + actual_width,
            y2=rect.y + actual_height,
            is_stock=True
        )
        self.cuts.append(cut)
        
        # 存储矩形信息
        rect.rid = plate.plate_id
        self.rectangles.append(rect)
        
        return True
    
    def fitness(self, width: int, height: int) -> Optional[float]:
        """
        计算给定尺寸矩形的适配度
        
        Args:
            width: 矩形宽度
            height: 矩形高度
            
        Returns:
            适配度值，如果无法放置则返回None
        """
        rect, max_rect = self._select_position(width, height)
        if rect is None:
            return None
        
        return self._rect_fitness(max_rect, rect.width, rect.height)
    
    def get_utilization(self) -> float:
        """计算当前利用率"""
        if not self.rectangles:
            return 0.0
        
        total_area = self.width * self.height
        used_area = sum(r.width * r.height for r in self.rectangles)
        return used_area / total_area if total_area > 0 else 0.0


class CustomGuillotinePacker:
    """自定义Guillotine装箱器 - 实现GuillotineBafMinas算法"""
    
    def __init__(self, width: int, height: int, config: CuttingConfig):
        self.width = width
        self.height = height
        self.config = config
        self.rot = True  # 允许旋转
        self.merge = True  # 允许合并空闲区域
        self.rectangles = []  # 已放置的矩形
        self.cuts: List[Cut] = []
        self.reset()
    
    def reset(self):
        """重置装箱器"""
        self._sections = []
        self._add_section(Rectangle(0, 0, self.width, self.height))
        self.rectangles = []
        self.cuts = []
    
    def _add_section(self, section: Rectangle):
        """添加新的空闲区域，并尝试与现有区域合并"""
        section.rid = 0
        plen = 0
        
        # 尝试合并区域
        while self.merge and self._sections and plen != len(self._sections):
            plen = len(self._sections)
            self._sections = [s for s in self._sections if not section.join(s)]
        
        self._sections.append(section)
    
    def _section_fitness_baf(self, section: Rectangle, width: int, height: int) -> Optional[float]:
        """Best Area Fit: 选择面积浪费最小的区域"""
        if width > section.width or height > section.height:
            return None
        return section.area() - width * height
    
    def _split_minas(self, section: Rectangle, width: int, height: int):
        """
        Min Area Split (MINAS): 最小化较小剩余区域的面积
        这有助于保持剩余空间更加集中
        """
        # 计算两种分割方式产生的剩余面积
        horizontal_area = width * (section.height - height)
        vertical_area = height * (section.width - width)
        
        if horizontal_area >= vertical_area:
            # 水平分割
            self._split_horizontal(section, width, height)
        else:
            # 垂直分割
            self._split_vertical(section, width, height)
    
    def _split_horizontal(self, section: Rectangle, width: int, height: int):
        """水平分割：矩形放置在左下角，水平线分割"""
        # 上方剩余区域
        if height < section.height:
            self._add_section(Rectangle(
                section.x, section.y + height,
                section.width, section.height - height
            ))
        
        # 右侧剩余区域
        if width < section.width:
            self._add_section(Rectangle(
                section.x + width, section.y,
                section.width - width, height
            ))
    
    def _split_vertical(self, section: Rectangle, width: int, height: int):
        """垂直分割：矩形放置在左下角，垂直线分割"""
        # 上方剩余区域
        if height < section.height:
            self._add_section(Rectangle(
                section.x, section.y + height,
                width, section.height - height
            ))
        
        # 右侧剩余区域
        if width < section.width:
            self._add_section(Rectangle(
                section.x + width, section.y,
                section.width - width, section.height
            ))
    
    def _select_best_section(self, w: int, h: int) -> Tuple[Optional[Rectangle], bool]:
        """选择最佳区域放置矩形"""
        best_fitness = None
        best_section = None
        best_rotated = False
        
        # 尝试正常方向
        for section in self._sections:
            fitness = self._section_fitness_baf(section, w, h)
            if fitness is not None:
                if best_fitness is None or fitness < best_fitness:
                    best_fitness = fitness
                    best_section = section
                    best_rotated = False
        
        # 尝试旋转方向
        if self.rot:
            for section in self._sections:
                fitness = self._section_fitness_baf(section, h, w)
                if fitness is not None:
                    if best_fitness is None or fitness < best_fitness:
                        best_fitness = fitness
                        best_section = section
                        best_rotated = True
        
        return best_section, best_rotated
    
    def add_rect(self, plate: SmallPlate) -> bool:
        """
        添加矩形到装箱器中
        
        Args:
            plate: 要添加的板材
            
        Returns:
            是否成功添加
        """
        # 考虑锯片厚度
        needed_width = plate.length + self.config.blade_thickness
        needed_height = plate.width + self.config.blade_thickness
        
        # 选择最佳区域
        section, rotated = self._select_best_section(needed_width, needed_height)
        if not section:
            return False
        
        if rotated:
            needed_width, needed_height = needed_height, needed_width
        
        # 移除选中的区域并分割
        self._sections.remove(section)
        self._split_minas(section, needed_width, needed_height)
        
        # 创建实际放置的矩形（不包含锯片厚度）
        actual_width = plate.length
        actual_height = plate.width
        
        if rotated:
            actual_width, actual_height = actual_height, actual_width
        
        # 记录切割
        cut = Cut(
            plate=plate,
            x1=section.x,
            y1=section.y,
            x2=section.x + actual_width,
            y2=section.y + actual_height,
            is_stock=True
        )
        self.cuts.append(cut)
        
        # 存储矩形信息
        rect = Rectangle(section.x, section.y, needed_width, needed_height, plate.plate_id)
        self.rectangles.append(rect)
        
        return True
    
    def fitness(self, width: int, height: int) -> Optional[float]:
        """计算给定尺寸矩形的适配度"""
        section, rotated = self._select_best_section(width, height)
        if not section:
            return None
        
        if rotated:
            return self._section_fitness_baf(section, height, width)
        else:
            return self._section_fitness_baf(section, width, height)
    
    def get_utilization(self) -> float:
        """计算当前利用率"""
        if not self.rectangles:
            return 0.0
        
        total_area = self.width * self.height
        used_area = sum(r.width * r.height for r in self.rectangles)
        return used_area / total_area if total_area > 0 else 0.0


class PlateOptimizer:
    """板材优化器 - 用于处理订单，保持原有逻辑"""
    
    def __init__(self, config: CuttingConfig, algorithm: rectpack = rectpack.GuillotineBssfMaxas):
        self.config = config
        self.algorithm = algorithm
        
    def create_packer(self, width: int, height: int) -> rectpack.packer:
        """创建 rectpack 装箱器"""
        
        packer = rectpack.newPacker(
            mode=rectpack.PackingMode.Offline,
            bin_algo = rectpack.PackingBin.Global,
            pack_algo=self.algorithm,
            sort_algo = rectpack.SORT_PERI,
            rotation=True  # 允许旋转
        )
        
        # 添加容器（大板）
        packer.add_bin(width, height)
        
        return packer
    
    def pack_orders(self, big_plate: SmallPlate, orders: List[SmallPlate]) -> Tuple[List[Cut], List[SmallPlate]]:
        """使用 rectpack 装箱订单板材"""
        packer = self.create_packer(big_plate.length, big_plate.width)
        
        # 添加所有订单矩形
        for i, order in enumerate(orders):
            # 考虑锯片厚度
            packer.add_rect(
                order.length + self.config.blade_thickness,
                order.width + self.config.blade_thickness,
                rid=i  # 使用索引作为ID
            )
        
        # 执行装箱
        packer.pack()
        
        # 提取结果
        cuts = []
        packed_indices = set()
        
        for bin_data in packer:
            for rect in bin_data:
                # 修复：rectpack返回Rectangle对象，需要访问其属性
                try:
                    x = rect.x
                    y = rect.y
                    w = rect.width
                    h = rect.height
                    
                    # 尝试多种方式获取rid
                    rid = None
                    for attr_name in ['rid', 'id', 'rect_id', 'tag']:
                        if hasattr(rect, attr_name):
                            rid = getattr(rect, attr_name)
                            break
                    
                    # 如果没有rid，通过匹配找到对应的订单
                    if rid is None:
                        rid = self._find_matching_order_index(orders, w - self.config.blade_thickness, h - self.config.blade_thickness, packed_indices)
                    
                except AttributeError as e:
                    logger.warning(f"Error accessing Rectangle attributes: {e}")
                    continue
                
                if rid is None or rid in packed_indices:
                    continue
                    
                order = orders[rid]
                
                # 判断是否旋转了
                rotated = (w - self.config.blade_thickness != order.length)
                actual_length = order.width if rotated else order.length
                actual_width = order.length if rotated else order.width
                
                cut = Cut(
                    plate=order,
                    x1=x,
                    y1=y,
                    x2=x + actual_length,
                    y2=y + actual_width,
                    is_stock=False
                )
                cuts.append(cut)
                packed_indices.add(rid)
        
        # 返回切割结果和剩余订单
        remaining = [orders[i] for i in range(len(orders)) if i not in packed_indices]
        return cuts, remaining
    
    def _find_matching_order_index(self, orders: List[SmallPlate], w: int, h: int, used_indices: set) -> Optional[int]:
        """查找匹配的订单索引（当rectpack没有返回rid时使用）"""
        for i, order in enumerate(orders):
            if i in used_indices:
                continue
            # 检查是否匹配（考虑旋转）
            if (w == order.length and h == order.width) or (w == order.width and h == order.length):
                return i
        return None


class StockOptimizer:
    """库存板材优化器 - 支持MaxRects BAF和Guillotine两种算法"""
    
    def __init__(self, config: CuttingConfig, algorithm: str = "maxrects"):
        """
        Args:
            config: 切割配置
            algorithm: 算法选择 ("maxrects" 或 "guillotine")
        """
        self.config = config
        self.algorithm = algorithm.lower()
    
    def fill_with_stock(self, width: int, height: int, existing_cuts: List[Cut], 
                       stock_plates: List[SmallPlate], optimize: bool = False) -> List[Cut]:
        """用库存板材填充剩余空间"""
        if not stock_plates:
            return []
        
        # 根据算法选择创建不同的装箱器
        if self.algorithm == "guillotine":
            packer = CustomGuillotinePacker(width, height, self.config)
            logger.debug("使用Guillotine BAF MINAS算法填充库存")
        else:  # maxrects
            packer = CustomStockPacker(width, height, self.config)
            logger.debug("使用MaxRects BAF算法填充库存")
        
        # 首先将已占用区域添加到装箱器中（模拟已放置的矩形）
        if self.algorithm == "maxrects":
            for cut in existing_cuts:
                occupied_rect = Rectangle(
                    cut.x1, cut.y1, 
                    cut.x2 - cut.x1 + self.config.blade_thickness, 
                    cut.y2 - cut.y1 + self.config.blade_thickness
                )
                # 分割被占用区域
                packer._split(occupied_rect)
                packer._remove_duplicates()
        else:  # guillotine
            # 对于Guillotine算法，需要从初始区域中移除已占用的部分
            for cut in existing_cuts:
                # 创建一个占位矩形（包含锯片厚度）
                occupied = Rectangle(
                    cut.x1, cut.y1,
                    cut.x2 - cut.x1 + self.config.blade_thickness,
                    cut.y2 - cut.y1 + self.config.blade_thickness
                )
                
                # 处理每个section，分割出未被占用的部分
                new_sections = []
                for section in packer._sections:
                    if not occupied.intersects(section):
                        # 不相交，保留整个section
                        new_sections.append(section)
                    else:
                        # 相交，需要分割section，保留未被占用的部分
                        # 可能产生最多4个新的section
                        
                        # 左侧部分（如果存在）
                        if section.x < occupied.x and occupied.x < section.x + section.width:
                            left_section = Rectangle(
                                section.x, section.y,
                                occupied.x - section.x, section.height
                            )
                            new_sections.append(left_section)
                        
                        # 右侧部分（如果存在）
                        if occupied.x + occupied.width < section.x + section.width:
                            right_x = max(occupied.x + occupied.width, section.x)
                            right_section = Rectangle(
                                right_x, section.y,
                                section.x + section.width - right_x, section.height
                            )
                            new_sections.append(right_section)
                        
                        # 下方部分（如果存在）
                        if section.y < occupied.y and occupied.y < section.y + section.height:
                            bottom_section = Rectangle(
                                section.x, section.y,
                                section.width, occupied.y - section.y
                            )
                            new_sections.append(bottom_section)
                        
                        # 上方部分（如果存在）
                        if occupied.y + occupied.height < section.y + section.height:
                            top_y = max(occupied.y + occupied.height, section.y)
                            top_section = Rectangle(
                                section.x, top_y,
                                section.width, section.y + section.height - top_y
                            )
                            new_sections.append(top_section)
                
                packer._sections = new_sections
                
            # 合并相邻的sections以优化空间
            if packer.merge and packer._sections:
                merged = True
                while merged:
                    merged = False
                    temp_sections = []
                    used_indices = set()
                    
                    for i, s1 in enumerate(packer._sections):
                        if i in used_indices:
                            continue
                        merged_section = Rectangle(s1.x, s1.y, s1.width, s1.height)
                        for j, s2 in enumerate(packer._sections):
                            if i != j and j not in used_indices:
                                if merged_section.join(s2):
                                    used_indices.add(j)
                                    merged = True
                        temp_sections.append(merged_section)
                        used_indices.add(i)
                    
                    packer._sections = temp_sections
        
        # 根据优化标志处理库存板
        if optimize:
            # 优化模式：按面积排序，大的优先
            sorted_stock = sorted(stock_plates, key=lambda p: p.area, reverse=True)
        else:
            # 非优化模式：按原始顺序
            sorted_stock = stock_plates
        
        # 尝试多轮填充以最大化利用率
        max_rounds = 10  # 最多尝试10轮
        round_count = 0
        
        while sorted_stock and round_count < max_rounds:
            round_count += 1
            placed_this_round = 0
            
            # 为每种库存板材尝试放置
            for stock in sorted_stock:
                # 检查适配度
                if packer.fitness(stock.length + self.config.blade_thickness, 
                                stock.width + self.config.blade_thickness) is not None:
                    if packer.add_rect(stock):
                        placed_this_round += 1
                        # 继续尝试放置同样的板材
                        while packer.add_rect(stock):
                            placed_this_round += 1
            
            # 如果这轮没有放置任何板材，结束
            if placed_this_round == 0:
                break
            
            logger.debug(f"第{round_count}轮填充了{placed_this_round}块库存板材")
        
        # 更新库存切割的 plate_id
        for cut in packer.cuts:
            if not cut.plate.plate_id:
                cut.plate.plate_id = f"STOCK_{len(packer.cuts)}"
        
        # 记录最终利用率
        utilization = packer.get_utilization()
        logger.debug(f"库存填充完成，总利用率: {utilization:.2%}，放置了{len(packer.cuts)}块库存板材")
        
        return packer.cuts


class DataConverter:
    """数据转换器"""
    
    def convert_plates(self, plates: List[Dict[str, Any]]) -> List[SmallPlate]:
        """转换大板数据"""
        result = []
        for plate_data in plates:
            quantity = plate_data.get('quantity', 0)
            if quantity > 0:
                for _ in range(quantity):
                    result.append(SmallPlate(
                        length=plate_data['length'],
                        width=plate_data['width']
                    ))
        return result
    
    def convert_orders(self, orders: List[Dict[str, Any]]) -> List[SmallPlate]:
        """转换订单数据"""
        result = []
        for order in orders:
            quantity = order.get('quantity', 0)
            if quantity > 0:
                for _ in range(quantity):
                    result.append(SmallPlate(
                        length=order['length'],
                        width=order['width'],
                        plate_id=str(order.get('id', ''))
                    ))
        return result
    
    def convert_stock(self, stock: List[Dict[str, Any]]) -> List[SmallPlate]:
        """转换库存数据"""
        result = []
        for item in stock:
            if item.get('length', 0) > 0 and item.get('width', 0) > 0:
                result.append(SmallPlate(
                    length=item['length'],
                    width=item['width'],
                    plate_id=str(item.get('id', ''))
                ))
        return result
    
    def convert_cuts_to_output(self, big_plate: SmallPlate, cuts: List[Cut]) -> Dict[str, Any]:
        """转换切割结果为输出格式"""
        cuts_data = []
        for cut in cuts:
            plate_id = cut.plate.plate_id
            cuts_data.append([
                cut.x1,  # start_x
                cut.y1,  # start_y
                cut.x2 - cut.x1,  # length
                cut.y2 - cut.y1,  # width
                1 if cut.is_stock else 0,  # is_stock
                plate_id  # id
            ])
        
        # 计算利用率
        used_area = sum((cut[2] * cut[3]) for cut in cuts_data)
        total_area = big_plate.length * big_plate.width
        utilization_rate = used_area / total_area if total_area > 0 else 0
        
        return {
            'rate': utilization_rate,
            'plate': [big_plate.length, big_plate.width],
            'cutted': cuts_data
        }


def calculate_cutting_metrics(results: List[Dict[str, Any]], remaining_orders: int) -> Dict[str, Any]:
    """
    计算切割方案的详细指标
    
    Returns:
        包含多个评价指标的字典
    """
    if not results:
        return {
            'used_plates': 0,
            'overall_rate': 0,
            'avg_rate': 0,
            'min_rate': 0,
            'rate_variance': 0,
            'total_cuts': 0,
            'avg_cuts_per_plate': 0,
            'order_completion': 0,
            'remaining_orders': remaining_orders
        }
    
    # 基础统计
    used_plates = len(results)
    rates = [r['rate'] for r in results]
    overall_rate = sum(rates) / used_plates if used_plates > 0 else 0
    
    # 利用率分布统计
    min_rate = min(rates) if rates else 0
    max_rate = max(rates) if rates else 0
    
    # 利用率方差（衡量利用率的均匀程度）
    avg_rate = overall_rate
    rate_variance = sum((r - avg_rate) ** 2 for r in rates) / used_plates if used_plates > 0 else 0
    
    # 切割复杂度统计
    total_cuts = sum(len(r['cutted']) for r in results)
    avg_cuts_per_plate = total_cuts / used_plates if used_plates > 0 else 0
    
    # 最大单板切割数（切割复杂度）
    max_cuts_single_plate = max(len(r['cutted']) for r in results) if results else 0
    
    # 订单完成度
    total_order_cuts = sum(
        1 for r in results 
        for cut in r['cutted'] 
        if cut[4] == 0  # is_stock == 0 表示订单板材
    )
    
    return {
        'used_plates': used_plates,
        'overall_rate': overall_rate,
        'avg_rate': avg_rate,
        'min_rate': min_rate,
        'max_rate': max_rate,
        'rate_variance': rate_variance,
        'total_cuts': total_cuts,
        'avg_cuts_per_plate': avg_cuts_per_plate,
        'max_cuts_single_plate': max_cuts_single_plate,
        'order_completion': total_order_cuts,
        'remaining_orders': remaining_orders
    }


def optimize_cutting(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]], 
                    others: List[Dict[str, Any]] = None, optim: int = 0, 
                    saw_blade: int = 4, algorithm: str = "auto", 
                    stock_algorithm: str = "guillotine") -> List[Dict[str, Any]]:
    """
    主优化函数
    
    Args:
        plates: 大板信息列表
        orders: 订单信息列表  
        others: 库存余料列表
        optim: 是否启用库存优化（仅影响库存板）
        saw_blade: 锯片厚度
        algorithm: 订单算法选择
            - "MaxRectsBaf": MaxRects Best Area Fit
            - "GuillotineBafMinas": Guillotine Best Area Fit with Minimal Area Split
            - "SkylineMwfWm": Skyline Minimal Waste Fit with Merge
            - "auto": 自动优化模式（默认）- 尝试三种算法，选择最优
        stock_algorithm: 库存填充算法选择
            - "maxrects": 使用MaxRects BAF算法（默认）
            - "guillotine": 使用Guillotine BAF MINAS算法
    
    Returns:
        切割方案列表
    """
    
    # 定义可用算法映射
    ALGORITHMS = {
        "MaxRectsBaf": rectpack.MaxRectsBaf,
        "GuillotineBafMinas": rectpack.GuillotineBafMinas,
        "SkylineMwfWm": rectpack.SkylineMwfWm,
    }
    
    # 配置
    config = CuttingConfig(blade_thickness=saw_blade)
    
    # 数据转换
    converter = DataConverter()
    big_plates = converter.convert_plates(plates)
    small_plates = converter.convert_orders(orders)
    stock_plates = converter.convert_stock(others) if others else []
    
    if not big_plates:
        return []
    
    # 选择订单处理算法
    if algorithm == "auto":
        # 自动优化模式：尝试所有算法，选择最优
        logger.info(f"使用自动优化模式，测试多种算法...")
        best_algorithm = None
        best_results = None
        best_remaining = small_plates
        
        for algo_name, algo_class in ALGORITHMS.items():
            logger.info(f"测试算法: {algo_name}")
            temp_results = []
            temp_remaining = small_plates.copy()
            
            plate_optimizer = PlateOptimizer(config, algo_class)
            stock_optimizer = StockOptimizer(config, stock_algorithm)
            
            for big_plate in big_plates:
                if not temp_remaining:
                    break
                    
                order_cuts, temp_remaining = plate_optimizer.pack_orders(big_plate, temp_remaining)
                
                if order_cuts:
                    stock_cuts = []
                    if stock_plates:
                        stock_cuts = stock_optimizer.fill_with_stock(
                            big_plate.length, big_plate.width, 
                            order_cuts, stock_plates, 
                            optimize=bool(optim)
                        )
                    
                    all_cuts = order_cuts + stock_cuts
                    result = converter.convert_cuts_to_output(big_plate, all_cuts)
                    temp_results.append(result)
            
            metrics = calculate_cutting_metrics(temp_results, len(temp_remaining))
            logger.info(f"  - 使用板材: {metrics['used_plates']} 块")
            logger.info(f"  - 平均利用率: {metrics['overall_rate']:.2%}")
            logger.info(f"  - 剩余订单: {metrics['remaining_orders']} 个")
            
            if best_results is None or metrics['overall_rate'] > calculate_cutting_metrics(best_results, len(best_remaining))['overall_rate']:
                best_algorithm = algo_class
                best_results = temp_results
                best_remaining = temp_remaining
                logger.info(f"  选择 {algo_name} 作为最优算法")
        
        return best_results
        
    elif algorithm in ALGORITHMS:
        # 使用指定算法
        order_algorithm = ALGORITHMS[algorithm]
    else:
        # 默认算法
        logger.warning(f"未知算法 '{algorithm}'，使用默认算法 MaxRectsBaf")
        order_algorithm = rectpack.MaxRectsBaf
    
    # 创建优化器
    plate_optimizer = PlateOptimizer(config, order_algorithm)
    stock_optimizer = StockOptimizer(config, stock_algorithm)
    
    # 记录使用的算法
    if stock_algorithm == "guillotine":
        logger.info(f"订单处理: {algorithm}, 库存填充: Guillotine BAF MINAS")
    else:
        logger.info(f"订单处理: {algorithm}, 库存填充: MaxRects BAF")
    
    # 主切割循环
    results = []
    remaining_orders = small_plates.copy()
    
    for i, big_plate in enumerate(big_plates):
        if not remaining_orders:
            break
        
        # 使用 rectpack 装箱订单
        order_cuts, remaining_orders = plate_optimizer.pack_orders(big_plate, remaining_orders)
        
        if order_cuts:
            # 库存填充
            stock_cuts = []
            if stock_plates:
                stock_cuts = stock_optimizer.fill_with_stock(
                    big_plate.length, big_plate.width, 
                    order_cuts, stock_plates, 
                    optimize=bool(optim)
                )
            
            # 合并切割结果
            all_cuts = order_cuts + stock_cuts
            
            # 转换为输出格式
            result = converter.convert_cuts_to_output(big_plate, all_cuts)
            results.append(result)
    
    # 计算并显示最终指标
    metrics = calculate_cutting_metrics(results, len(remaining_orders))
    logger.info(f"完成切割:")
    logger.info(f"  - 使用板材: {metrics['used_plates']} 块")
    logger.info(f"  - 平均利用率: {metrics['overall_rate']:.2%}")
    logger.info(f"  - 平均切割数: {metrics['avg_cuts_per_plate']:.1f} 次/板")
    
    return results


# 使用示例
if __name__ == "__main__":
    # 示例数据
    plates = [
        {"length": 2440, "width": 1220, "quantity": 5}
    ]
    
    orders = [
        {"id": "A001", "length": 600, "width": 400, "quantity": 3},
        {"id": "A002", "length": 800, "width": 500, "quantity": 2},
        {"id": "A003", "length": 400, "width": 300, "quantity": 4},
    ]
    
    others = [
        {"id": "R001", "length": 200, "width": 150},
        {"id": "R002", "length": 300, "width": 200},
    ]
    
    print("=== 测试不同的库存填充算法 ===\n")
    
    # 使用MaxRects BAF算法填充库存
    print("1. MaxRects BAF 填充库存:")
    results_maxrects = optimize_cutting(
        plates, orders, others, 
        optim=1, 
        algorithm="MaxRectsBaf",
        stock_algorithm="maxrects"
    )
    print(f"   生成 {len(results_maxrects)} 个切割方案\n")
    
    # 使用Guillotine BAF MINAS算法填充库存
    print("2. Guillotine BAF MINAS 填充库存:")
    results_guillotine = optimize_cutting(
        plates, orders, others,
        optim=1,
        algorithm="MaxRectsBaf", 
        stock_algorithm="guillotine"
    )
    print(f"   生成 {len(results_guillotine)} 个切割方案\n")
    
    # 自动模式
    print("3. 自动优化模式（订单）+ MaxRects BAF（库存）:")
    results_auto = optimize_cutting(
        plates, orders, others,
        optim=1,
        algorithm="auto",
        stock_algorithm="maxrects"
    )
    print(f"   生成 {len(results_auto)} 个切割方案\n")
    
    # 自动模式 + Guillotine
    print("4. 自动优化模式（订单）+ Guillotine BAF MINAS（库存）:")
    results_auto_g = optimize_cutting(
        plates, orders, others,
        optim=1,
        algorithm="auto",
        stock_algorithm="guillotine"
    )
    print(f"   生成 {len(results_auto_g)} 个切割方案")