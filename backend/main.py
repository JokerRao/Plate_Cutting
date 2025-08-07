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


# ============================================================================
# 数据类定义
# ============================================================================

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


# ============================================================================
# 基础几何类
# ============================================================================

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
    
    def __repr__(self) -> str:
        return f"Rectangle({self.x}, {self.y}, {self.width}, {self.height})"


# ============================================================================
# 库存板装箱算法
# ============================================================================

class BaseStockPacker:
    """库存板装箱器基类"""
    
    def __init__(self, width: int, height: int, config: CuttingConfig):
        self.width = width
        self.height = height
        self.config = config
        self.rot = True  # 允许旋转
        self.rectangles = []  # 已放置的矩形
        self.cuts: List[Cut] = []
        self.reset()
    
    def reset(self):
        """重置装箱器 - 由子类实现"""
        raise NotImplementedError
    
    def add_rect(self, plate: SmallPlate) -> bool:
        """添加矩形到装箱器中 - 由子类实现"""
        raise NotImplementedError
    
    def fitness(self, width: int, height: int) -> Optional[float]:
        """计算给定尺寸矩形的适配度 - 由子类实现"""
        raise NotImplementedError
    
    def get_utilization(self) -> float:
        """计算当前利用率"""
        if not self.rectangles:
            return 0.0
        
        total_area = self.width * self.height
        used_area = sum(r.width * r.height for r in self.rectangles)
        return used_area / total_area if total_area > 0 else 0.0


class MaxRectsBafPacker(BaseStockPacker):
    """自定义MaxRects BAF算法实现"""
    
    def __init__(self, width: int, height: int, config: CuttingConfig):
        super().__init__(width, height, config)
    
    def reset(self):
        """重置装箱器"""
        self._max_rects = [Rectangle(0, 0, self.width, self.height)]
        self.rectangles = []
        self.cuts = []
    
    def _rect_fitness_baf(self, max_rect: Rectangle, width: int, height: int) -> Optional[float]:
        """
        计算矩形适配度 - 使用 Best Area Fit (BAF) 策略
        选择面积最小的能容纳的最大矩形，最小化浪费空间
        
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
        # 返回剩余面积，越小说明浪费越少，适配度越好
        return (max_rect.width * max_rect.height) - (width * height)
    
    def _select_position(self, w: int, h: int) -> Tuple[Optional[Rectangle], Optional[Rectangle]]:
        """
        选择最佳放置位置（BAF策略）
        
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
        fitn = ((self._rect_fitness_baf(m, w, h), w, h, m) for m in self._max_rects 
                if self._rect_fitness_baf(m, w, h) is not None)
        
        # 旋转后的矩形
        fitr = ((self._rect_fitness_baf(m, h, w), h, w, m) for m in self._max_rects 
                if self._rect_fitness_baf(m, h, w) is not None)
        
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
        
        return self._rect_fitness_baf(max_rect, rect.width, rect.height)


class GuillotineBafMinasPacker(BaseStockPacker):
    """自定义Guillotine BAF + MINAS算法实现"""
    
    def __init__(self, width: int, height: int, config: CuttingConfig):
        self._merge = True  # 启用区域合并（必须在super().__init__之前设置）
        super().__init__(width, height, config)
    
    def reset(self):
        """重置装箱器"""
        self.rectangles = []
        self.cuts = []
        self._sections = []
        # 确保_merge属性存在
        if not hasattr(self, '_merge'):
            self._merge = True
        self._add_section(Rectangle(0, 0, self.width, self.height))
    
    def _add_section(self, section: Rectangle):
        """添加新的空闲区域，并尝试与现有区域合并"""
        section.rid = 0
        plen = 0
        
        # 尝试合并区域
        while self._merge and self._sections and plen != len(self._sections):
            plen = len(self._sections)
            merged = []
            for s in self._sections:
                if not self._try_join(section, s):
                    merged.append(s)
            self._sections = merged
        
        self._sections.append(section)
    
    def _try_join(self, rect1: Rectangle, rect2: Rectangle) -> bool:
        """尝试合并两个矩形区域"""
        # 水平合并：相同高度，相邻位置
        if (rect1.y == rect2.y and rect1.height == rect2.height):
            if rect1.right == rect2.left:
                rect1.width += rect2.width
                return True
            elif rect2.right == rect1.left:
                rect1.x = rect2.x
                rect1.width += rect2.width
                return True
        
        # 垂直合并：相同宽度，相邻位置
        if (rect1.x == rect2.x and rect1.width == rect2.width):
            if rect1.top == rect2.bottom:
                rect1.height += rect2.height
                return True
            elif rect2.top == rect1.bottom:
                rect1.y = rect2.y
                rect1.height += rect2.height
                return True
        
        return False
    
    def _section_fitness_baf(self, section: Rectangle, width: int, height: int) -> Optional[float]:
        """Best Area Fit 适配度计算"""
        if width > section.width or height > section.height:
            return None
        # 返回剩余面积（越小越好）
        return section.width * section.height - width * height
    
    def _split_minas(self, section: Rectangle, width: int, height: int):
        """
        Min Area Axis Split (MINAS) 分割策略
        选择产生最小剩余面积的分割方式
        """
        # 计算两种分割方式产生的剩余面积
        horizontal_waste = width * (section.height - height)
        vertical_waste = height * (section.width - width)
        
        if horizontal_waste >= vertical_waste:
            self._split_horizontal(section, width, height)
        else:
            self._split_vertical(section, width, height)
    
    def _split_horizontal(self, section: Rectangle, width: int, height: int):
        """水平分割"""
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
        """垂直分割"""
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
        """选择最佳放置区域"""
        best_fitness = None
        best_section = None
        rotated = False
        
        # 尝试正常方向
        for section in self._sections:
            fitness = self._section_fitness_baf(section, w, h)
            if fitness is not None:
                if best_fitness is None or fitness < best_fitness:
                    best_fitness = fitness
                    best_section = section
                    rotated = False
        
        # 尝试旋转方向
        if self.rot:
            for section in self._sections:
                fitness = self._section_fitness_baf(section, h, w)
                if fitness is not None:
                    if best_fitness is None or fitness < best_fitness:
                        best_fitness = fitness
                        best_section = section
                        rotated = True
        
        return best_section, rotated
    
    def add_rect(self, plate: SmallPlate) -> bool:
        """添加矩形到装箱器中"""
        # 考虑锯片厚度
        needed_width = plate.length + self.config.blade_thickness
        needed_height = plate.width + self.config.blade_thickness
        
        # 选择最佳区域
        section, rotated = self._select_best_section(needed_width, needed_height)
        if not section:
            return False
        
        if rotated:
            needed_width, needed_height = needed_height, needed_width
        
        # 移除选中的区域
        self._sections.remove(section)
        
        # 执行MINAS分割
        self._split_minas(section, needed_width, needed_height)
        
        # 创建实际放置的矩形（不包含锯片厚度）
        actual_width = plate.length
        actual_height = plate.width
        
        # 检查是否旋转了
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


# ============================================================================
# 优化器类
# ============================================================================

class PlateOptimizer:
    """板材优化器"""
    
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
                    # 调试：打印Rectangle对象的所有属性
                    if hasattr(rect, '__dict__'):
                        logger.debug(f"Rectangle attributes: {rect.__dict__}")
                    else:
                        logger.debug(f"Rectangle dir: {[attr for attr in dir(rect) if not attr.startswith('_')]}")
                    
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
                    logger.warning(f"Rectangle type: {type(rect)}")
                    logger.warning(f"Available attributes: {[attr for attr in dir(rect) if not attr.startswith('_')]}")
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
    """库存板材优化器 - 支持 MaxRects BAF 和 Guillotine BAF+MINAS 算法"""
    
    def __init__(self, config: CuttingConfig, algorithm: str = "maxrects_baf"):
        """
        Args:
            config: 切割配置
            algorithm: 算法选择
                - "maxrects_baf": MaxRects Best Area Fit（默认）
                - "guillotine_baf_minas": Guillotine BAF + Min Area Split
        """
        self.config = config
        self.algorithm = algorithm.lower()
    
    def _create_packer(self, width: int, height: int) -> BaseStockPacker:
        """根据算法选择创建相应的装箱器"""
        if self.algorithm == "guillotine_baf_minas":
            return GuillotineBafMinasPacker(width, height, self.config)
        else:  # 默认使用 maxrects_baf
            return MaxRectsBafPacker(width, height, self.config)
    
    def fill_with_stock(self, width: int, height: int, existing_cuts: List[Cut], 
                   stock_plates: List[SmallPlate], optimize: bool = False) -> List[Cut]:
        """用库存板材填充剩余空间"""
        if not stock_plates:
            return []

        def _try_stock_arrangement(sorted_stock: List[SmallPlate]) -> Tuple[List[Cut], float]:
            """
            尝试一种库存板排列，返回切割结果和利用率
            
            Args:
                sorted_stock: 排序后的库存板列表
                
            Returns:
                (切割结果, 利用率)
            """
            # 创建新的装箱器
            packer = self._create_packer(width, height)
            
            # 处理已占用区域
            if isinstance(packer, MaxRectsBafPacker):
                for cut in existing_cuts:
                    occupied_rect = Rectangle(
                        cut.x1, cut.y1, 
                        cut.x2 - cut.x1 + self.config.blade_thickness, 
                        cut.y2 - cut.y1 + self.config.blade_thickness
                    )
                    packer._split(occupied_rect)
                    packer._remove_duplicates()
            elif isinstance(packer, GuillotineBafMinasPacker):
                self._update_guillotine_sections(packer, existing_cuts)
            
            # 尝试多轮填充以最大化利用率
            max_rounds = 10
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
            
            # 更新库存切割的 plate_id
            for i, cut in enumerate(packer.cuts):
                if not cut.plate.plate_id:
                    cut.plate.plate_id = f"STOCK_{i+1}"
            
            # 计算利用率
            utilization = packer.get_utilization()
            
            return packer.cuts, utilization

        if optimize:
            # 优化模式：尝试多种库存板排列顺序
            logger.info("启用库存优化模式，尝试多种排列...")
            
            best_cuts = []
            best_utilization = 0.0
            best_arrangement = "默认"
            
            # 生成不同的排列策略
            arrangements = []
            
            # 1. 按面积从大到小排序（原有逻辑）
            arrangements.append(("面积降序", sorted(stock_plates, key=lambda p: p.area, reverse=True)))
            
            # 2. 按面积从小到大排序
            arrangements.append(("面积升序", sorted(stock_plates, key=lambda p: p.area)))
            
            # 3. 按长度降序排列
            arrangements.append(("长度降序", sorted(stock_plates, key=lambda p: p.length, reverse=True)))
            
            # 4. 按宽度降序排列
            arrangements.append(("宽度降序", sorted(stock_plates, key=lambda p: p.width, reverse=True)))
            
            # 5. 按周长降序排列
            arrangements.append(("周长降序", sorted(stock_plates, key=lambda p: 2*(p.length + p.width), reverse=True)))
            
            # 6. 原始顺序
            arrangements.append(("原始顺序", stock_plates.copy()))
            
            # 7. 尝试从不同位置开始的循环排列（限制数量避免过度计算）
            if len(stock_plates) <= 10:  # 只对较小的库存列表尝试循环排列
                for i in range(1, min(len(stock_plates), 6)):  # 最多尝试前3个位置开始
                    rotated = stock_plates[i:] + stock_plates[:i]
                    arrangements.append((f"从第{i+1}个开始", rotated))
            
            # 8. 如果库存板种类较少，尝试优化的混合策略
            if len(set((p.length, p.width) for p in stock_plates)) <= 6:
                # 大板优先，小板填充策略
                large_plates = [p for p in stock_plates if p.area >= 1.2*10**5] 
                small_plates = [p for p in stock_plates if p.area < 1.2*10**5]
                large_first = sorted(large_plates, key=lambda p: p.area, reverse=True) + \
                            sorted(small_plates, key=lambda p: p.area)
                arrangements.append(("大板优先", large_first))
                
                # 长条优先策略
                long_plates = [p for p in stock_plates if max(p.length, p.width) / min(p.length, p.width) >= 2]
                square_plates = [p for p in stock_plates if max(p.length, p.width) / min(p.length, p.width) < 2]
                long_first = sorted(long_plates, key=lambda p: p.area, reverse=True) + \
                            sorted(square_plates, key=lambda p: p.area, reverse=True)
                arrangements.append(("长条优先", long_first))
            
            # 测试每种排列
            for arrangement_name, sorted_stock in arrangements:
                logger.debug(f"测试排列: {arrangement_name}")
                
                try:
                    cuts, utilization = _try_stock_arrangement(sorted_stock)
                    
                    logger.debug(f"  {arrangement_name} - 利用率: {utilization:.3%}, 切割数: {len(cuts)}")
                    
                    # 选择更优的方案
                    # 优先考虑利用率，其次考虑切割数量
                    is_better = False
                    if utilization > best_utilization + 0.001:  # 利用率高0.1%以上
                        is_better = True
                    elif abs(utilization - best_utilization) <= 0.001:  # 利用率相近
                        if len(cuts) < len(best_cuts):  # 切割数量更少
                            is_better = True
                    
                    if is_better:
                        best_cuts = cuts
                        best_utilization = utilization
                        best_arrangement = arrangement_name
                        
                except Exception as e:
                    logger.warning(f"测试排列 {arrangement_name} 时发生错误: {e}")
                    continue
            
            # 输出最优结果信息
            algorithm_name = "Guillotine BAF+MINAS" if self.algorithm == "guillotine_baf_minas" else "MaxRects BAF"
            logger.info(f"库存填充完成（{algorithm_name}）")
            logger.info(f"最优排列: {best_arrangement}")
            logger.info(f"总利用率: {best_utilization:.2%}")
            logger.info(f"放置了 {len(best_cuts)} 块库存板材")
            
            # 显示排列优化的收益
            if best_arrangement != "原始顺序":
                # 计算默认方案的利用率进行比较
                default_cuts, default_utilization = _try_stock_arrangement(stock_plates)
                improvement = best_utilization - default_utilization
                if improvement > 0.001:
                    logger.info(f"相比默认排列提升利用率: +{improvement:.2%}")
            
            return best_cuts
            
        else:
            # 非优化模式：使用原始顺序
            logger.debug("使用原始顺序填充库存")
            cuts, utilization = _try_stock_arrangement(stock_plates)
            
            algorithm_name = "Guillotine BAF+MINAS" if self.algorithm == "guillotine_baf_minas" else "MaxRects BAF"
            logger.debug(f"库存填充完成（{algorithm_name}），利用率: {utilization:.2%}，放置了{len(cuts)}块库存板材")
            
            return cuts
    
    def _update_guillotine_sections(self, packer: GuillotineBafMinasPacker, existing_cuts: List[Cut]):
        """更新Guillotine算法的空闲区域，排除已占用部分"""
        # 简化实现：重新计算空闲区域
        # 这里可以实现更复杂的算法来精确计算剩余空闲区域
        occupied_rects = []
        for cut in existing_cuts:
            occupied_rects.append(Rectangle(
                cut.x1, cut.y1,
                cut.x2 - cut.x1 + self.config.blade_thickness,
                cut.y2 - cut.y1 + self.config.blade_thickness
            ))
        
        # 对于Guillotine算法，我们需要从初始区域中减去已占用区域
        # 这里使用简化的方法：将初始区域分割成更小的空闲区域
        if occupied_rects:
            # 清空当前sections
            packer._sections = []
            # 创建初始大区域
            initial_section = Rectangle(0, 0, packer.width, packer.height)
            # 通过占用区域进行分割（简化处理）
            free_sections = self._compute_free_sections(initial_section, occupied_rects)
            for section in free_sections:
                packer._add_section(section)
    
    def _compute_free_sections(self, container: Rectangle, occupied: List[Rectangle]) -> List[Rectangle]:
        """计算除去已占用区域后的空闲区域（简化版本）"""
        # 这是一个简化实现，实际应用中可能需要更复杂的算法
        free_sections = []
        
        # 找出所有占用区域的边界
        x_coords = [0, container.width]
        y_coords = [0, container.height]
        
        for rect in occupied:
            x_coords.extend([rect.x, rect.right])
            y_coords.extend([rect.y, rect.top])
        
        x_coords = sorted(set(x_coords))
        y_coords = sorted(set(y_coords))
        
        # 检查每个网格单元是否空闲
        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                x, y = x_coords[i], y_coords[j]
                w, h = x_coords[i+1] - x, y_coords[j+1] - y
                
                # 检查这个区域是否与任何占用区域重叠
                test_rect = Rectangle(x, y, w, h)
                is_free = True
                for occ in occupied:
                    if test_rect.intersects(occ):
                        is_free = False
                        break
                
                if is_free and w > 0 and h > 0:
                    free_sections.append(Rectangle(x, y, w, h))
        
        # 合并相邻的空闲区域
        merged = self._merge_adjacent_sections(free_sections)
        return merged
    
    def _merge_adjacent_sections(self, sections: List[Rectangle]) -> List[Rectangle]:
        """合并相邻的空闲区域"""
        if not sections:
            return []
        
        merged = []
        used = [False] * len(sections)
        
        for i, s1 in enumerate(sections):
            if used[i]:
                continue
            
            current = Rectangle(s1.x, s1.y, s1.width, s1.height)
            merged_any = True
            
            while merged_any:
                merged_any = False
                for j, s2 in enumerate(sections):
                    if used[j] or j == i:
                        continue
                    
                    # 尝试水平合并
                    if (current.y == s2.y and current.height == s2.height):
                        if current.right == s2.x:
                            current.width += s2.width
                            used[j] = True
                            merged_any = True
                        elif s2.right == current.x:
                            current.x = s2.x
                            current.width += s2.width
                            used[j] = True
                            merged_any = True
                    
                    # 尝试垂直合并
                    elif (current.x == s2.x and current.width == s2.width):
                        if current.top == s2.y:
                            current.height += s2.height
                            used[j] = True
                            merged_any = True
                        elif s2.top == current.y:
                            current.y = s2.y
                            current.height += s2.height
                            used[j] = True
                            merged_any = True
            
            merged.append(current)
            used[i] = True
        
        return merged


# ============================================================================
# 数据转换器
# ============================================================================

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


# ============================================================================
# 工具函数
# ============================================================================

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


def compare_algorithms(metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> int:
    """
    比较两个算法的优劣
    
    Returns:
        -1: metrics1 更优
         0: 相同
         1: metrics2 更优
    """
    # 1. 首先比较使用板材数量（越少越好）
    if metrics1['used_plates'] < metrics2['used_plates']:
        return -1
    elif metrics1['used_plates'] > metrics2['used_plates']:
        return 1
    
    # # 2. 板材数量相同，比较总体利用率（越高越好）
    # rate_diff = abs(metrics1['overall_rate'] - metrics2['overall_rate'])
    # if rate_diff > 0.001:  # 利用率差异大于0.1%
    #     return -1 if metrics1['overall_rate'] > metrics2['overall_rate'] else 1
    
    # # 3. 利用率相近，比较最小利用率（避免某块板材利用率特别低）
    # min_rate_diff = abs(metrics1['min_rate'] - metrics2['min_rate'])
    # if min_rate_diff > 0.01:  # 差异大于1%
    #     return -1 if metrics1['min_rate'] > metrics2['min_rate'] else 1
    
    # # 4. 比较利用率方差（越小越好，表示各板材利用率更均匀）
    # variance_diff = abs(metrics1['rate_variance'] - metrics2['rate_variance'])
    # if variance_diff > 0.0001:
    #     return -1 if metrics1['rate_variance'] < metrics2['rate_variance'] else 1
    
    # 5. 比较切割复杂度（切割次数越少越好，降低加工成本）
    if metrics1['avg_cuts_per_plate'] != metrics2['avg_cuts_per_plate']:
        return -1 if metrics1['avg_cuts_per_plate'] < metrics2['avg_cuts_per_plate'] else 1
    
    # 6. 比较最大单板切割数（越大越好，能放更多板）
    if metrics1['max_cuts_single_plate'] != metrics2['max_cuts_single_plate']:
        return -1 if metrics1['max_cuts_single_plate'] > metrics2['max_cuts_single_plate'] else 1
    
    # 7. 如果所有指标都相同，返回相等
    return 0


# ============================================================================
# 主要函数
# ============================================================================

def run_single_algorithm(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]], 
                        others: List[Dict[str, Any]], optim: int, saw_blade: int,
                        algorithm, stock_algorithm: str = "maxrects_baf") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    运行单个算法的切割方案
    
    Args:
        stock_algorithm: 库存填充算法
            - "maxrects_baf": MaxRects Best Area Fit（默认）
            - "guillotine_baf_minas": Guillotine BAF + Min Area Split
    
    Returns:
        (切割方案列表, 评价指标字典)
    """
    # 配置
    config = CuttingConfig(blade_thickness=saw_blade)
    
    # 数据转换
    converter = DataConverter()
    big_plates = converter.convert_plates(plates)
    small_plates = converter.convert_orders(orders)
    stock_plates = converter.convert_stock(others) if others else []
    
    if not big_plates:
        return [], calculate_cutting_metrics([], len(small_plates))
    
    # 创建优化器
    plate_optimizer = PlateOptimizer(config, algorithm)
    stock_optimizer = StockOptimizer(config, stock_algorithm)  # 添加算法参数
    
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
    
    # 计算详细指标
    metrics = calculate_cutting_metrics(results, len(remaining_orders))
    
    return results, metrics


def optimize_cutting(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]], 
                    others: List[Dict[str, Any]] = None, optim: int = 0, 
                    saw_blade: int = 4, algorithm: str = "auto",
                    stock_algorithm: str = "guillotine_baf_minas") -> List[Dict[str, Any]]:
    """
    主优化函数
    
    Args:
        plates: 大板信息列表
        orders: 订单信息列表  
        others: 库存余料列表
        optim: 是否启用库存优化（仅影响库存板）
        saw_blade: 锯片厚度
        algorithm: 主算法选择
            - "MaxRectsBaf": MaxRects Best Area Fit
            - "GuillotineBafMinas": Guillotine Best Area Fit with Minimal Area Split
            - "SkylineMwfWm": Skyline Minimal Waste Fit with Merge
            - "auto": 自动优化模式（默认）- 尝试三种算法，选择最优
        stock_algorithm: 库存填充算法
            - "maxrects_baf": MaxRects Best Area Fit（默认）
            - "guillotine_baf_minas": Guillotine BAF + Min Area Split
    
    Returns:
        切割方案列表
    """
    
    # 定义可用算法映射
    ALGORITHMS = {
        "MaxRectsBaf": rectpack.MaxRectsBaf,
        "GuillotineBafMinas": rectpack.GuillotineBafMinas,
        "SkylineMwfWm": rectpack.SkylineMwfWm,
        "GuillotineBssfLlas": rectpack.GuillotineBssfLlas,
    }
    
    # 定义库存算法名称映射
    STOCK_ALGORITHMS = {
        "maxrects_baf": "MaxRects BAF",
        "guillotine_baf_minas": "Guillotine BAF+MINAS"
    }
    
    if algorithm == "auto":
        # 自动优化模式：尝试所有算法，选择最优
        logger.info(f"使用自动优化模式，测试多种算法...")
        logger.info(f"库存填充策略: {STOCK_ALGORITHMS.get(stock_algorithm, stock_algorithm)}")
        
        best_results = None
        best_metrics = None
        best_algorithm_name = None
        
        algorithm_results = []
        
        for algo_name, algo_class in ALGORITHMS.items():
            logger.info(f"测试算法: {algo_name}")
            
            results, metrics = run_single_algorithm(
                plates, orders, others, optim, saw_blade, algo_class, stock_algorithm
            )
            
            algorithm_results.append((algo_name, results, metrics))
            
            # 详细日志
            logger.info(f"  {algo_name} 结果:")
            logger.info(f"    - 使用板材: {metrics['used_plates']} 块")
            logger.info(f"    - 平均利用率: {metrics['overall_rate']:.2%}")
            logger.info(f"    - 最低利用率: {metrics['min_rate']:.2%}")
            logger.info(f"    - 利用率方差: {metrics['rate_variance']:.4f}")
            logger.info(f"    - 平均切割数: {metrics['avg_cuts_per_plate']:.1f} 次/板")
            logger.info(f"    - 最大单板切割: {metrics['max_cuts_single_plate']} 次")
            logger.info(f"    - 剩余订单: {metrics['remaining_orders']} 个")
            
            # 比较选择最优
            if best_metrics is None:
                best_results = results
                best_metrics = metrics
                best_algorithm_name = algo_name
            else:
                comparison = compare_algorithms(metrics, best_metrics)
                if comparison < 0:
                    best_results = results
                    best_metrics = metrics
                    best_algorithm_name = algo_name
        
        # 输出最终选择理由
        logger.info(f"\n最优算法: {best_algorithm_name}")
        logger.info(f"选择理由:")
        
        # 分析为什么选择这个算法
        for algo_name, _, metrics in algorithm_results:
            if algo_name != best_algorithm_name:
                comparison = compare_algorithms(best_metrics, metrics)
                if best_metrics['used_plates'] < metrics['used_plates']:
                    logger.info(f"  - 比 {algo_name} 少用 {metrics['used_plates'] - best_metrics['used_plates']} 块板")
                # elif best_metrics['overall_rate'] > metrics['overall_rate']:
                #     logger.info(f"  - 比 {algo_name} 利用率高 {(best_metrics['overall_rate'] - metrics['overall_rate'])*100:.2f}%")
                # elif best_metrics['min_rate'] > metrics['min_rate']:
                #     logger.info(f"  - 比 {algo_name} 最低利用率高 {(best_metrics['min_rate'] - metrics['min_rate'])*100:.2f}%")
                # elif best_metrics['rate_variance'] < metrics['rate_variance']:
                #     logger.info(f"  - 比 {algo_name} 利用率更均匀（方差小 {metrics['rate_variance'] - best_metrics['rate_variance']:.4f}）")
                elif best_metrics['avg_cuts_per_plate'] < metrics['avg_cuts_per_plate']:
                    logger.info(f"  - 比 {algo_name} 切割更简单（平均多 {metrics['avg_cuts_per_plate'] - best_metrics['avg_cuts_per_plate']:.1f} 次/板）")
        
        return best_results
        
    elif algorithm in ALGORITHMS:
        # 使用指定算法
        logger.info(f"使用算法: {algorithm}")
        logger.info(f"库存填充策略: {STOCK_ALGORITHMS.get(stock_algorithm, stock_algorithm)}")
        results, metrics = run_single_algorithm(
            plates, orders, others, optim, saw_blade, ALGORITHMS[algorithm], stock_algorithm
        )
        logger.info(f"完成切割:")
        logger.info(f"  - 使用板材: {metrics['used_plates']} 块")
        logger.info(f"  - 平均利用率: {metrics['overall_rate']:.2%}")
        logger.info(f"  - 平均切割数: {metrics['avg_cuts_per_plate']:.1f} 次/板")
        return results
        
    else:
        # 无效算法名称，使用默认算法
        logger.warning(f"未知算法 '{algorithm}'，使用默认算法 MaxRectsBssf")
        results, metrics = run_single_algorithm(
            plates, orders, others, optim, saw_blade, rectpack.MaxRectsBssf, stock_algorithm
        )
        return results


# ============================================================================
# 主程序入口
# ============================================================================

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
    
    print("=== 板材切割优化器演示 ===\n")
    
    # 1. 使用自动优化模式 + MaxRects BAF库存算法
    print("1. 自动优化模式 + MaxRects BAF库存算法:")
    results_auto = optimize_cutting(plates, orders, others, optim=1, 
                                   algorithm="auto", stock_algorithm="maxrects_baf")
    print(f"   生成 {len(results_auto)} 个切割方案\n")
    
    # 2. 使用自动优化模式 + Guillotine BAF+MINAS库存算法
    print("2. 自动优化模式 + Guillotine BAF+MINAS库存算法:")
    results_guillotine = optimize_cutting(plates, orders, others, optim=1, 
                                         algorithm="auto", stock_algorithm="guillotine_baf_minas")
    print(f"   生成 {len(results_guillotine)} 个切割方案\n")
    
    # 3. 使用MaxRects BAF主算法 + MaxRects BAF库存算法
    print("3. MaxRects BAF主算法 + MaxRects BAF库存算法:")
    results_maxrects = optimize_cutting(plates, orders, others, optim=1, 
                                       algorithm="MaxRectsBaf", stock_algorithm="maxrects_baf")
    print(f"   生成 {len(results_maxrects)} 个切割方案\n")
    
    # 4. 使用MaxRects BAF主算法 + Guillotine BAF+MINAS库存算法
    print("4. MaxRects BAF主算法 + Guillotine BAF+MINAS库存算法:")
    results_mixed = optimize_cutting(plates, orders, others, optim=1, 
                                    algorithm="MaxRectsBaf", stock_algorithm="guillotine_baf_minas")
    print(f"   生成 {len(results_mixed)} 个切割方案\n")
    
    # 显示库存算法说明
    print("=== 库存填充算法说明 ===")
    print("MaxRects BAF: 使用最大矩形算法，选择面积最小的可用区域")
    print("Guillotine BAF+MINAS: 使用切割线算法，采用最小面积分割策略")