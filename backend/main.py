import copy
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

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
    tolerance: int = 30  # 容差值
    attach_distance: int = 80  # 边界吸附距离
    rotation_ratio: float = 0.4  # 旋转判断比例阈值
    area_threshold: float = 1.2e5  # 面积阈值
    narrow_width_threshold: int = 400  # 窄板判断阈值
    max_optimization_attempts: int = 10  # 最大优化尝试次数


class RotationStrategy(Enum):
    """旋转策略枚举"""
    NO_ROTATION = "no_rotation"
    SMART_ROTATION = "smart_rotation"
    FORCE_ROTATION = "force_rotation"


@dataclass
class Gap:
    """表示空隙/余料区域"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def can_fit_plate(self, length: int, width: int) -> bool:
        """检查是否能放下指定尺寸的板材"""
        return self.width >= length and self.height >= width


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
    
    def rotated(self) -> 'SmallPlate':
        """返回旋转后的板材"""
        return SmallPlate(self.width, self.length, self.plate_id, self.quantity)


@dataclass
class Cut:
    """切割记录"""
    plate: SmallPlate
    x1: int
    y1: int
    x2: int
    y2: int
    is_stock: bool = False


class GapManager:
    """空隙管理器"""
    
    def __init__(self, config: CuttingConfig):
        self.config = config
        self.gaps: List[Gap] = []
    
    def add_gap(self, gap: Gap) -> None:
        """添加空隙，尝试与现有空隙合并"""
        merged = False
        for existing_gap in self.gaps:
            if self._can_merge(gap, existing_gap):
                merged_gap = self._merge_gaps(gap, existing_gap)
                self.gaps.remove(existing_gap)
                self.gaps.append(merged_gap)
                merged = True
                break
        
        if not merged:
            self.gaps.append(gap)
            logger.debug(f"添加新空隙: ({gap.x1}, {gap.y1}, {gap.x2}, {gap.y2})")
    
    def _can_merge(self, gap1: Gap, gap2: Gap) -> bool:
        """判断两个空隙是否可以合并"""
        tolerance = self.config.tolerance
        
        # 检查水平合并
        if (abs(gap1.y1 - gap2.y1) <= tolerance and 
            abs(gap1.y2 - gap2.y2) <= tolerance and
            (gap1.x2 == gap2.x1 or gap1.x1 == gap2.x2)):
            return True
        
        # 检查垂直合并
        if (abs(gap1.x1 - gap2.x1) <= tolerance and 
            abs(gap1.x2 - gap2.x2) <= tolerance and
            (gap1.y2 == gap2.y1 or gap1.y1 == gap2.y2)):
            return True
        
        return False
    
    def _merge_gaps(self, gap1: Gap, gap2: Gap) -> Gap:
        """合并两个空隙"""
        return Gap(
            min(gap1.x1, gap2.x1),
            min(gap1.y1, gap2.y1),
            max(gap1.x2, gap2.x2),
            max(gap1.y2, gap2.y2)
        )
    
    def find_best_gap(self, plate: SmallPlate) -> Optional[Gap]:
        """找到最适合放置板材的空隙"""
        suitable_gaps = [gap for gap in self.gaps 
                        if gap.can_fit_plate(plate.length, plate.width)]
        
        if not suitable_gaps:
            return None
        
        # 选择面积最小的合适空隙（最佳拟合）
        return min(suitable_gaps, key=lambda g: g.area)
    
    def remove_gap(self, gap: Gap) -> None:
        """移除空隙"""
        if gap in self.gaps:
            self.gaps.remove(gap)
    
    def update_gaps_after_cut(self, cut: Cut) -> None:
        """切割后更新空隙"""
        # 移除被占用的空隙，添加新产生的空隙
        gaps_to_remove = []
        new_gaps = []
        
        for gap in self.gaps:
            if self._gap_intersects_cut(gap, cut):
                gaps_to_remove.append(gap)
                # 计算剩余空隙
                remaining_gaps = self._calculate_remaining_gaps(gap, cut)
                new_gaps.extend(remaining_gaps)
        
        for gap in gaps_to_remove:
            self.remove_gap(gap)
        
        for gap in new_gaps:
            self.add_gap(gap)
    
    def _gap_intersects_cut(self, gap: Gap, cut: Cut) -> bool:
        """检查空隙是否与切割区域相交"""
        return not (cut.x2 <= gap.x1 or cut.x1 >= gap.x2 or 
                   cut.y2 <= gap.y1 or cut.y1 >= gap.y2)
    
    def _calculate_remaining_gaps(self, original_gap: Gap, cut: Cut) -> List[Gap]:
        """计算切割后剩余的空隙"""
        remaining_gaps = []
        
        # 右侧空隙
        if cut.x2 + self.config.blade_thickness < original_gap.x2:
            right_gap = Gap(
                cut.x2 + self.config.blade_thickness, 
                original_gap.y1,
                original_gap.x2, 
                cut.y2
            )
            if right_gap.area > 0:
                remaining_gaps.append(right_gap)
        
        # 下方空隙
        if cut.y2 + self.config.blade_thickness < original_gap.y2:
            bottom_gap = Gap(
                original_gap.x1, 
                cut.y2 + self.config.blade_thickness,
                original_gap.x2, 
                original_gap.y2
            )
            if bottom_gap.area > 0:
                remaining_gaps.append(bottom_gap)
        
        return remaining_gaps


class RotationOptimizer:
    """旋转优化器"""
    
    def __init__(self, config: CuttingConfig):
        self.config = config
    
    def should_rotate(self, plate: SmallPlate, container_length: int, 
                     container_width: int, strategy: RotationStrategy = RotationStrategy.SMART_ROTATION) -> bool:
        """判断是否应该旋转板材"""
        if strategy == RotationStrategy.NO_ROTATION:
            return False
        
        if strategy == RotationStrategy.FORCE_ROTATION:
            return True
        
        # 智能旋转判断
        original_fit = self._calculate_fit_count(
            plate.length, plate.width, container_length, container_width
        )
        rotated_fit = self._calculate_fit_count(
            plate.width, plate.length, container_length, container_width
        )
        
        # 如果旋转后能放下更多，且板材不是太大或形状接近正方形，则旋转
        if rotated_fit > original_fit:
            if (plate.area < self.config.area_threshold or 
                abs(plate.length - plate.width) / max(plate.length, plate.width) < self.config.rotation_ratio):
                return True
        
        return False
    
    def _calculate_fit_count(self, plate_length: int, plate_width: int, 
                           container_length: int, container_width: int) -> int:
        """计算在给定容器中能放下多少个板材"""
        if plate_length == 0 or plate_width == 0:
            return 0
        
        horizontal_count = container_length // plate_length
        vertical_count = container_width // plate_width
        return horizontal_count * vertical_count


class PlateOptimizer:
    """板材优化器"""
    
    def __init__(self, config: CuttingConfig = None):
        self.config = config or CuttingConfig()
        self.rotation_optimizer = RotationOptimizer(self.config)
    
    def sort_plates_by_efficiency(self, plates: List[SmallPlate], 
                                 reference_length: int, reference_width: int) -> List[SmallPlate]:
        """按照切割效率对板材排序"""
        # 统计每种宽度的出现次数
        width_counts = Counter(plate.width for plate in plates)
        
        # 忽略过窄板材的计数
        for width in list(width_counts.keys()):
            if width < self.config.narrow_width_threshold:
                width_counts[width] = 0
        
        # 预处理：智能旋转
        processed_plates = []
        for plate in plates:
            if self.rotation_optimizer.should_rotate(plate, reference_length, reference_width):
                processed_plates.append(plate.rotated())
                logger.debug(f"预旋转板材: {plate.length}x{plate.width} -> {plate.width}x{plate.length}")
            else:
                processed_plates.append(plate)
        
        # 排序：按宽度计数、宽度、长度降序
        processed_plates.sort(
            key=lambda p: (width_counts[p.width], p.width, p.length), 
            reverse=True
        )
        
        return processed_plates


class Plate:
    """优化后的板材类"""
    
    def __init__(self, length: int, width: int, config: CuttingConfig = None):
        self.length = length
        self.width = width
        self.config = config or CuttingConfig()
        
        # 状态变量
        self.cuts: List[Cut] = []
        self.current_x = 0
        self.current_y = 0
        self.row_height = 0
        self.last_cut_size = (0, 0)
        
        # 管理器
        self.gap_manager = GapManager(self.config)
        self.rotation_optimizer = RotationOptimizer(self.config)
        
        logger.debug(f"创建板材: {length}x{width}")
    
    @property
    def used_area(self) -> int:
        """已使用面积"""
        return sum(cut.plate.area for cut in self.cuts)
    
    @property
    def total_area(self) -> int:
        """总面积"""
        return self.length * self.width
    
    @property
    def utilization_rate(self) -> float:
        """利用率"""
        if self.total_area == 0:
            return 0.0
        return (self.used_area / self.total_area) * 100
    
    def can_fit_plate(self, plate: SmallPlate) -> bool:
        """检查是否能放下板材"""
        # 检查当前行
        if (self.current_x + plate.length <= self.length and 
            self.current_y + plate.width <= self.width):
            return True
        
        # 检查新行
        if self.current_y + self.row_height + plate.width <= self.width:
            return True
        
        return False
    
    def add_cut_in_gap(self, plate: SmallPlate) -> bool:
        """在空隙中添加切割"""
        # 尝试原始方向
        gap = self.gap_manager.find_best_gap(plate)
        if gap:
            return self._place_in_gap(plate, gap)
        
        # 尝试旋转方向
        rotated_plate = plate.rotated()
        gap = self.gap_manager.find_best_gap(rotated_plate)
        if gap:
            return self._place_in_gap(rotated_plate, gap)
        
        return False
    
    def _place_in_gap(self, plate: SmallPlate, gap: Gap) -> bool:
        """在指定空隙中放置板材"""
        cut = Cut(
            plate=plate,
            x1=gap.x1,
            y1=gap.y1,
            x2=gap.x1 + plate.length,
            y2=gap.y1 + plate.width,
            is_stock=plate.plate_id.startswith('R')
        )
        
        self.cuts.append(cut)
        self.gap_manager.remove_gap(gap)
        self.gap_manager.update_gaps_after_cut(cut)
        
        logger.info(f"在空隙中切割: {plate.length}x{plate.width} at ({gap.x1}, {gap.y1})")
        return True
    
    def add_cut_in_main_area(self, plate: SmallPlate) -> bool:
        """在主区域添加切割"""
        if not self.can_fit_plate(plate):
            return False
        
        # 智能旋转判断
        if self._should_rotate_for_main_area(plate):
            plate = plate.rotated()
            logger.debug("为主区域放置旋转板材")
        
        # 检查是否需要换行
        if self.current_x + plate.length > self.length:
            self._start_new_row()
        
        # 记录切割
        cut = Cut(
            plate=plate,
            x1=self.current_x,
            y1=self.current_y,
            x2=self.current_x + plate.length,
            y2=self.current_y + plate.width,
            is_stock=plate.plate_id.startswith('R')
        )
        
        self.cuts.append(cut)
        self._update_position_after_cut(cut)
        self._handle_edge_attachment(cut)
        
        logger.info(f"在主区域切割: {plate.length}x{plate.width} at ({cut.x1}, {cut.y1})")
        return True
    
    def _should_rotate_for_main_area(self, plate: SmallPlate) -> bool:
        """判断在主区域是否应该旋转"""
        # 如果与上次切割尺寸相反，且满足条件，则旋转
        if ((plate.width, plate.length) == self.last_cut_size and 
            plate.length > 0 and plate.width > 0 and
            (plate.area < self.config.area_threshold or 
             abs(plate.length - plate.width) / max(plate.length, plate.width) < self.config.rotation_ratio)):
            return True
        
        # 如果旋转后能获得更好的拟合效果
        remaining_length = self.length - self.current_x
        remaining_width = self.width - self.current_y - self.row_height
        
        if self.rotation_optimizer.should_rotate(plate, remaining_length, remaining_width):
            return True
        
        return False
    
    def _start_new_row(self) -> None:
        """开始新行"""
        # 更新上一行的空隙
        if self.current_x < self.length and self.row_height > 0:
            row_gap = Gap(
                self.current_x, self.current_y, 
                self.length, self.current_y + self.row_height
            )
            self.gap_manager.add_gap(row_gap)
        
        self.current_x = 0
        self.current_y += self.row_height
        self.row_height = 0
    
    def _update_position_after_cut(self, cut: Cut) -> None:
        """切割后更新位置"""
        self.current_x = cut.x2 + self.config.blade_thickness
        self.row_height = max(self.row_height, cut.plate.width + self.config.blade_thickness)
        self.last_cut_size = (cut.plate.length, cut.plate.width)
        
        # 处理行内空隙
        if cut.plate.width + self.config.blade_thickness < self.row_height - self.config.tolerance:
            internal_gap = Gap(
                cut.x1, cut.y2 + self.config.blade_thickness,
                cut.x2, self.current_y + self.row_height
            )
            self.gap_manager.add_gap(internal_gap)
    
    def _handle_edge_attachment(self, cut: Cut) -> None:
        """处理边缘吸附"""
        if (cut.x1 == 0 and cut.y1 == 0 and 
            cut.y1 + self.row_height <= self.width and 
            cut.y1 + self.row_height > self.width - self.config.attach_distance):
            
            self.row_height = self.width - cut.y1
            edge_gap = Gap(
                cut.x1, cut.y2 + self.config.blade_thickness,
                cut.x2, self.width
            )
            self.gap_manager.add_gap(edge_gap)
            logger.debug("应用边缘吸附")


class StockOptimizer:
    """库存板材优化器"""
    
    def __init__(self, config: CuttingConfig = None):
        self.config = config or CuttingConfig()
    
    def optimize_stock_placement(self, plate: Plate, stock_plates: List[SmallPlate]) -> None:
        """优化库存板材放置顺序"""
        if not stock_plates:
            return
        
        base_util = plate.utilization_rate
        best_order = None
        best_util = base_util
        
        # 尝试不同的起始板材
        for i in range(min(self.config.max_optimization_attempts, len(stock_plates))):
            # 创建模拟板材
            sim_plate = copy.deepcopy(plate)
            
            # 重新排序库存板材
            test_order = stock_plates.copy()
            first_stock = test_order.pop(i)
            test_order.insert(0, first_stock)
            
            # 执行模拟填充
            self._fill_with_stock(sim_plate, test_order)
            
            # 检查结果
            sim_util = sim_plate.utilization_rate
            if sim_util > best_util:
                best_util = sim_util
                best_order = test_order
                logger.debug(f"找到更优库存排序，利用率: {best_util:.2f}%")
        
        # 应用最优排序
        if best_order:
            self._fill_with_stock(plate, best_order)
        else:
            self._fill_with_stock(plate, stock_plates)
    
    def _fill_with_stock(self, plate: Plate, stock_plates: List[SmallPlate]) -> None:
        """用库存板材填充"""
        for stock_plate in stock_plates:
            # 估算可以放置的数量
            remaining_area = (100 - plate.utilization_rate) / 100 * plate.total_area
            max_count = int(remaining_area / stock_plate.area) if stock_plate.area > 0 else 0
            
            for _ in range(max_count):
                placed = False
                
                # 尝试在空隙中放置
                if plate.add_cut_in_gap(stock_plate):
                    placed = True
                # 尝试在主区域放置
                elif plate.add_cut_in_main_area(stock_plate):
                    placed = True
                
                if not placed:
                    break  # 无法放置更多


def optimize_cutting(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]], 
                    others: List[Dict[str, Any]] = None, optim: int = 0, 
                    saw_blade: int = 4) -> List[Dict[str, Any]]:
    """
    主优化函数
    
    Args:
        plates: 大板信息列表
        orders: 订单信息列表  
        others: 库存余料列表
        optim: 是否启用库存优化
        saw_blade: 锯片厚度
    
    Returns:
        切割方案列表
    """
    logger.info("开始板材切割优化")
    
    # 配置
    config = CuttingConfig(blade_thickness=saw_blade)
    
    # 数据转换
    converter = DataConverter()
    big_plates = converter.convert_plates(plates)
    small_plates = converter.convert_orders(orders)
    stock_plates = converter.convert_stock(others) if others else []
    
    if not big_plates:
        logger.warning("没有可用的大板")
        return []
    
    logger.info(f"处理 {len(big_plates)} 块大板, {len(small_plates)} 个订单, {len(stock_plates)} 个库存")
    
    # 优化器
    plate_optimizer = PlateOptimizer(config)
    stock_optimizer = StockOptimizer(config)
    
    # 排序小板
    reference_plate = big_plates[0]
    sorted_small_plates = plate_optimizer.sort_plates_by_efficiency(
        small_plates, reference_plate.length, reference_plate.width
    )
    
    # 主切割循环
    used_plates = []
    remaining_plates = sorted_small_plates.copy()
    
    for i, big_plate_template in enumerate(big_plates):
        if not remaining_plates:
            break
        
        logger.info(f"处理第 {i+1}/{len(big_plates)} 块大板")
        
        # 创建板材实例
        plate = Plate(big_plate_template.length, big_plate_template.width, config)
        
        # 尝试放置小板
        placed_indices = []
        for j, small_plate in enumerate(remaining_plates):
            placed = False
            
            # 优先在空隙中放置
            if plate.add_cut_in_gap(small_plate):
                placed = True
            # 在主区域放置
            elif plate.add_cut_in_main_area(small_plate):
                placed = True
            
            if placed:
                placed_indices.append(j)
        
        # 移除已放置的小板
        if placed_indices:
            used_plates.append(plate)
            for index in sorted(placed_indices, reverse=True):
                del remaining_plates[index]
    
    # 库存填充
    if stock_plates:
        logger.info("开始库存填充")
        for plate in used_plates:
            if optim:
                stock_optimizer.optimize_stock_placement(plate, stock_plates)
            else:
                stock_optimizer._fill_with_stock(plate, stock_plates)
    
    # 转换输出格式
    return converter.convert_to_output(used_plates)


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
                    length=max(item['length'], item['width']),
                    width=min(item['length'], item['width']),
                    plate_id=f"R{item.get('id', '')}"
                ))
        return result
    
    def convert_to_output(self, plates: List[Plate]) -> List[Dict[str, Any]]:
        """转换为输出格式"""
        result = []
        for plate in plates:
            if plate.cuts:
                cuts_data = []
                for cut in plate.cuts:
                    plate_id = cut.plate.plate_id
                    if cut.is_stock and plate_id.startswith('R'):
                        plate_id = plate_id[1:]  # 移除'R'前缀
                    
                    cuts_data.append([
                        cut.x1,  # start_x
                        cut.y1,  # start_y
                        cut.plate.length,  # length
                        cut.plate.width,   # width
                        1 if cut.is_stock else 0,  # is_stock
                        plate_id  # id
                    ])
                
                result.append({
                    'rate': plate.utilization_rate / 100,
                    'plate': [plate.length, plate.width],
                    'cutted': cuts_data
                })
        
        logger.info(f"生成 {len(result)} 个切割方案")
        return result


# 使用示例
if __name__ == "__main__":
    # 示例数据
    plates_data = [
        {'length': 2440, 'width': 1220, 'quantity': 2}
    ]
    
    orders_data = [
        {'length': 800, 'width': 600, 'quantity': 3, 'id': 'A1'},
        {'length': 900, 'width': 400, 'quantity': 2, 'id': 'A2'},
    ]
    
    stock_data = [
        {'length': 300, 'width': 200, 'id': 'S1'},
        {'length': 400, 'width': 300, 'id': 'S2'},
    ]
    
    # 执行优化
    result = optimize_cutting(plates_data, orders_data, stock_data, optim=1)
    
    # 打印结果
    for i, solution in enumerate(result):
        print(f"\n板材 {i+1}:")
        print(f"  利用率: {solution['rate']:.2%}")
        print(f"  尺寸: {solution['plate'][0]}x{solution['plate'][1]}")
        print(f"  切割数量: {len(solution['cutted'])}")