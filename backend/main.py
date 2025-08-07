import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import rectpack

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


class CustomStockPacker:
    """自定义库存板装箱器 - 实现 MaxRects 算法"""
    
    def __init__(self, width: int, height: int, config: CuttingConfig):
        self.width = width
        self.height = height
        self.config = config
        self.free_rects = [(0, 0, width, height)]  # (x, y, w, h)
        self.cuts: List[Cut] = []
        
    def add_rect(self, plate: SmallPlate) -> bool:
        """添加矩形"""
        best_rect = None
        best_score = float('inf')
        best_rotated = False
        best_index = -1
        
        # 尝试两种方向
        for rotated in [False, True]:
            w = plate.width if rotated else plate.length
            h = plate.length if rotated else plate.width
            
            # 在所有空闲矩形中寻找最佳位置
            for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
                if w <= fw and h <= fh:
                    # 使用最佳短边适配策略
                    leftover_x = fw - w
                    leftover_y = fh - h
                    score = min(leftover_x, leftover_y)
                    
                    if score < best_score:
                        best_score = score
                        best_rect = (fx, fy, fw, fh)
                        best_rotated = rotated
                        best_index = i
        
        if best_rect:
            fx, fy, fw, fh = best_rect
            
            # 确定实际尺寸
            actual_width = plate.width if best_rotated else plate.length
            actual_height = plate.length if best_rotated else plate.width
            
            # 记录切割
            cut = Cut(
                plate=plate,
                x1=fx,
                y1=fy,
                x2=fx + actual_width,
                y2=fy + actual_height,
                is_stock=True
            )
            self.cuts.append(cut)
            
            # 分割空闲矩形
            self._split_free_rect(best_index, fx, fy, actual_width, actual_height)
            
            return True
        
        return False
    
    def _split_free_rect(self, index: int, used_x: int, used_y: int, used_w: int, used_h: int):
        """分割空闲矩形"""
        fx, fy, fw, fh = self.free_rects[index]
        del self.free_rects[index]
        
        # 添加锯片厚度
        used_w_with_blade = used_w + self.config.blade_thickness
        used_h_with_blade = used_h + self.config.blade_thickness
        
        # 右侧剩余
        if used_x + used_w_with_blade < fx + fw:
            self.free_rects.append((
                used_x + used_w_with_blade,
                fy,
                fx + fw - used_x - used_w_with_blade,
                fh
            ))
        
        # 上方剩余
        if used_y + used_h_with_blade < fy + fh:
            self.free_rects.append((
                fx,
                used_y + used_h_with_blade,
                fw,
                fy + fh - used_y - used_h_with_blade
            ))
        
        # 清理被完全包含的矩形
        self._clean_free_rects()
    
    def _clean_free_rects(self):
        """清理被完全包含的空闲矩形"""
        i = 0
        while i < len(self.free_rects):
            j = i + 1
            while j < len(self.free_rects):
                if self._is_contained(self.free_rects[i], self.free_rects[j]):
                    del self.free_rects[i]
                    i -= 1
                    break
                elif self._is_contained(self.free_rects[j], self.free_rects[i]):
                    del self.free_rects[j]
                else:
                    j += 1
            i += 1
    
    def _is_contained(self, rect1: Tuple, rect2: Tuple) -> bool:
        """检查 rect1 是否被 rect2 包含"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2


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
    """库存板材优化器"""
    
    def __init__(self, config: CuttingConfig):
        self.config = config
    
    def fill_with_stock(self, width: int, height: int, existing_cuts: List[Cut], 
                       stock_plates: List[SmallPlate], optimize: bool = False) -> List[Cut]:
        """用库存板材填充剩余空间"""
        if not stock_plates:
            return []
        
        # 创建自定义装箱器
        packer = CustomStockPacker(width, height, self.config)
        
        # 首先标记已占用区域
        for cut in existing_cuts:
            # 简化处理：将已占用区域从空闲矩形中移除
            packer.free_rects = self._remove_occupied_area(
                packer.free_rects, 
                cut.x1, cut.y1, 
                cut.x2 - cut.x1, 
                cut.y2 - cut.y1
            )
        
        # 根据优化标志处理库存板
        if optimize:
            # 优化模式：按面积排序
            sorted_stock = sorted(stock_plates, key=lambda p: p.area, reverse=True)
        else:
            # 非优化模式：按原始顺序
            sorted_stock = stock_plates
        
        # 填充库存板材
        stock_cuts = []
        for stock in sorted_stock:
            # 尝试多次放置同一库存板材（无数量限制）
            while packer.add_rect(stock):
                # 成功放置一个，继续尝试
                pass
        
        # 更新库存切割的 plate_id
        for cut in packer.cuts:
            cut.plate.plate_id = f"R{cut.plate.plate_id}"
            stock_cuts.append(cut)
        
        return stock_cuts
    
    def _remove_occupied_area(self, free_rects: List[Tuple], x: int, y: int, w: int, h: int) -> List[Tuple]:
        """从空闲矩形列表中移除已占用区域"""
        new_rects = []
        
        for fx, fy, fw, fh in free_rects:
            # 检查是否有重叠
            if not (x >= fx + fw or x + w <= fx or y >= fy + fh or y + h <= fy):
                # 有重叠，需要分割
                # 左侧
                if fx < x:
                    new_rects.append((fx, fy, x - fx, fh))
                # 右侧
                if x + w < fx + fw:
                    new_rects.append((x + w, fy, fx + fw - x - w, fh))
                # 下方
                if fy < y:
                    new_rects.append((fx, fy, fw, y - fy))
                # 上方
                if y + h < fy + fh:
                    new_rects.append((fx, y + h, fw, fy + fh - y - h))
            else:
                # 无重叠，保留原矩形
                new_rects.append((fx, fy, fw, fh))
        
        return new_rects


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
            if cut.is_stock and plate_id.startswith('R'):
                plate_id = plate_id[1:]  # 移除'R'前缀
            
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
    
    # 2. 板材数量相同，比较总体利用率（越高越好）
    rate_diff = abs(metrics1['overall_rate'] - metrics2['overall_rate'])
    if rate_diff > 0.001:  # 利用率差异大于0.1%
        return -1 if metrics1['overall_rate'] > metrics2['overall_rate'] else 1
    
    # 3. 利用率相近，比较最小利用率（避免某块板材利用率特别低）
    min_rate_diff = abs(metrics1['min_rate'] - metrics2['min_rate'])
    if min_rate_diff > 0.01:  # 差异大于1%
        return -1 if metrics1['min_rate'] > metrics2['min_rate'] else 1
    
    # 4. 比较利用率方差（越小越好，表示各板材利用率更均匀）
    variance_diff = abs(metrics1['rate_variance'] - metrics2['rate_variance'])
    if variance_diff > 0.0001:
        return -1 if metrics1['rate_variance'] < metrics2['rate_variance'] else 1
    
    # 5. 比较切割复杂度（切割次数越少越好，降低加工成本）
    if metrics1['avg_cuts_per_plate'] != metrics2['avg_cuts_per_plate']:
        return -1 if metrics1['avg_cuts_per_plate'] < metrics2['avg_cuts_per_plate'] else 1
    
    # 6. 比较最大单板切割数（越小越好，降低单板加工复杂度）
    if metrics1['max_cuts_single_plate'] != metrics2['max_cuts_single_plate']:
        return -1 if metrics1['max_cuts_single_plate'] < metrics2['max_cuts_single_plate'] else 1
    
    # 7. 如果所有指标都相同，返回相等
    return 0


def run_single_algorithm(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]], 
                        others: List[Dict[str, Any]], optim: int, saw_blade: int,
                        algorithm) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    运行单个算法的切割方案
    
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
    stock_optimizer = StockOptimizer(config)
    
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
                    saw_blade: int = 4, algorithm: str = "auto") -> List[Dict[str, Any]]:
    """
    主优化函数
    
    Args:
        plates: 大板信息列表
        orders: 订单信息列表  
        others: 库存余料列表
        optim: 是否启用库存优化（仅影响库存板）
        saw_blade: 锯片厚度
        algorithm: 算法选择
            - "MaxRectsBaf": MaxRects Best Area Fit
            - "GuillotineBafMinas": Guillotine Best Area Fit with Minimal Area Split
            - "SkylineMwfWm": Skyline Minimal Waste Fit with Merge
            - "auto": 自动优化模式（默认）- 尝试三种算法，选择最优
    
    Returns:
        切割方案列表
    """
    
    # 定义可用算法映射
    ALGORITHMS = {
        "MaxRectsBaf": rectpack.MaxRectsBaf,
        "GuillotineBafMinas": rectpack.GuillotineBafMinas,
        "SkylineMwfWm": rectpack.SkylineMwfWm,
    }
    
    if algorithm == "auto":
        # 自动优化模式：尝试所有算法，选择最优
        logger.info("使用自动优化模式，测试多种算法...")
        
        best_results = None
        best_metrics = None
        best_algorithm_name = None
        
        algorithm_results = []
        
        for algo_name, algo_class in ALGORITHMS.items():
            logger.info(f"测试算法: {algo_name}")
            
            results, metrics = run_single_algorithm(
                plates, orders, others, optim, saw_blade, algo_class
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
                elif best_metrics['overall_rate'] > metrics['overall_rate']:
                    logger.info(f"  - 比 {algo_name} 利用率高 {(best_metrics['overall_rate'] - metrics['overall_rate'])*100:.2f}%")
                elif best_metrics['min_rate'] > metrics['min_rate']:
                    logger.info(f"  - 比 {algo_name} 最低利用率高 {(best_metrics['min_rate'] - metrics['min_rate'])*100:.2f}%")
                elif best_metrics['rate_variance'] < metrics['rate_variance']:
                    logger.info(f"  - 比 {algo_name} 利用率更均匀（方差小 {metrics['rate_variance'] - best_metrics['rate_variance']:.4f}）")
                elif best_metrics['avg_cuts_per_plate'] < metrics['avg_cuts_per_plate']:
                    logger.info(f"  - 比 {algo_name} 切割更简单（平均少 {metrics['avg_cuts_per_plate'] - best_metrics['avg_cuts_per_plate']:.1f} 次/板）")
        
        return best_results
        
    elif algorithm in ALGORITHMS:
        # 使用指定算法
        logger.info(f"使用算法: {algorithm}")
        results, metrics = run_single_algorithm(
            plates, orders, others, optim, saw_blade, ALGORITHMS[algorithm]
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
            plates, orders, others, optim, saw_blade, rectpack.MaxRectsBssf
        )
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
    
    # 使用自动优化模式
    results_auto = optimize_cutting(plates, orders, others, optim=1, algorithm="auto")
    print(f"自动优化模式: 生成 {len(results_auto)} 个切割方案")
    
    # 使用指定算法
    results_maxrects = optimize_cutting(plates, orders, others, optim=1, algorithm="MaxRectsBaf")
    print(f"MaxRectsBaf: 生成 {len(results_maxrects)} 个切割方案")