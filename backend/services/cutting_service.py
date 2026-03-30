import logging
from typing import Any, Dict, List, Optional, Tuple

import rectpack

from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter, calculate_cutting_metrics, compare_algorithms
from engine.optimizers import PlateOptimizer, StockOptimizer

logger = logging.getLogger('plate_cutting')

# ============================================================================
# 主要函数
# ============================================================================

def run_single_algorithm(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]],
                         others: List[Dict[str, Any]], optim: int, saw_blade: float,
                         algorithm, stock_algorithm: str = "maxrects_baf") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    运行单个算法的切割方案

    Args:
        stock_algorithm: 库存填充算法
            - "maxrects_baf": MaxRects Best Area Fit（默认）
            - "guillotine_bssf_llas": "Guillotine BSSF+LLAS"

    Returns:
        (切割方案列表, 评价指标字典)
    """
    # 配置
    config = CuttingConfig(blade_thickness=saw_blade)
    plates0 = [{**plate} for plate in plates]

    for plate_data in plates0:
        quantity = plate_data.get('quantity', 0)
        if quantity > 0:
            plate_data['length'] += config.blade_thickness
            plate_data['width'] += config.blade_thickness

    # 数据转换
    converter = DataConverter()
    big_plates = converter.convert_plates(plates0)
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
        order_cuts, remaining_orders = plate_optimizer.pack_orders(
            big_plate, remaining_orders)

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
            big_plate.length = big_plate.length - config.blade_thickness
            big_plate.width = big_plate.width - config.blade_thickness
            result = converter.convert_cuts_to_output(big_plate, all_cuts)
            results.append(result)

    # 计算详细指标
    metrics = calculate_cutting_metrics(results, len(remaining_orders))

    return results, metrics


def optimize_cutting(plates: List[Dict[str, Any]], orders: List[Dict[str, Any]],
                     others: List[Dict[str, Any]] = None, optim: int = 0,
                     saw_blade: float = 4.0, algorithm: str = "auto",
                     stock_algorithm: str = "maxrects_baf") -> List[Dict[str, Any]]:
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
            - "guillotine_bssf_llas": "Guillotine BSSF+LLAS"

    Returns:
        切割方案列表
    """

    # 定义可用算法映射
    ALGORITHMS = {
        "GuillotineBafMinas": rectpack.GuillotineBafMinas,
        "GuillotineBssfLlas": rectpack.GuillotineBssfLlas,
        "GuillotineBssfSlas": rectpack.GuillotineBssfSlas,
        "GuillotineBlsfLlas": rectpack.GuillotineBlsfLlas,
        "GuillotineBlsfSlas": rectpack.GuillotineBlsfSlas,
        "MaxRectsBaf": rectpack.MaxRectsBaf,
        "SkylineMwfWm": rectpack.SkylineMwfWm,
        # "GuillotineBssfSas": rectpack.GuillotineBssfSas,
        # "GuillotineBssfLas": rectpack.GuillotineBssfLas,
        # "GuillotineBssfSlas": rectpack.GuillotineBssfSlas,
        # "GuillotineBssfLlas": rectpack.GuillotineBssfLlas,
        # "GuillotineBssfMaxas": rectpack.GuillotineBssfMaxas,
        # "GuillotineBssfMinas": rectpack.GuillotineBssfMinas,
        # "GuillotineBlsfSas": rectpack.GuillotineBlsfSas,
        # "GuillotineBlsfLas": rectpack.GuillotineBlsfLas,
        # "GuillotineBlsfSlas": rectpack.GuillotineBlsfSlas,
        # "GuillotineBlsfLlas": rectpack.GuillotineBlsfLlas,
        # "GuillotineBlsfMaxas": rectpack.GuillotineBlsfMaxas,
        # "GuillotineBlsfMinas": rectpack.GuillotineBlsfMinas,
        # "GuillotineBafSas": rectpack.GuillotineBafSas,
        # "GuillotineBafLas": rectpack.GuillotineBafLas,
        # "GuillotineBafSlas": rectpack.GuillotineBafSlas,
        # "GuillotineBafLlas": rectpack.GuillotineBafLlas,
        # "GuillotineBafMaxas": rectpack.GuillotineBafMaxas,
        # "GuillotineBafMinas": rectpack.GuillotineBafMinas,
        # "SkylineBl": rectpack.SkylineBl,
        # "SkylineBlWm": rectpack.SkylineBlWm,
        # "SkylineMwf": rectpack.SkylineMwf,
        # "SkylineMwfl": rectpack.SkylineMwfl,
        # "SkylineMwfWm": rectpack.SkylineMwfWm,
        # "SkylineMwflWm": rectpack.SkylineMwflWm,
        # "MaxRectsBl": rectpack.MaxRectsBl,
        # "MaxRectsBssf": rectpack.MaxRectsBssf,
        # "MaxRectsBaf": rectpack.MaxRectsBaf,
        # "MaxRectsBlsf": rectpack.MaxRectsBlsf,
    }

    # 定义库存算法名称映射
    STOCK_ALGORITHMS = {
        "maxrects_baf": "MaxRects BAF",
        "guillotine_bssf_llas": "Guillotine BSSF+LLAS"
    }

    if algorithm == "auto":
        # 自动优化模式：尝试所有算法，选择最优
        logger.info("使用自动优化模式，测试多种算法...")
        logger.info(
            f"库存填充策略: {
                STOCK_ALGORITHMS.get(
                    stock_algorithm,
                    stock_algorithm)}")

        best_results = None
        best_metrics = None
        best_algorithm_name = None

        algorithm_results = []

        for algo_name, algo_class in ALGORITHMS.items():
            logger.info(f"测试算法: {algo_name}")

            results, metrics = run_single_algorithm(
                plates, orders, others, optim, saw_blade, algo_class, stock_algorithm)

            algorithm_results.append((algo_name, results, metrics))

            # 详细日志
            logger.info(f"  {algo_name} 结果:")
            logger.info(f"    - 使用板材: {metrics['used_plates']} 块")
            logger.info(f"    - 平均利用率: {metrics['overall_rate']:.2%}")
            logger.info(f"    - 最低利用率: {metrics['min_rate']:.2%}")
            logger.info(f"    - 利用率方差: {metrics['rate_variance']:.4f}")
            logger.info(
                f"    - 平均切割数: {metrics['avg_cuts_per_plate']:.1f} 次/板")
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
        logger.info("选择理由:")

        # 分析为什么选择这个算法
        for algo_name, _, metrics in algorithm_results:
            if algo_name != best_algorithm_name:
                comparison = compare_algorithms(best_metrics, metrics)
                if best_metrics['used_plates'] < metrics['used_plates']:
                    logger.info(
                        f"  - 比 {algo_name} 少用 {metrics['used_plates'] - best_metrics['used_plates']} 块板")
                elif best_metrics['last_rate'] < metrics['last_rate']:
                    logger.info(
                        f"  - 比 {algo_name} 的最后一张板少占用 {(metrics['last_rate'] - best_metrics['last_rate']) * 100:.2f}%")
                elif best_metrics['max_rate'] > metrics['max_rate']:
                    logger.info(
                        f"  - 比 {algo_name} 最高利用率高 {(best_metrics['max_rate'] - metrics['max_rate']) * 100:.2f}%")

        return best_results

    elif algorithm in ALGORITHMS:
        # 使用指定算法
        logger.info(f"使用算法: {algorithm}")
        logger.info(
            f"库存填充策略: {
                STOCK_ALGORITHMS.get(
                    stock_algorithm,
                    stock_algorithm)}")
        results, metrics = run_single_algorithm(
            plates, orders, others, optim, saw_blade, ALGORITHMS[algorithm], stock_algorithm)
        logger.info("完成切割:")
        logger.info(f"  - 使用板材: {metrics['used_plates']} 块")
        logger.info(f"  - 平均利用率: {metrics['overall_rate']:.2%}")
        logger.info(f"  - 平均切割数: {metrics['avg_cuts_per_plate']:.1f} 次/板")
        return results

    else:
        # 无效算法名称，使用默认算法
        logger.warning(f"未知算法 '{algorithm}'，使用默认算法 MaxRectsBssf")
        results, metrics = run_single_algorithm(
            plates, orders, others, optim, saw_blade, rectpack.MaxRectsBssf, stock_algorithm)
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
    results_auto = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="auto",
        stock_algorithm="maxrects_baf")
    print(f"   生成 {len(results_auto)} 个切割方案\n")

    # 2. 使用自动优化模式 + Guillotine BSSF + LLAS库存算法
    print("2. 自动优化模式 + Guillotine BSSF + LLAS库存算法:")
    results_guillotine = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="auto",
        stock_algorithm="guillotine_bssf_llas")
    print(f"   生成 {len(results_guillotine)} 个切割方案\n")

    # 3. 使用MaxRects BAF主算法 + MaxRects BAF库存算法
    print("3. MaxRects BAF主算法 + MaxRects BAF库存算法:")
    results_maxrects = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="MaxRectsBaf",
        stock_algorithm="maxrects_baf")
    print(f"   生成 {len(results_maxrects)} 个切割方案\n")

    # 4. 使用MaxRects BAF主算法 + Guillotine BSSF + LLAS库存算法
    print("4. MaxRects BAF主算法 + Guillotine BSSF + LLAS库存算法:")
    results_mixed = optimize_cutting(
        plates,
        orders,
        others,
        optim=1,
        algorithm="MaxRectsBaf",
        stock_algorithm="guillotine_bssf_llas")
    print(f"   生成 {len(results_mixed)} 个切割方案\n")

    # 显示库存算法说明
    print("=== 库存填充算法说明 ===")
    print("MaxRects BAF: 使用最大矩形算法，选择面积最小的可用区域")
    print("Guillotine BSSF + LLAS: 使用切割线算法，采用短边最佳适配和长剩余边分割策略")
