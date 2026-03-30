import logging
from typing import Any, Dict, List

from core.models import SmallPlate, Cut

logger = logging.getLogger('plate_cutting')

# ============================================================================
# 数据转换器
# ============================================================================

class DataConverter:
    """数据转换器"""

    def _round_dimension(self, value: float, precision: int = 1) -> float:
        """对尺寸进行统一的四舍五入，并消除极小的浮点误差"""
        rounded = round(value, precision)
        return 0.0 if abs(rounded) < 10 ** (-precision) else rounded

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

    def convert_cuts_to_output(
            self, big_plate: SmallPlate, cuts: List[Cut]) -> Dict[str, Any]:
        """转换切割结果为输出格式"""
        cuts_data = []
        for cut in cuts:
            plate_id = cut.plate.plate_id
            cuts_data.append([
                self._round_dimension(cut.x1),
                self._round_dimension(cut.y1),
                self._round_dimension(cut.x2 - cut.x1),
                self._round_dimension(cut.y2 - cut.y1),
                1 if cut.is_stock else 0,  # is_stock
                plate_id  # id
            ])

        # 计算利用率
        used_area = sum((cut[2] * cut[3]) for cut in cuts_data)
        total_area = self._round_dimension(
            big_plate.length) * self._round_dimension(big_plate.width)
        utilization_rate = used_area / total_area if total_area > 0 else 0

        return {
            'rate': utilization_rate,
            'plate': [
                self._round_dimension(big_plate.length),
                self._round_dimension(big_plate.width)
            ],
            'cutted': cuts_data
        }


# ============================================================================
# 工具函数
# ============================================================================

def calculate_cutting_metrics(
        results: List[Dict[str, Any]], remaining_orders: int) -> Dict[str, Any]:
    """
    计算切割方案的详细指标

    Returns:
        包含多个评价指标的字典
    """
    if not results:
        return {
            'used_plates': 0,
            'overall_rate': 0,
            'last_rate': 0,
            'avg_rate': 0,
            'min_rate': 0,
            'max_rate': 0,
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

    # 最后一张板的使用率
    last_rate = rates[-1]

    # 利用率分布统计
    min_rate = min(rates) if rates else 0
    max_rate = max(rates) if rates else 0

    # 利用率方差（衡量利用率的均匀程度）
    avg_rate = overall_rate
    rate_variance = sum((r - avg_rate) ** 2 for r in rates) / \
        used_plates if used_plates > 0 else 0

    # 切割复杂度统计
    total_cuts = sum(len(r['cutted']) for r in results)
    avg_cuts_per_plate = total_cuts / used_plates if used_plates > 0 else 0

    # 最大单板切割数（切割复杂度）
    max_cuts_single_plate = max(len(r['cutted'])
                                for r in results) if results else 0

    # 订单完成度
    total_order_cuts = sum(
        1 for r in results
        for cut in r['cutted']
        if cut[4] == 0  # is_stock == 0 表示订单板材
    )

    return {
        'used_plates': used_plates,
        'overall_rate': overall_rate,
        'last_rate': last_rate,
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


def compare_algorithms(
        metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> int:
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

    # 2. 最后一张板的利用率（越低越好）
    cut_diff = abs(metrics1['last_rate'] - metrics2['last_rate'])
    if cut_diff > 0.001:  # 利用率差异大于0.1%
        return -1 if metrics1['last_rate'] < metrics2['last_rate'] else 1

    # 3. 利用率相近，比较最大利用率，越大越好
    max_rate_diff = abs(metrics1['max_rate'] - metrics2['max_rate'])
    if max_rate_diff > 0.01:  # 差异大于1%
        return -1 if metrics1['max_rate'] > metrics2['max_rate'] else 1

    # # 4. 比较利用率方差（越小越好，表示各板材利用率更均匀）
    # variance_diff = abs(metrics1['rate_variance'] - metrics2['rate_variance'])
    # if variance_diff > 0.0001:
    # return -1 if metrics1['rate_variance'] < metrics2['rate_variance'] else
    # 1

    # # 5. 比较最大单板切割数（越大越好，能放更多板）
    # if metrics1['max_cuts_single_plate'] != metrics2['max_cuts_single_plate']:
    # return -1 if metrics1['max_cuts_single_plate'] >
    # metrics2['max_cuts_single_plate'] else 1

    # 7. 如果所有指标都相同，返回相等
    return 0


