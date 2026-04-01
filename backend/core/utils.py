import logging
from typing import Any, Dict, List

from core.metrics import calculate_cutting_metrics, compare_algorithms
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

