import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger('plate_cutting')

# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class CuttingConfig:
    """切割配置参数"""
    blade_thickness: float = 4.0  # 锯片厚度


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
    x1: float
    y1: float
    x2: float
    y2: float
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


