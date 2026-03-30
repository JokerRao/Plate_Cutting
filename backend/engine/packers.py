import collections
import itertools
import logging
import operator
from typing import List, Optional, Tuple

from core.models import CuttingConfig, Cut, Rectangle, SmallPlate

logger = logging.getLogger('plate_cutting')

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

    def _rect_fitness_baf(
            self,
            max_rect: Rectangle,
            width: int,
            height: int) -> Optional[float]:
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

    def _select_position(self,
                         w: int,
                         h: int) -> Tuple[Optional[Rectangle],
                                          Optional[Rectangle]]:
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
            new_rects.append(
                Rectangle(
                    m.left,
                    m.bottom,
                    r.left - m.left,
                    m.height))

        # 右侧剩余
        if r.right < m.right:
            new_rects.append(
                Rectangle(
                    r.right,
                    m.bottom,
                    m.right -
                    r.right,
                    m.height))

        # 上方剩余
        if r.top < m.top:
            new_rects.append(Rectangle(m.left, r.top, m.width, m.top - r.top))

        # 下方剩余
        if r.bottom > m.bottom:
            new_rects.append(
                Rectangle(
                    m.left,
                    m.bottom,
                    m.width,
                    r.bottom -
                    m.bottom))

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


class GuillotineBssfLlasPacker(BaseStockPacker):
    """自定义Guillotine BSSF + LLAS算法实现"""

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

    def _section_fitness_bssf(
            self,
            section: Rectangle,
            width: int,
            height: int) -> Optional[float]:
        """Best Short Side Fit (BSSF) 适配度计算
        选择短边剩余空间最小的区域，最小化短边浪费
        """
        if width > section.width or height > section.height:
            return None
        # 返回短边的剩余空间（越小越好）
        return min(section.width - width, section.height - height)

    def _split_llas(self, section: Rectangle, width: int, height: int):
        """
        Long Leftover Axis Split (LLAS) 分割策略
        选择产生较长剩余边的分割方式
        """
        # 计算两个方向的剩余长度
        leftover_horizontal = section.width - width    # 水平分割后的剩余宽度
        leftover_vertical = section.height - height    # 垂直分割后的剩余高度

        # 选择剩余边较长的分割方式
        if leftover_horizontal >= leftover_vertical:
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

    def _select_best_section(
            self, w: int, h: int) -> Tuple[Optional[Rectangle], bool]:
        """选择最佳放置区域"""
        best_fitness = None
        best_section = None
        rotated = False

        # 尝试正常方向
        for section in self._sections:
            fitness = self._section_fitness_bssf(section, w, h)
            if fitness is not None:
                if best_fitness is None or fitness < best_fitness:
                    best_fitness = fitness
                    best_section = section
                    rotated = False

        # 尝试旋转方向
        if self.rot:
            for section in self._sections:
                fitness = self._section_fitness_bssf(section, h, w)
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
        section, rotated = self._select_best_section(
            needed_width, needed_height)
        if not section:
            return False

        if rotated:
            needed_width, needed_height = needed_height, needed_width

        # 移除选中的区域
        self._sections.remove(section)

        # 执行LLAS分割
        self._split_llas(section, needed_width, needed_height)

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
        rect = Rectangle(
            section.x,
            section.y,
            needed_width,
            needed_height,
            plate.plate_id)
        self.rectangles.append(rect)

        return True

    def fitness(self, width: int, height: int) -> Optional[float]:
        """计算给定尺寸矩形的适配度"""
        section, rotated = self._select_best_section(width, height)
        if not section:
            return None

        if rotated:
            return self._section_fitness_bssf(section, height, width)
        else:
            return self._section_fitness_bssf(section, width, height)


