import logging
from typing import Dict, List, Optional, Tuple

import rectpack

from core.models import Cut, CuttingConfig, Rectangle, SmallPlate
from engine.complementary_pairs import (
    band_fill_details_if_applicable,
    find_complementary_pairs,
)
from engine.rectpack_trials import RectpackTrialRunner
from engine.row_layout import (
    pack_orders_row_based as pack_row_based_layout,
    pack_orders_layer_based as pack_layer_based_layout,
    pack_orders_composite_stack as pack_composite_stack_layout,
    pack_orders_band_fill as pack_band_fill_layout,
)
from engine.packers import (
    BaseStockPacker,
    GuillotineBssfLlasPacker,
    MaxRectsBafPacker,
)

logger = logging.getLogger('plate_cutting')

# ============================================================================
# 优化器类
# ============================================================================

class PlateOptimizer:
    """板材优化器"""

    def __init__(
            self,
            config: CuttingConfig,
            algorithm: rectpack = rectpack.GuillotineBssfMaxas):
        self.config = config
        self.algorithm = algorithm
        self._trials = RectpackTrialRunner(self)

    def create_packer(self, width: int, height: int) -> rectpack.packer:
        """创建 rectpack 装箱器"""

        packer = rectpack.newPacker(
            mode=rectpack.PackingMode.Offline,
            bin_algo=rectpack.PackingBin.Global,
            pack_algo=self.algorithm,
            sort_algo=rectpack.SORT_PERI,
            rotation=True  # 允许旋转
        )

        # 添加容器（大板）
        packer.add_bin(width, height)

        return packer

    def _select_best_pattern(
        self,
        complementary: Dict,
        pattern_details: Dict,
        length0: float,
        width0: float,
    ) -> Tuple[Tuple, Dict]:
        """
        从所有候选互补对中选出利用率最高的模式，并尝试将 composite 升级为 band_fill。

        Returns:
            (best_pair, details)
        """
        bt = self.config.blade_thickness

        composite_pairs = {
            k: v for k, v in pattern_details.items()
            if v.get('type') == 'composite' and k in complementary
        }
        if composite_pairs:
            best_pair = max(composite_pairs, key=lambda k: composite_pairs[k].get('util', 0))
        else:
            best_pair = max(complementary, key=complementary.get)

        details = pattern_details[best_pair]

        if not composite_pairs:
            return best_pair, details

        # 扫描所有 composite 对，取 band-fill 可行且 util 最优者
        bf_best: Optional[Dict] = None
        bf_pair_key = None
        for k, v in composite_pairs.items():
            if v.get('type') != 'composite':
                continue
            bf = band_fill_details_if_applicable(k, int(length0), int(width0), bt)
            if bf is None:
                continue
            if bf_best is None or bf['util'] > bf_best['util'] + 1e-12:
                bf_best = bf
                bf_pair_key = k

        comp_util = details.get('util', 0)
        if bf_best is not None and bf_pair_key is not None and bf_best['util'] > comp_util + 1e-9:
            logger.info(
                "Selected band-fill over top composite (%.4f > %.4f util)",
                bf_best['util'],
                comp_util,
            )
            return bf_pair_key, bf_best

        # If the top composite pair itself qualifies for band-fill with equal or better util, upgrade it
        if details.get('type') == 'composite':
            bf = band_fill_details_if_applicable(best_pair, int(length0), int(width0), bt)
            if bf is not None and bf.get('util', 0) >= comp_util - 1e-9:
                logger.info("Upgraded composite pair to band-fill")
                return best_pair, bf

        return best_pair, details

    def _sort_orders_for_optimal_packing(self,
                                         orders: List[SmallPlate],
                                         big_plate: SmallPlate) -> List[Tuple[int,
                                                                              SmallPlate,
                                                                              bool]]:
        """
        对订单进行排序以优化装箱利用率，优先考虑混合组合
        策略：使用互补尺寸检测，识别混合组合优于单一尺寸的情况

        Returns:
            排序后的 (原始索引, 订单, 是否旋转) 元组列表
        """
        bt = self.config.blade_thickness
        length0 = big_plate.length
        width0 = big_plate.width

        # 按尺寸分组板材（考虑旋转后的最佳尺寸）
        def get_optimal_size(order: SmallPlate) -> Tuple[float, float, bool]:
            """获取板材的最佳尺寸和旋转状态"""
            x1 = order.length + bt
            x2 = order.width + bt

            # 计算两种方向的适配度
            fit1 = (length0 // x1) * (width0 //
                                      x2) if x1 <= length0 and x2 <= width0 else 0
            fit2 = (length0 // x2) * (width0 //
                                      x1) if x2 <= length0 and x1 <= width0 else 0

            # 决定是否旋转
            cond1 = (x1 < x2 and (length0 // x2) * (width0 // x1)
                     >= (length0 // x1) * (width0 // x2))
            cond2 = (abs(x1 - x2) / max(x1, x2) < 0.56 and (length0 // x2)
                     * (width0 // x1) > (length0 // x1) * (width0 // x2))
            should_rotate = cond1 or cond2

            # 使用最佳方向的尺寸
            if fit2 > fit1 or (fit2 == fit1 and should_rotate):
                return (x2, x1, True)
            else:
                return (x1, x2, False)

        # 为每个订单计算最佳尺寸和旋转状态
        orders_with_info = []
        for i, order in enumerate(orders):
            w, h, rotate = get_optimal_size(order)
            orders_with_info.append({
                'index': i,
                'order': order,
                'width': w,
                'height': h,
                'rotate': rotate,
                'area': order.length * order.width,
                'fit_count': (length0 // w) * (width0 // h) if w <= length0 and h <= width0 else 0
            })

        # 按尺寸分组（使用宽度和高度作为键，考虑浮点误差）
        def size_key(w: float, h: float) -> Tuple[int, int]:
            """将尺寸转换为整数键，用于分组（考虑1mm的误差）"""
            return (int(round(w)), int(round(h)))

        size_groups = {}
        for info in orders_with_info:
            key = size_key(info['width'], info['height'])
            if key not in size_groups:
                size_groups[key] = []
            size_groups[key].append(info)

        # 使用互补尺寸检测（混合组合优化）
        complementary, pattern_details = find_complementary_pairs(
            size_groups, length0, width0)

        # 多种尺寸并存时，互补/复合列/band-fill 只针对「主尺寸对」，会把其余规格挤到队尾，
        # 逐张板重复该策略会导致极差结果（例如 7 种零件时板数接近翻倍）。仅少量规格组时才启用。
        _max_groups_for_complementary = 3

        # 如果找到互补尺寸对，优先使用交错排列
        if complementary and len(size_groups) <= _max_groups_for_complementary:
            best_pair, details = self._select_best_pattern(
                complementary, pattern_details, length0, width0)
            w1, h1, w2, h2 = best_pair
            group1 = size_groups.get((w1, h1), [])
            group2 = size_groups.get((w2, h2), [])

            # 存储模式详情供pack_orders使用
            self._detected_pattern = {
                'pair': best_pair,
                'details': details
            }

            logger.info(
                "Using complementary pair strategy: (%sx%s, %sx%s) with %.2f%% gain",
                w1, h1, w2, h2, complementary[best_pair] * 100,
            )

            # 交错排列互补尺寸
            result = []
            max_len = max(len(group1), len(group2))
            for i in range(max_len):
                if i < len(group1):
                    result.append(group1[i])
                if i < len(group2):
                    result.append(group2[i])

            # 添加其他尺寸的板材 — 但对于 composite 模式，将其他 small 类型放在 group2 之前，
            # 以便它们有机会和剩余的 group1 (big) 配合使用
            other_keys = [k for k in size_groups.keys() if k not in [(w1, h1), (w2, h2)]]
            other_groups = [size_groups[k] for k in other_keys]
            if details.get('type') in ('composite', 'band_fill') and other_groups:
                for og in other_groups:
                    result.extend(og)
            else:
                for key, group in size_groups.items():
                    if key not in [(w1, h1), (w2, h2)]:
                        result.extend(group)

            # 转换为结果格式
            return [(info['index'], info['order'], info['rotate'])
                    for info in result]

        # 清除模式详情
        self._detected_pattern = None

        # 如果没有找到互补尺寸对，使用原有的多策略评估方法
        logger.info(
            "No significant complementary pairs found, using multi-strategy evaluation")

        # 计算每种尺寸的适配度和组合潜力
        size_stats = {}
        for key, group in size_groups.items():
            w, h = key
            fit_count = (length0 // w) * (width0 //
                                          h) if w <= length0 and h <= width0 else 0
            total_count = len(group)
            # 组合潜力：如果单一尺寸能放满整张板，组合潜力较低
            # 如果单一尺寸放不满，组合潜力较高
            utilization_if_single = min(
                total_count, fit_count) * (w * h) / (length0 * width0)
            combination_potential = 1.0 - min(utilization_if_single, 1.0)

            size_stats[key] = {
                'fit_count': fit_count,
                'total_count': total_count,
                'combination_potential': combination_potential,
                'avg_area': sum(item['area'] for item in group) / len(group)
            }

        # 排序策略：优先考虑组合潜力高的尺寸，然后采用轮询方式混合排列
        def calculate_sort_priority(info: dict) -> Tuple[float, float, float]:
            """计算排序优先级"""
            key = size_key(info['width'], info['height'])
            stats = size_stats[key]

            # 优先级1：组合潜力（越高越优先，负数表示降序）
            # 组合潜力高的尺寸应该优先放置，以便与其他尺寸形成组合
            priority1 = -stats['combination_potential']

            # 优先级2：适配难度（能放的数量越少，难度越高，优先放置）
            # 难以适配的板材应该优先放置
            difficulty = - \
                stats['fit_count'] if stats['fit_count'] > 0 else -999999
            priority2 = difficulty

            # 优先级3：面积（越大越优先）
            priority3 = -info['area']

            return (priority1, priority2, priority3)

        # 尝试多种排列策略，选择最优的
        def evaluate_arrangement(arrangement: List[dict]) -> float:
            """
            评估排列的潜在利用率
            综合考虑：面积利用率、尺寸多样性、组合潜力
            """
            if not arrangement:
                return 0.0

            # 1. 计算已使用的面积
            total_area = sum(info['width'] * info['height']
                             for info in arrangement)
            used_ratio = min(total_area / (length0 * width0), 1.0)

            # 2. 计算尺寸多样性（不同尺寸的数量）
            unique_sizes = len(
                set(size_key(info['width'], info['height']) for info in arrangement))
            diversity_score = unique_sizes / \
                len(arrangement) if arrangement else 0

            # 3. 评估组合潜力：检查不同尺寸是否能形成更好的组合
            # 计算每种尺寸的占比
            size_counts = {}
            for info in arrangement:
                key = size_key(info['width'], info['height'])
                size_counts[key] = size_counts.get(key, 0) + 1

            # 如果单一尺寸占比过高，组合潜力较低
            max_size_ratio = max(size_counts.values()) / \
                len(arrangement) if arrangement else 0
            balance_score = 1.0 - max_size_ratio  # 越平衡，得分越高

            # 4. 评估适配度：检查板材是否能更好地适配大板
            avg_fit_score = sum(
                size_stats[size_key(info['width'], info['height'])]['fit_count']
                for info in arrangement
            ) / len(arrangement) if arrangement else 0
            normalized_fit = min(avg_fit_score / 10.0, 1.0)  # 归一化到0-1

            # 综合评分：多样性、平衡性、适配度、面积利用率
            # 权重：多样性20%，平衡性30%，适配度20%，面积利用率30%
            final_score = (
                diversity_score * 0.2 +
                balance_score * 0.3 +
                normalized_fit * 0.2 +
                used_ratio * 0.3
            )

            return final_score

        def generate_round_robin_arrangement() -> List[dict]:
            """生成轮询排列"""
            grouped_by_size = {}
            for info in orders_with_info:
                key = size_key(info['width'], info['height'])
                if key not in grouped_by_size:
                    grouped_by_size[key] = []
                grouped_by_size[key].append(info)

            result = []
            max_group_size = max(
                len(group) for group in grouped_by_size.values()) if grouped_by_size else 0

            for round_idx in range(max_group_size):
                sorted_groups = sorted(
                    grouped_by_size.items(),
                    key=lambda x: size_stats[x[0]]['combination_potential'],
                    reverse=True
                )

                for key, group in sorted_groups:
                    if round_idx < len(group):
                        result.append(group[round_idx])

            return result

        def generate_greedy_arrangement() -> List[dict]:
            """生成贪心排列：每次选择能形成最好组合的下一个板材"""
            remaining = orders_with_info.copy()
            result = []

            # 按组合潜力排序尺寸组
            grouped_by_size = {}
            for info in orders_with_info:
                key = size_key(info['width'], info['height'])
                if key not in grouped_by_size:
                    grouped_by_size[key] = []
                grouped_by_size[key].append(info)

            # 按组合潜力排序尺寸组
            sorted_size_keys = sorted(
                grouped_by_size.keys(),
                key=lambda k: size_stats[k]['combination_potential'],
                reverse=True
            )

            # 贪心选择：优先从组合潜力高的尺寸组中选择
            size_group_indices = {key: 0 for key in sorted_size_keys}

            while remaining:
                best_info = None
                best_score = -1
                best_key = None

                # 从每个尺寸组中选择一个候选
                for key in sorted_size_keys:
                    group = grouped_by_size[key]
                    idx = size_group_indices[key]
                    if idx < len(group):
                        candidate = group[idx]
                        if candidate in remaining:
                            # 评估这个候选与已选板材的组合效果
                            temp_result = result + [candidate]
                            score = evaluate_arrangement(temp_result)

                            if score > best_score:
                                best_score = score
                                best_info = candidate
                                best_key = key

                if best_info:
                    result.append(best_info)
                    remaining.remove(best_info)
                    size_group_indices[best_key] += 1
                else:
                    # 如果没有找到，直接添加剩余的
                    result.extend(remaining)
                    break

            return result

        def generate_balanced_arrangement() -> List[dict]:
            """生成平衡排列：确保不同尺寸均匀分布"""
            grouped_by_size = {}
            for info in orders_with_info:
                key = size_key(info['width'], info['height'])
                if key not in grouped_by_size:
                    grouped_by_size[key] = []
                grouped_by_size[key].append(info)

            # 按组合潜力排序尺寸组
            sorted_groups = sorted(
                grouped_by_size.items(),
                key=lambda x: size_stats[x[0]]['combination_potential'],
                reverse=True
            )

            result = []

            # 计算每个尺寸组的权重（基于组合潜力）
            total_potential = sum(
                size_stats[key]['combination_potential'] for key,
                _ in sorted_groups)
            group_weights = {
                key: size_stats[key]['combination_potential'] /
                total_potential if total_potential > 0 else 1.0 /
                len(sorted_groups) for key,
                _ in sorted_groups}

            # 根据权重分配每个尺寸组的数量
            total_items = len(orders_with_info)
            group_targets = {
                key: max(1, int(weight * total_items))
                for key, weight in group_weights.items()
            }

            max_count = max(len(group) for _, group in sorted_groups)
            added = set()
            for round_idx in range(max_count):
                for key, group in sorted_groups:
                    if round_idx < len(group):
                        if round_idx < group_targets.get(key, len(group)):
                            result.append(group[round_idx])
                            added.add(id(group[round_idx]))

            for key, group in sorted_groups:
                for item in group:
                    if id(item) not in added:
                        result.append(item)

            return result

        # 尝试多种策略并选择最好的
        strategies = [
            ("round_robin", generate_round_robin_arrangement),
            ("greedy", generate_greedy_arrangement),
            ("balanced", generate_balanced_arrangement),
        ]

        best_arrangement = None
        best_score = -1
        best_strategy_name = None

        for strategy_name, strategy_func in strategies:
            try:
                arrangement = strategy_func()
                score = evaluate_arrangement(arrangement)
                if score > best_score:
                    best_score = score
                    best_arrangement = arrangement
                    best_strategy_name = strategy_name
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue

        # 如果所有策略都失败，使用简单的优先级排序
        if best_arrangement is None:
            orders_with_info.sort(key=calculate_sort_priority)
            best_arrangement = orders_with_info

        logger.info(
            f"Selected arrangement strategy: {best_strategy_name} with score: {
                best_score:.4f}")

        # 转换为结果格式
        result = [(info['index'], info['order'], info['rotate'])
                  for info in best_arrangement]

        return result

    def _pick_rotation(self, order: SmallPlate, big_plate: SmallPlate) -> bool:
        """与 _sort_orders_for_optimal_packing 中 get_optimal_size 一致的旋转判定。"""
        bt = self.config.blade_thickness
        length0 = big_plate.length
        width0 = big_plate.width
        x1 = order.length + bt
        x2 = order.width + bt
        fit1 = (length0 // x1) * (width0 //
                                  x2) if x1 <= length0 and x2 <= width0 else 0
        fit2 = (length0 // x2) * (width0 //
                                  x1) if x2 <= length0 and x1 <= width0 else 0
        cond1 = (x1 < x2 and (length0 // x2) * (width0 // x1)
                 >= (length0 // x1) * (width0 // x2))
        cond2 = (abs(x1 - x2) / max(x1, x2) < 0.56 and (length0 // x2)
                 * (width0 // x1) > (length0 // x1) * (width0 // x2))
        should_rotate = cond1 or cond2
        if fit2 > fit1 or (fit2 == fit1 and should_rotate):
            return True
        return False

    def _indices_to_sorted_tuples(
            self,
            orders: List[SmallPlate],
            indices: List[int],
            big_plate: SmallPlate) -> List[Tuple[int, SmallPlate, bool]]:
        return [
            (i, orders[i], self._pick_rotation(orders[i], big_plate))
            for i in indices
        ]

    def pack_orders(self,
                    big_plate: SmallPlate,
                    orders: List[SmallPlate]) -> Tuple[List[Cut],
                                                       List[SmallPlate]]:
        """使用 rectpack 装箱订单板材"""

        # 对订单进行排序以优化装箱利用率（并设置互补/行式检测状态）
        primary_sorted = self._sort_orders_for_optimal_packing(
            orders, big_plate)
        # 互补/行式排序后的零件序列（必须与 _detected_pattern 的穿插策略一致）
        layout_orders = [t[1] for t in primary_sorted]

        # 检查是否检测到行式布局模式
        if (
            self.config.enable_row_complementary
            and hasattr(self, '_detected_pattern')
            and self._detected_pattern
        ):
            pattern = self._detected_pattern
            if pattern['details']['type'] == 'row':
                logger.info("Using custom row-based packing")
                w1, h1, w2, h2 = pattern['pair']
                return pack_row_based_layout(
                    self.config,
                    big_plate,
                    layout_orders,
                    (w1, h1),
                    (w2, h2),
                    pattern['details']['count1'],
                    pattern['details']['count2'],
                )
            elif pattern['details']['type'] == 'layer':
                logger.info("Using custom layer-based packing")
                w1, h1, w2, h2 = pattern['pair']
                return pack_layer_based_layout(
                    self.config,
                    big_plate,
                    layout_orders,
                    (w1, h1),
                    (w2, h2),
                    pattern['details']['r1'],
                    pattern['details']['r2'],
                    pattern['details']['c1'],
                    pattern['details']['c2'],
                )
            elif pattern['details']['type'] == 'composite':
                logger.info("Using custom composite-stack packing")
                return pack_composite_stack_layout(
                    big_plate,
                    layout_orders,
                    pattern['details'],
                    self.config.blade_thickness,
                )
            elif pattern['details']['type'] == 'band_fill':
                d = pattern['details']
                logger.info(
                    "Using band-fill packing (%d big + %d bottom strips + %d mid strips)",
                    d.get("n_big", 2), d.get("n_bottom", 3), d.get("n_mid", 1),
                )
                cuts, rem = pack_band_fill_layout(
                    big_plate,
                    layout_orders,
                    pattern['details'],
                    self.config.blade_thickness,
                )
                if not cuts:
                    return self._trials.best_rectpack_for_bin(
                        orders, big_plate, primary_sorted)
                return cuts, rem

        return self._trials.best_rectpack_for_bin(
            orders, big_plate, primary_sorted)

    def _find_matching_order_index(
            self,
            orders: List[SmallPlate],
            w: int,
            h: int,
            used_indices: set) -> Optional[int]:
        """查找匹配的订单索引（当rectpack没有返回rid时使用）"""
        for i, order in enumerate(orders):
            if i in used_indices:
                continue
            # 检查是否匹配（考虑旋转）
            if (w == order.length and h == order.width) or (
                    w == order.width and h == order.length):
                return i
        return None


class StockOptimizer:
    """库存板材优化器 - 支持 MaxRects BAF 和 Guillotine BSSF + LLAS 算法"""

    def __init__(self, config: CuttingConfig, algorithm: str = "maxrects_baf"):
        """
        Args:
            config: 切割配置
            algorithm: 算法选择
                - "maxrects_baf": MaxRects Best Area Fit（默认）
                - "guillotine_bssf_llas": Guillotine BSSF + Long Leftover Axis Split
        """
        self.config = config
        self.algorithm = algorithm.lower()

    def _create_packer(self, width: int, height: int) -> BaseStockPacker:
        """根据算法选择创建相应的装箱器"""
        if self.algorithm == "guillotine_bssf_llas":
            return GuillotineBssfLlasPacker(width, height, self.config)
        else:  # 默认使用 maxrects_baf
            return MaxRectsBafPacker(width, height, self.config)

    def fill_with_stock(
            self,
            width: int,
            height: int,
            existing_cuts: List[Cut],
            stock_plates: List[SmallPlate],
            optimize: bool = False) -> List[Cut]:
        """用库存板材填充剩余空间"""
        if not stock_plates:
            return []

        def _try_stock_arrangement(
                sorted_stock: List[SmallPlate]) -> Tuple[List[Cut], float]:
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
            elif isinstance(packer, GuillotineBssfLlasPacker):
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
                    if packer.fitness(
                            stock.length + self.config.blade_thickness,
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
                    cut.plate.plate_id = f"STOCK_{i + 1}"

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
            arrangements.append(
                ("面积降序",
                 sorted(
                     stock_plates,
                     key=lambda p: p.area,
                     reverse=True)))

            # 2. 按面积从小到大排序
            arrangements.append(
                ("面积升序", sorted(stock_plates, key=lambda p: p.area)))

            # 3. 按长度降序排列
            arrangements.append(
                ("长度降序",
                 sorted(
                     stock_plates,
                     key=lambda p: p.length,
                     reverse=True)))

            # 4. 按宽度降序排列
            arrangements.append(
                ("宽度降序",
                 sorted(
                     stock_plates,
                     key=lambda p: p.width,
                     reverse=True)))

            # 5. 按周长降序排列
            arrangements.append(("周长降序", sorted(
                stock_plates, key=lambda p: 2 * (p.length + p.width), reverse=True)))

            # 6. 原始顺序
            arrangements.append(("原始顺序", stock_plates.copy()))

            # 7. 尝试从不同位置开始的循环排列（限制数量避免过度计算）
            if len(stock_plates) <= 10:  # 只对较小的库存列表尝试循环排列
                for i in range(1, min(len(stock_plates), 6)):  # 最多尝试前3个位置开始
                    rotated = stock_plates[i:] + stock_plates[:i]
                    arrangements.append((f"从第{i + 1}个开始", rotated))

            # 8. 如果库存板种类较少，尝试优化的混合策略
            if len(set((p.length, p.width) for p in stock_plates)) <= 6:
                # 大板优先，小板填充策略
                large_plates = [
                    p for p in stock_plates if p.area >= 1.2 * 10**5]
                small_plates = [
                    p for p in stock_plates if p.area < 1.2 * 10**5]
                large_first = sorted(large_plates,
                                     key=lambda p: p.area,
                                     reverse=True) + sorted(small_plates,
                                                            key=lambda p: p.area)
                arrangements.append(("大板优先", large_first))

                # 长条优先策略
                long_plates = [p for p in stock_plates if max(
                    p.length, p.width) / min(p.length, p.width) >= 2]
                square_plates = [p for p in stock_plates if max(
                    p.length, p.width) / min(p.length, p.width) < 2]
                long_first = sorted(long_plates,
                                    key=lambda p: p.area,
                                    reverse=True) + sorted(square_plates,
                                                           key=lambda p: p.area,
                                                           reverse=True)
                arrangements.append(("长条优先", long_first))

            # 测试每种排列，同时记录"原始顺序"结果以备对比（避免事后重跑）
            default_utilization = 0.0
            for arrangement_name, sorted_stock in arrangements:
                logger.debug(f"测试排列: {arrangement_name}")

                try:
                    cuts, utilization = _try_stock_arrangement(sorted_stock)

                    logger.debug(
                        f"  {arrangement_name} - 利用率: {utilization:.3%}, 切割数: {len(cuts)}")

                    if arrangement_name == "原始顺序":
                        default_utilization = utilization

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
            algorithm_name = "Guillotine BSSF + LLAS" if self.algorithm == "guillotine_bssf_llas" else "MaxRects BAF"
            logger.info(f"库存填充完成（{algorithm_name}）")
            logger.info(f"最优排列: {best_arrangement}")
            logger.info(f"总利用率: {best_utilization:.2%}")
            logger.info(f"放置了 {len(best_cuts)} 块库存板材")

            improvement = best_utilization - default_utilization
            if improvement > 0.001:
                logger.info(f"相比默认排列提升利用率: +{improvement:.2%}")

            return best_cuts

        else:
            # 非优化模式：使用原始顺序
            logger.debug("使用原始顺序填充库存")
            cuts, utilization = _try_stock_arrangement(stock_plates)

            algorithm_name = "Guillotine BSSF + LLAS" if self.algorithm == "guillotine_bssf_llas" else "MaxRects BAF"
            logger.debug(
                f"库存填充完成（{algorithm_name}），利用率: {
                    utilization:.2%}，放置了{
                    len(cuts)}块库存板材")

            return cuts

    def _update_guillotine_sections(
            self,
            packer: GuillotineBssfLlasPacker,
            existing_cuts: List[Cut]):
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
            free_sections = self._compute_free_sections(
                initial_section, occupied_rects)
            for section in free_sections:
                packer._add_section(section)

    def _compute_free_sections(
            self,
            container: Rectangle,
            occupied: List[Rectangle]) -> List[Rectangle]:
        """计算除去已占用区域后的空闲区域（经过水平预合并优化）"""
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
        for j in range(len(y_coords) - 1):
            y = y_coords[j]
            h = y_coords[j + 1] - y
            if h <= 0:
                continue

            # 筛选在当前高度范围内可能相交的占用区域
            active_occupied = [
                occ for occ in occupied
                if occ.y < y + h and occ.top > y
            ]

            row_sections = []
            for i in range(len(x_coords) - 1):
                x = x_coords[i]
                w = x_coords[i + 1] - x
                if w <= 0:
                    continue

                is_free = True
                for occ in active_occupied:
                    if occ.x < x + w and occ.right > x:
                        is_free = False
                        break

                if is_free:
                    if row_sections and row_sections[-1].right == x:
                        # 立即水平合并
                        row_sections[-1].width += w
                    else:
                        row_sections.append(Rectangle(x, y, w, h))
            
            free_sections.extend(row_sections)

        # 跨行合并相邻的空闲区域
        merged = self._merge_adjacent_sections(free_sections)
        return merged

    def _merge_adjacent_sections(
            self, sections: List[Rectangle]) -> List[Rectangle]:
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


