"""
布局整合优化（Layout Consolidation）

执行时机：全局择优完成后、返回最终结果前（余板已在 finalize_plate_output 中填入，
         整合时先抽取订单件重排，再对新板重新 finalize）。

目标：
  当前方案由若干"版型"组成：n1×A + n2×B + n3×C + …
  （同一版型的板零件组合完全相同，可视为同一套切割程序）

  整合优化尝试：
    1. 找出「独板版型」——只出现 1 次的排版，代表只需生产一张但程序独立。
    2. 将所有独板版型的订单件汇总，用相同数量的大板重新装箱。
    3. 若新结果的独立版型数减少（且总板数不增加），或总板数减少，则接受新方案。
    4. 重复最多 MAX_PASSES 轮，直至无法继续改进。

示例：
  整合前：5×A + 1×B + 1×C + 1×D  →  4 种版型
  整合后：5×A + 2×B              →  2 种版型（B 与 D 被合并到新的 B′）

注意：
  - 仅处理订单件（is_stock=0），余板（is_stock=1）不参与重排。
  - 整合使用 MaxRectsBaf 作为兜底 packer；互补/band_fill 等高级模式不在此路径运行
    （高级模式保留在初始多算法竞选阶段）。
  - 若整合后利用率显著下降可能意味着 packer 选择不佳；此时多轮结构确保只接受严格
    更优的结果。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.models import CuttingConfig, SmallPlate
from core.utils import DataConverter
from engine.optimizers import StockOptimizer
from engine.plate_packing_protocol import PlatePackingEngine
from engine.pipeline.output import finalize_plate_output, orders_from_result_orders_only
from engine.pipeline.templates import clone_plate_template

logger = logging.getLogger("plate_cutting")

MAX_PASSES = 3


# ---------------------------------------------------------------------------
# 版型指纹（Layout Fingerprint）
# ---------------------------------------------------------------------------

def board_fingerprint(result: Dict[str, Any]) -> Tuple:
    """
    计算一张板的版型指纹：提取订单件的规格，以旋转归一化（小边在前）后排序成元组。

    指纹相同 ↔ 两张板的订单件组合（规格 + 件数）完全一致，可视为同种切割程序。
    浮点精度：尺寸保留 1 位小数，与 DataConverter._round_dimension 保持一致。
    """
    order_cuts = [c for c in result["cutted"] if c[4] == 0]
    pieces = []
    for c in order_cuts:
        w = round(float(c[2]), 1)
        h = round(float(c[3]), 1)
        pieces.append((min(w, h), max(w, h)))
    return tuple(sorted(pieces))


def layout_groups(results: List[Dict[str, Any]]) -> Dict[Tuple, List[int]]:
    """
    将结果按版型分组。

    Returns:
        {版型指纹: [结果索引列表]}，按指纹出现频次降序排列。
    """
    groups: Dict[Tuple, List[int]] = {}
    for i, r in enumerate(results):
        fp = board_fingerprint(r)
        groups.setdefault(fp, []).append(i)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def layout_summary(results: List[Dict[str, Any]]) -> str:
    """
    生成可读的版型分布摘要，例如：「3 种版型：5×A + 2×B + 1×C（共 8 张）」。
    """
    groups = layout_groups(results)
    labels = [chr(65 + j) if j < 26 else f"T{j}" for j in range(len(groups))]
    parts = [
        f"{len(idxs)}×{label}"
        for label, (_, idxs) in zip(labels, groups.items())
    ]
    return f"{len(groups)} 种版型：{' + '.join(parts)}（共 {len(results)} 张）"


# ---------------------------------------------------------------------------
# 整合核心
# ---------------------------------------------------------------------------

def _single_consolidation_pass(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    plate_engine: PlatePackingEngine,
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[List[Dict[str, Any]]]:
    """
    单轮整合。

    策略：
      1. 找出独板版型（该版型只出现 1 次的所有板）。
      2. 将这些板的订单件汇总成一个 pool，用相同数量的大板重新装箱。
      3. 检查验收条件：
           - 独立版型数严格减少，且总板数不增加；或
           - 总板数严格减少（说明重排把多板合进了更少板）。
         不满足则废弃此次结果。

    Returns:
        整合后结果（改进时），否则 None。
    """
    groups = layout_groups(results)
    n_types_before = len(groups)
    n_boards_before = len(results)

    if n_types_before <= 1:
        return None  # 已完全一致，无需整合

    # 稀有版型：出现次数 <= 阈值
    # 动态阈值：如果有超过 10 张板，我们尝试把出现次数 <= min(3, 10%) 的版型也拿来重排
    # 如果板数少，就还是只针对独板 (len <= 1)
    max_freq = 1
    if n_boards_before > 10:
        max_freq = max(1, min(3, int(n_boards_before * 0.1)))

    singleton_idxs = sorted(
        i for idxs in groups.values() if len(idxs) <= max_freq for i in idxs
    )
    if not singleton_idxs:
        return None  # 无稀有版型，无需整合

    kept_idxs = [i for i in range(n_boards_before) if i not in set(singleton_idxs)]

    # 从稀有版型中抽取订单件
    pool: List[SmallPlate] = []
    for i in singleton_idxs:
        pool.extend(orders_from_result_orders_only(results[i]))
    if not pool:
        return None

    logger.info(
        "Consolidation: %d 个稀有版型（%d 张板 / %d 件）准备重排。",
        n_types_before - len(set(board_fingerprint(results[i]) for i in kept_idxs)),
        len(singleton_idxs),
        len(pool),
    )

    # 重排：使用与稀有版型数量相同的模板槽位（所有大板尺寸相同，取靠前的槽位即可）
    n_slots = len(singleton_idxs)
    repack_templates = [
        plate_templates[min(j, len(plate_templates) - 1)] for j in range(n_slots)
    ]
    remaining = list(pool)
    new_rows: List[Dict[str, Any]] = []

    for tpl in repack_templates:
        if not remaining:
            break
        bp = clone_plate_template(tpl)
        order_cuts, remaining = plate_engine.pack_orders(bp, remaining)
        if order_cuts:
            row = finalize_plate_output(
                bp, order_cuts, stock_plates, stock_optimizer, optim, config, converter
            )
            new_rows.append(row)

    if remaining:
        logger.info(
            "Consolidation: 重排后仍有 %d 件无法放置，本轮整合放弃。", len(remaining)
        )
        return None

    candidate = [results[i] for i in kept_idxs] + new_rows
    new_groups = layout_groups(candidate)
    n_types_after = len(new_groups)
    n_boards_after = len(candidate)

    # 验收：版型数严格减少（板数不增）OR 板数严格减少
    improved = (
        (n_types_after < n_types_before and n_boards_after <= n_boards_before)
        or n_boards_after < n_boards_before
    )
    if not improved:
        logger.info(
            "Consolidation: 本轮无改进（版型 %d→%d，板数 %d→%d），丢弃。",
            n_types_before, n_types_after,
            n_boards_before, n_boards_after,
        )
        return None

    logger.info(
        "Consolidation: 本轮通过（版型 %d→%d，板数 %d→%d）。",
        n_types_before, n_types_after,
        n_boards_before, n_boards_after,
    )
    return candidate


def consolidate_layout_groups(
    results: List[Dict[str, Any]],
    plate_templates: List[SmallPlate],
    plate_engine: PlatePackingEngine,
    stock_optimizer: StockOptimizer,
    stock_plates: List[SmallPlate],
    optim: int,
    config: CuttingConfig,
    converter: DataConverter,
) -> Optional[List[Dict[str, Any]]]:
    """
    多轮布局整合入口（最多 MAX_PASSES 轮）。

    每轮将当前结果中所有「独板版型」的订单件汇总重排，若产生改进则接受并
    继续下一轮；直至无更多改进或达到轮次上限。

    Args:
        results:         当前最优切割结果（已含余板 finalize）。
        plate_templates: 大板模板列表（与输入 plates 对应）。
        plate_engine:    PlateOptimizer 实例（用于重排独板件）。
        stock_optimizer: StockOptimizer 实例（用于重排后余板 finalize）。
        stock_plates:    余板列表。
        optim:           余板优化模式（0=标准，1=多排列试验）。
        config:          切割配置（锯片厚度等）。
        converter:       DataConverter。

    Returns:
        整合后结果（至少一轮有改进），若无改进则返回 None。
    """
    if len(results) <= 1:
        return None

    logger.info("Consolidation 开始：%s", layout_summary(results))

    current = results
    any_improved = False

    for pass_no in range(1, MAX_PASSES + 1):
        nxt = _single_consolidation_pass(
            current,
            plate_templates,
            plate_engine,
            stock_optimizer,
            stock_plates,
            optim,
            config,
            converter,
        )
        if nxt is None:
            break
        current = nxt
        any_improved = True
        logger.info("Consolidation 第 %d/%d 轮完成。", pass_no, MAX_PASSES)

    if not any_improved:
        logger.info("Consolidation: 无改进，保留原方案。")
        return None

    logger.info("Consolidation 完成：%s", layout_summary(current))
    return current
