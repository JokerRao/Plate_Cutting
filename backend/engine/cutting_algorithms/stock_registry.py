"""
库存填充算法注册与启用控制（与主装箱解耦）。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Set

logger = logging.getLogger('plate_cutting')

# API / run_single_algorithm 使用的库存策略 id → 展示名
STOCK_ALGORITHM_LABELS: Dict[str, str] = {
    "maxrects_baf": "MaxRects BAF",
    "guillotine_bssf_llas": "Guillotine BSSF+LLAS",
}

DEFAULT_STOCK_ID = "maxrects_baf"


def normalize_enabled_stock_ids(config_value: Optional[str]) -> Optional[Set[str]]:
    """
    返回允许使用的库存算法 id 集合；None 表示不限制（全部允许）。
    """
    if not config_value or not str(config_value).strip():
        return None
    ids = {x.strip() for x in str(config_value).split(",") if x.strip()}
    return ids if ids else None


def resolve_stock_algorithm(
        requested: str,
        enabled: Optional[Set[str]],
) -> str:
    """
    若配置了 STOCK_ALGORITHMS_ENABLED 且 requested 不在集合内，回退到集合中字典序第一个。
    """
    if enabled is None or requested in enabled:
        return requested
    if not enabled:
        return requested
    fallback = sorted(enabled)[0]
    logger.warning(
        "库存算法 %s 未启用，改用 %s（请检查 STOCK_ALGORITHMS_ENABLED）",
        requested,
        fallback,
    )
    return fallback
