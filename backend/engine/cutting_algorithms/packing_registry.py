"""
主装箱（rectpack）算法注册表：每种算法独立注册，可通过配置启用子集。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import rectpack

logger = logging.getLogger('plate_cutting')

# 与 rectpack 算法类一一对应；键为稳定 ID，供配置 CUTTING_ALGORITHMS_ENABLED 使用
PACKING_ALGORITHM_CLASSES: Dict[str, Any] = {
    "CommonDimStrip": "CommonDimStrip",
    "GuillotineBafMinas": rectpack.GuillotineBafMinas,
    "GuillotineBssfLlas": rectpack.GuillotineBssfLlas,
    "GuillotineBssfSlas": rectpack.GuillotineBssfSlas,
    "GuillotineBlsfLlas": rectpack.GuillotineBlsfLlas,
    "GuillotineBlsfSlas": rectpack.GuillotineBlsfSlas,
    "MaxRectsBaf": rectpack.MaxRectsBaf,
    "SkylineMwfWm": rectpack.SkylineMwfWm,
}

# auto 模式默认尝试的顺序（配置为空串时使用）
DEFAULT_AUTO_PACKING_IDS: List[str] = [
    "CommonDimStrip",
    "GuillotineBafMinas",
    "GuillotineBssfLlas",
    "GuillotineBssfSlas",
    "GuillotineBlsfLlas",
    "GuillotineBlsfSlas",
    "MaxRectsBaf",
    "SkylineMwfWm",
]

FALLBACK_PACKING_ID = "MaxRectsBssf"
# 未在注册表中时使用的回退类（旧逻辑）
FALLBACK_PACKING_CLASS = rectpack.MaxRectsBssf

# 非 rectpack 类、由 services 层特殊处理的主算法 id
NON_RECTPACK_PACKING_IDS = frozenset({"ORToolsAssignMaxRects", "ORToolsCP2D", "CommonDimStrip"})
ORTOOLS_PACKING_IDS = NON_RECTPACK_PACKING_IDS


def normalize_enabled_packing_ids(config_value: Optional[str]) -> List[str]:
    """
    解析配置中的启用列表。空或仅空白 → 使用 DEFAULT_AUTO_PACKING_IDS。
    格式：逗号分隔，如 GuillotineBafMinas,MaxRectsBaf
    """
    if not config_value or not str(config_value).strip():
        return list(DEFAULT_AUTO_PACKING_IDS)
    parsed = [x.strip() for x in str(config_value).split(",") if x.strip()]
    return parsed if parsed else list(DEFAULT_AUTO_PACKING_IDS)


def iter_enabled_packing_algorithms(
        enabled_ids: List[str],
) -> Iterable[Tuple[str, Any]]:
    """按配置顺序产出 (算法 id, rectpack 算法类 或 OR-Tools 占位 id 字符串)。"""
    for algo_id in enabled_ids:
        if algo_id in ORTOOLS_PACKING_IDS:
            yield algo_id, algo_id
            continue
        cls = PACKING_ALGORITHM_CLASSES.get(algo_id)
        if cls is not None:
            yield algo_id, cls
        else:
            logger.warning(
                "CUTTING_ALGORITHMS_ENABLED 含有未知算法 id：%s，已跳过",
                algo_id,
            )


def resolve_packing_class(algorithm_id: str) -> Any:
    """指定 id → 算法类；未知 id 时返回回退类。"""
    cls = PACKING_ALGORITHM_CLASSES.get(algorithm_id)
    if cls is not None:
        return cls
    logger.warning(
        "未知主算法 id %s，使用回退 %s",
        algorithm_id,
        FALLBACK_PACKING_ID,
    )
    return FALLBACK_PACKING_CLASS


def is_registered_packing_id(algorithm_id: str) -> bool:
    return (
        algorithm_id in PACKING_ALGORITHM_CLASSES
        or algorithm_id in ORTOOLS_PACKING_IDS
    )
