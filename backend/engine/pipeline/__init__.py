"""切割流水线：输入归一、顺序装箱、低利用率 refine、OR-Tools 分板、追踪日志。"""

from engine.pipeline.constants import REFINE_LOW_UTIL_THRESHOLD, REFINE_MAX_PASSES

__all__ = [
    "REFINE_LOW_UTIL_THRESHOLD",
    "REFINE_MAX_PASSES",
]
