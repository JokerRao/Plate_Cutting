"""切割流水线阶段日志与可选 JSON dump（配置 CUTTING_TRACE_LOG_STAGES / CUTTING_DEBUG_DUMP_DIR）。"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("plate_cutting")


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if hasattr(obj, "length") and hasattr(obj, "width"):
        return {
            "length": getattr(obj, "length", None),
            "width": getattr(obj, "width", None),
            "plate_id": getattr(obj, "plate_id", None),
            "quantity": getattr(obj, "quantity", None),
        }
    return str(obj)


@dataclass
class CuttingTraceContext:
    """单次 optimize / run_single 的追踪上下文。"""

    run_id: str
    algorithm_label: str
    log_stages: bool
    dump_dir: Optional[str]
    _seq: int = 0

    @classmethod
    def from_settings(cls, algorithm_label: str, settings: Any) -> "CuttingTraceContext":
        dump = getattr(settings, "CUTTING_DEBUG_DUMP_DIR", "") or ""
        dump_dir = dump.strip() or None
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        return cls(
            run_id=str(uuid.uuid4())[:8],
            algorithm_label=algorithm_label,
            log_stages=bool(getattr(settings, "CUTTING_TRACE_LOG_STAGES", False)),
            dump_dir=dump_dir,
        )

    def stage(self, name: str, **fields: Any) -> None:
        if self.log_stages:
            parts = [f"{k}={fields[k]!r}" for k in sorted(fields)]
            logger.info(
                "cutting_trace run_id=%s algorithm=%s stage=%s %s",
                self.run_id,
                self.algorithm_label,
                name,
                " ".join(parts),
            )
        if self.dump_dir:
            self._seq += 1
            payload: Dict[str, Any] = {
                "run_id": self.run_id,
                "algorithm": self.algorithm_label,
                "stage": name,
                "seq": self._seq,
                "ts": time.time(),
                "fields": _json_safe(fields),
            }
            path = os.path.join(
                self.dump_dir,
                f"{self.run_id}_{self._seq:03d}_{name}.json",
            )
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.warning("cutting_trace dump failed: %s", e)

    def summarize_plates_orders(
        self,
        n_big: int,
        n_orders: int,
        n_stock: int,
    ) -> None:
        self.stage(
            "inputs_ready",
            n_big_plates=n_big,
            n_order_pieces=n_orders,
            n_stock_pieces=n_stock,
        )
