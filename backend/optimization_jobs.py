"""异步优化任务内存注册表（多进程/重启后任务丢失，生产可换 Redis）。"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OptimizationJob:
    job_id: str
    status: str  # pending | running | completed | failed
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OptimizationJobRegistry:
    def __init__(self, max_jobs: int = 500) -> None:
        self._max_jobs = max_jobs
        self._jobs: Dict[str, OptimizationJob] = {}
        self._lock = asyncio.Lock()

    def _evict_oldest(self) -> None:
        while len(self._jobs) >= self._max_jobs:
            finished_jobs = {k: v for k, v in self._jobs.items() if v.status in ("completed", "failed")}
            if finished_jobs:
                oldest_id = min(finished_jobs.items(), key=lambda x: x[1].created_at)[0]
                del self._jobs[oldest_id]
            else:
                break

    async def create_pending(self) -> str:
        async with self._lock:
            self._evict_oldest()
            job_id = str(uuid.uuid4())
            self._jobs[job_id] = OptimizationJob(
                job_id=job_id, status="pending")
            return job_id

    async def get(self, job_id: str) -> Optional[OptimizationJob]:
        async with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            return OptimizationJob(
                job_id=j.job_id,
                status=j.status,
                created_at=j.created_at,
                result=j.result,
                error=j.error,
            )

    async def mark_running(self, job_id: str) -> bool:
        async with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return False
            j.status = "running"
            return True

    async def mark_completed(
            self, job_id: str, result: Dict[str, Any]) -> bool:
        async with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return False
            j.status = "completed"
            j.result = result
            j.error = None
            return True

    async def mark_failed(self, job_id: str, message: str) -> bool:
        async with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return False
            j.status = "failed"
            j.error = message
            j.result = None
            return True
