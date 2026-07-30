from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.bootstrap.runtime import AppRuntime


def create_lifespan(runtime: AppRuntime):
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            # 降级启动仍保留进程，使 /health 和 /ready 能报告依赖错误；业务入口再拒绝流量。
            runtime.start()
            yield
        finally:
            # shutdown 是终态：资源释放后不能复用同一 Runtime。
            runtime.shutdown()

    return lifespan
