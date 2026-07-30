from __future__ import annotations

from uuid import UUID

from fastapi import Header, HTTPException, Request

from app.bootstrap.runtime import AppRuntime


def device_id(x_device_id: str = Header(alias="X-Device-Id")) -> str:
    try:
        return str(UUID(x_device_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="无效设备标识") from error


def runtime(request: Request) -> AppRuntime:
    try:
        return request.app.state.runtime.start()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail="服务正在停止") from error
