from fastapi import APIRouter

from app.api.v1.endpoints import chat, health

router = APIRouter()
router.include_router(chat.router)

health_router = APIRouter()
health_router.include_router(health.router)
