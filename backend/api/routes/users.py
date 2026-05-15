"""User profile + GDPR rights (erasure, portability)."""
from fastapi import APIRouter, Depends

from backend.api.dependencies import get_current_user
from backend.api.schemas import UserOut

router = APIRouter()


@router.get("/me", response_model=UserOut)
async def get_me(user=Depends(get_current_user)) -> UserOut:
    return user


@router.patch("/me", response_model=UserOut)
async def update_me(user=Depends(get_current_user)) -> UserOut:
    raise NotImplementedError


# GDPR: right to erasure
@router.delete("/me")
async def delete_me(user=Depends(get_current_user)) -> dict[str, str]:
    # AUDIT: user.delete.requested
    raise NotImplementedError


# GDPR: right to portability
@router.get("/me/export")
async def export_me(user=Depends(get_current_user)) -> dict:
    # AUDIT: user.export.requested
    raise NotImplementedError
