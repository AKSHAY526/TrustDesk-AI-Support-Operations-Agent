"""Token authentication with three roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fastapi import Depends, Header, HTTPException, status

from .config import settings


class Role(StrEnum):
    AGENT = "support_agent"
    MANAGER = "support_manager"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    user_id: str
    role: Role

    @property
    def can_approve(self) -> bool:
        return self.role in (Role.MANAGER, Role.ADMIN)

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN


def _token_table() -> dict[str, Principal]:
    return {
        settings.token_agent: Principal("agent@demo", Role.AGENT),
        settings.token_manager: Principal("manager@demo", Role.MANAGER),
        settings.token_admin: Principal("admin@demo", Role.ADMIN),
    }


def current_principal(authorization: str | None = Header(default=None)) -> Principal:
    """Resolve the caller from an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "missing_credentials", "message": "Authorization header is required."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ").strip()
    principal = _token_table().get(token)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_token", "message": "The supplied token is not recognised."},
        )
    return principal


def require_approver(principal: Principal = Depends(current_principal)) -> Principal:
    """Only a manager or admin may approve or reject a sensitive action."""
    if not principal.can_approve:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "insufficient_role",
                "message": "Approving a sensitive action requires the support_manager role.",
                "your_role": str(principal.role),
            },
        )
    return principal


def require_manager(principal: Principal = Depends(current_principal)) -> Principal:
    """The approval queue and the policy-gap report are for managers and admins."""
    if not principal.can_approve:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "insufficient_role",
                "message": "This is only available to support managers and admins.",
                "your_role": str(principal.role),
            },
        )
    return principal


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    """Knowledge-base ingestion and evaluations are admin-only."""
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "insufficient_role",
                "message": "This operation requires the admin role.",
                "your_role": str(principal.role),
            },
        )
    return principal
