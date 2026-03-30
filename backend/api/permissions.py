"""
Role-Based Access Control (RBAC) for the API.

Roles:
  - viewer: Read-only access to data, models, and results
  - analyst: Can run backtests, train models, collect data
  - admin: Full access including model deletion and system configuration
"""

from enum import StrEnum

from fastapi import Depends, HTTPException, status

from backend.api.dependencies import get_current_user


class Role(StrEnum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


# Role hierarchy: higher roles include all lower permissions
ROLE_HIERARCHY = {
    Role.VIEWER: 0,
    Role.ANALYST: 1,
    Role.ADMIN: 2,
}


def require_role(minimum_role: Role):
    """
    Dependency that enforces a minimum role level.

    Usage:
        @router.delete("/models/{id}", dependencies=[Depends(require_role(Role.ADMIN))])
        async def delete_model(id: str): ...
    """

    async def _check_role(user: dict = Depends(get_current_user)):
        user_role = user.get("payload", {}).get("role", Role.VIEWER.value)
        try:
            user_level = ROLE_HIERARCHY[Role(user_role)]
        except (ValueError, KeyError):
            user_level = 0

        if user_level < ROLE_HIERARCHY[minimum_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires {minimum_role.value} role or higher.",
            )
        return user

    return _check_role
