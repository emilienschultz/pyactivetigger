import logging
import threading
import time
from collections import deque
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    oauth2_scheme,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    AuthActions,
    AuthUserModel,
    ChangeEmailModel,
    ChangePasswordModel,
    NewUserModel,
    ResetPasswordResultModel,
    UserCredentialInput,
    UserCredentialPublic,
    UserInDBModel,
    UserModel,
    UserStatistics,
)
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator

router = APIRouter(tags=["users"])

logger = logging.getLogger("activetigger.fastapi")


class _SlidingWindowLimiter:
    """Per-key sliding-window limiter. In-process; multi-worker setups multiply
    the effective limit by the worker count, which is acceptable for the
    unauthenticated reset endpoint but documented here for honesty."""

    def __init__(self, max_calls: int, window_seconds: float, max_keys: int = 10_000):
        self.max_calls = max_calls
        self.window = window_seconds
        self.max_keys = max_keys
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    # Evict the oldest-touched key to bound memory.
                    oldest = min(self._buckets, key=lambda k: self._buckets[k][-1])
                    self._buckets.pop(oldest, None)
                bucket = deque()
                self._buckets[key] = bucket
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_calls:
                return False
            bucket.append(now)
            return True


# Reset is expensive (sends an email) and unauthenticated, so the limits are
# tight. Tune by editing these constants; no live config needed.
_reset_ip_limiter = _SlidingWindowLimiter(max_calls=5, window_seconds=3600)
_reset_mail_limiter = _SlidingWindowLimiter(max_calls=3, window_seconds=3600)


@router.post("/users/disconnect", dependencies=[Depends(verified_user)], tags=["users"])
def disconnect_user(token: Annotated[str, Depends(oauth2_scheme)]) -> None:
    """
    Revoke user connexion
    """
    try:
        get_orchestrator().revoke_access_token(token)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/me", tags=["users"])
def read_users_me(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> UserModel:
    """
    Information on current user
    """
    try:
        contact = get_orchestrator().users.get_contact(current_user.username)
        return UserModel(
            username=current_user.username,
            status=current_user.status,
            contact=contact,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users", tags=["users"])
def existing_users(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict[str, UserModel]:
    """
    Get existing users
    """
    try:
        return get_orchestrator().users.existing_users(current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/recent", tags=["users"])
def recent_users() -> int:
    """
    Get the number of recently connected users
    """
    return len(get_orchestrator().db_manager.users_service.get_current_users(300))


@router.post("/users/create", dependencies=[Depends(verified_user)], tags=["users"])
def create_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    new_user: NewUserModel,
) -> None:
    """
    Create user
    """
    test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        get_orchestrator().users.add_user(new_user, current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/delete", dependencies=[Depends(verified_user)], tags=["users"])
def delete_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], user_to_delete: str
) -> None:
    """
    Delete user
    - root can delete all
    - users can only delete account they created
    """
    test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        get_orchestrator().users.delete_user(user_to_delete, current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/changepwd", dependencies=[Depends(verified_user)], tags=["users"])
def change_password(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    changepwd: ChangePasswordModel,
) -> None:
    """
    Change our own password for an account
    """
    try:
        get_orchestrator().users.change_password(
            current_user.username, changepwd.pwdold, changepwd.pwd1, changepwd.pwd2
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/admin-resetpwd", dependencies=[Depends(verified_user)], tags=["users"])
def admin_reset_password(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    username: str,
) -> ResetPasswordResultModel:
    """
    Reset a user's password (admin action). Generates a new random
    password, stores it, and returns it once to the caller.
    """
    test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        new_password = get_orchestrator().users.admin_reset_password(username)
        get_orchestrator().log_action(
            current_user.username, f"ADMIN RESET PASSWORD: {username}", "all"
        )
        return ResetPasswordResultModel(username=username, new_password=new_password)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/users/changemail", dependencies=[Depends(verified_user)], tags=["users"])
def change_email(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    changemail: ChangeEmailModel,
) -> None:
    """
    Change the contact email of the current user
    """
    try:
        get_orchestrator().users.change_email(
            current_user.username, changemail.email, changemail.password
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/users/credentials", dependencies=[Depends(verified_user)], tags=["users"])
def list_user_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[UserCredentialPublic]:
    """
    List the endpoint/credentials entries saved by the current user (secrets stay in the backend)
    """
    try:
        return get_orchestrator().users.list_credentials(current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/credentials", dependencies=[Depends(verified_user)], tags=["users"])
def add_user_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    credential: UserCredentialInput,
) -> None:
    """
    Save an endpoint/credentials entry for the current user (replaces an entry with the same name)
    """
    try:
        get_orchestrator().users.add_credentials(current_user.username, credential)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/users/credentials/delete", dependencies=[Depends(verified_user)], tags=["users"])
def delete_user_credentials(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete a saved endpoint/credentials entry of the current user
    """
    try:
        get_orchestrator().users.delete_credentials(current_user.username, name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/users/auth/{action}", dependencies=[Depends(verified_user)], tags=["users"])
def set_auth(
    action: AuthActions,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    auth: AuthUserModel,
) -> None:
    """
    Modify user auth on a specific project
    """
    test_rights(ProjectAction.UPDATE, current_user.username, auth.project_slug)
    if action == "add":
        if not auth.status:
            raise HTTPException(status_code=400, detail="Missing status")
        try:
            orchestrator = get_orchestrator()
            orchestrator.users.set_auth(auth)
            orchestrator.log_action(current_user.username, f"ADD AUTH USER: {auth.username}", "all")
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        return None

    if action == "delete":
        try:
            orchestrator = get_orchestrator()
            orchestrator.users.delete_auth(auth.username, auth.project_slug)
            orchestrator.log_action(
                current_user.username, f"DELETE AUTH USER: {auth.username}", "all"
            )
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500) from e

        return None

    raise HTTPException(status_code=400, detail="Action not found")


@router.get("/users/statistics", dependencies=[Depends(verified_user)], tags=["users"])
def get_statistics(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], username: str
) -> UserStatistics:
    """
    Get statistics for specific user.
    Users can access their own statistics; other users require MANAGE_USERS.
    """
    if username != current_user.username:
        test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        return get_orchestrator().users.get_statistics(username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/resetpwd", tags=["users"])
def reset_password(request: Request, mail: str) -> dict[str, str]:
    """
    Trigger a password reset email.

    Always returns a constant response, regardless of whether the address is
    registered or the mailer succeeded. This prevents account enumeration via
    response variance and prevents the mailer's error messages from leaking.
    Rate limits are applied per source IP and per target mail to bound abuse.
    """
    generic_response = {
        "detail": "If this address is registered, a reset email has been sent.",
    }

    client_ip = request.client.host if request.client else "unknown"
    normalized_mail = mail.strip().lower()

    if not _reset_ip_limiter.allow(client_ip):
        logger.warning("password reset rate-limited by IP %s", client_ip)
        return generic_response
    if not _reset_mail_limiter.allow(normalized_mail):
        logger.warning("password reset rate-limited for mail (ip=%s)", client_ip)
        return generic_response

    try:
        get_orchestrator().users.reset_password(mail)
    except Exception:
        # Swallow on purpose: leaking "unknown user" vs "mailer failed" enables
        # account enumeration. Log the real cause for operators instead.
        logger.exception("password reset failed (ip=%s)", client_ip)

    return generic_response
