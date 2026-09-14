import asyncio
from enum import Enum
from typing import Annotated

from fastapi import (
    Depends,
    HTTPException,
    Request,
)
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Transient database failures ("database is locked", connection-pool timeout).
# These must never be reported as 401: the frontend drops the whole user
# session on 401, so a DB hiccup would log every user out.
DB_UNAVAILABLE_ERRORS = (OperationalError, SQLAlchemyTimeoutError)

# per-project locks to avoid duplicate loading without blocking unrelated projects
_project_locks: dict[str, asyncio.Lock] = {}


def _get_lock(project_slug: str) -> asyncio.Lock:
    if project_slug not in _project_locks:
        _project_locks[project_slug] = asyncio.Lock()
    return _project_locks[project_slug]


async def get_project(project_slug: str) -> Project:
    """
    Dependency to get existing project
    - if already loaded, return it
    - if not loaded, load it first (in a thread to avoid blocking the event loop)
    """
    orchestrator = get_orchestrator()

    # test if project exists
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project not found")

    # fast path: project already loaded
    if project_slug in orchestrator.projects:
        return orchestrator.projects[project_slug]

    # slow path: load in a background thread, with a per-project lock
    # so concurrent requests for the same project wait instead of loading twice
    async with _get_lock(project_slug):
        # re-check after acquiring the lock (another request may have loaded it)
        if project_slug in orchestrator.projects:
            return orchestrator.projects[project_slug]
        try:
            await asyncio.to_thread(orchestrator.manage_fifo_queue)
            await asyncio.to_thread(orchestrator.start_project, project_slug)
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return orchestrator.projects[project_slug]


def verified_user(request: Request, token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDBModel:
    """
    Dependency to test if the user is authentified with its token
    """
    orchestrator = get_orchestrator()
    # decode token
    try:
        payload = orchestrator.decode_access_token(token)
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Problem with token")
    except HTTPException:
        raise
    except JWTError:
        raise HTTPException(status_code=401, detail="Problem with token")
    except DB_UNAVAILABLE_ERRORS as e:
        raise HTTPException(
            status_code=503, detail="Database temporarily unavailable, please retry"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=401, detail="Problem with token") from e

    # get user caracteristics
    try:
        return orchestrator.users.get_user(name=username)
    except DB_UNAVAILABLE_ERRORS as e:
        raise HTTPException(
            status_code=503, detail="Database temporarily unavailable, please retry"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=404) from e


def check_auth_exists(
    request: Request,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> None:
    """
    Check if a user is associated to a project
    """
    try:
        auth = get_orchestrator().users.auth(current_user.username, project_slug)
        if not auth:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid rights")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid rights") from e


# --------------------------
# Rights management
# --------------------------


class ServerAction(str, Enum):
    MANAGE_USERS = "manage users"
    MANAGE_SERVER = "manage server"
    KILL_PROCESS = "kill process"
    CREATE_PROJECT = "create project"
    DELETE_PROJECT = "delete project"


class ProjectAction(str, Enum):
    ADD = "add to project"
    DELETE = "delete from project"
    UPDATE = "update project"
    GET = "get project information"
    ADD_ANNOTATION = "add annotation"
    UPDATE_ANNOTATION = "modify annotation"
    EXPORT_DATA = "export data"
    MANAGE_FILES = "manage files"
    MONITOR = "access specific project information"
    GENERATE = "use generation features"


def test_rights(
    action: ServerAction | ProjectAction,
    username: str,
    project_slug: str | None = None,
    scheme: str | None = None,
) -> bool:
    """
    Management of rights on the routes

    Based on an action, a user, a project
    Existing status : root, manager, annotator, demo
    Existing rights : manager, contributor
    Not implemented : rights specific to scheme
    """
    orchestrator = get_orchestrator()
    try:
        user = orchestrator.users.get_user(name=username)
    except Exception as e:
        raise HTTPException(404) from e

    if action not in ServerAction and action not in ProjectAction:
        raise HTTPException(
            status_code=500,
            detail=f"Action {action} is not a valid action",
        )

    # general status
    status = user.status

    # root user can do anything
    if status == "root":
        return True

    match action:
        case ServerAction.CREATE_PROJECT | ServerAction.DELETE_PROJECT:
            if status in ["manager"]:
                return True

    # specific case demo
    if status == "demo":
        if action in [ProjectAction.GET, ProjectAction.ADD_ANNOTATION]:
            return True
        else:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: User {username} has no rights to perform action {action} on project {project_slug}",
            )

    # Get auth for the project
    if not project_slug:
        raise HTTPException(500, "Project name missing")
    auth = orchestrator.users.auth(username, project_slug)
    if auth is None:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: User {username} has no rights to perform action {action} on project {project_slug}",
        )

    match action:
        # only manager can delete/modify elements
        case (
            ProjectAction.DELETE
            | ProjectAction.UPDATE
            | ProjectAction.UPDATE_ANNOTATION
            | ProjectAction.EXPORT_DATA
            | ProjectAction.MANAGE_FILES
        ):
            if auth in ["manager"]:
                return True
        # only manager and contributor can create
        case ProjectAction.ADD | ProjectAction.MONITOR | ProjectAction.GENERATE:
            if auth in ["manager", "contributor"]:
                return True
        # everyone can get info or add annotation
        case ProjectAction.ADD_ANNOTATION | ProjectAction.GET:
            return True

    # by default, no rights
    raise HTTPException(
        status_code=403,
        detail=f"Forbidden: User {username} has no rights to perform action {action} on project {project_slug}",
    )
