import importlib.resources
import logging
import time
from contextlib import asynccontextmanager
from importlib.abc import Traversable
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from activetigger import __version__
from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    test_rights,
    verified_user,
)
from activetigger.app.routers import (
    annotations,
    bertopic,
    export,
    features,
    files,
    generation,
    messages,
    models,
    monitoring,
    projects,
    prompts,
    schemes,
    tasks,
    toolbox,
    upload,
    users,
)
from activetigger.config import config
from activetigger.datamodels import (
    ServerStateModel,
    TableOutModel,
    TokenModel,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator

# ensure the static dir exists before the mount below; orchestrator init runs in lifespan
(Path(config.data_path) / "projects" / "static").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Frame the execution of the api
    """
    orchestrator = get_orchestrator()
    # If the orchestrator was built at module-import time (no event loop),
    # its background loops weren't scheduled — start them now.
    orchestrator.ensure_update_task()
    app.state.orchestrator = orchestrator
    print("Active Tigger starting")
    yield
    print("Active Tigger closing")


# starting the app
app = FastAPI(lifespan=lifespan, root_path="/api")


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(OverflowError)
async def overflow_error_handler(request: Request, exc: OverflowError) -> JSONResponse:
    # numeric input beyond what the storage layer accepts (SQLite INTEGER, C int, dates)
    return JSONResponse(status_code=422, content={"detail": f"Value out of range: {exc}"})


# setup file logger for fastapi events
log_dir = Path(config.data_path) / "projects" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir.joinpath("fastapi_events.log")

fastapi_logger = logging.getLogger("activetigger.fastapi")
fastapi_logger.setLevel(logging.INFO)
fastapi_logger.propagate = False

if not fastapi_logger.handlers:
    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="H",
        interval=1,
        backupCount=24,
        encoding="utf-8",
        utc=False,
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S%z",
        )
    )
    fastapi_logger.addHandler(handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    warning_delay = 1000
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    path = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        fastapi_logger.info(
            '%s "%s %s" status=%s duration_ms=%s %s',
            client,
            method,
            path,
            response.status_code,
            duration_ms,
            "DELAY" if duration_ms > warning_delay else "",
        )
        return response
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        fastapi_logger.exception(
            '%s "%s %s" status=500 duration_ms=%s PB',
            client,
            method,
            path,
            duration_ms,
        )
        raise


# add static folder
get_orchestrator()  # fix to create all folders
app.mount(
    "/static", StaticFiles(directory=Path(config.data_path) / "projects" / "static"), name="static"
)

# error statuses any authenticated route can answer, documented in the OpenAPI schema
COMMON_ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"description": "Invalid request"},
    401: {"description": "Not authenticated"},
    403: {"description": "Not enough rights"},
    404: {"description": "Resource not found"},
    409: {"description": "Resource already exists"},
    500: {"description": "Internal server error"},
}

# add routers
app.include_router(users.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(projects.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(annotations.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(schemes.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(features.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(prompts.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(export.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(models.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(generation.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(files.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(bertopic.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(messages.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(monitoring.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(toolbox.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(upload.router, responses=COMMON_ERROR_RESPONSES)
app.include_router(tasks.router, responses=COMMON_ERROR_RESPONSES)

# allow multiple servers (avoir CORS error)
# TODO : Read allowed origins from config: `allow_origins=config.cors_origins`
# (default to `[]` in prod). Keep `allow_credentials=True` only when origins is an explicit list.
# Restrict methods/headers to what the frontend actually uses.


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------
# Generic routes
# --------------


@app.get("/", response_class=HTMLResponse)
def welcome() -> str:
    """
    Welcome page for the API
    """
    data_path: Traversable = importlib.resources.files("activetigger")
    with open(str(data_path.joinpath("html", "welcome.html")), "r") as f:
        r = f.read()
    return r


@app.get("/version")
def get_version() -> str:
    """
    Get the version of the server
    """
    return __version__


@app.post(
    "/server/restart", dependencies=[Depends(verified_user)], responses=COMMON_ERROR_RESPONSES
)
def restart_queue(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Restart the queue & the memory
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    try:
        get_orchestrator().reset()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/server")
def get_queue() -> ServerStateModel:
    """
    Get the state of the server
    """
    return get_orchestrator().server_state


@app.post("/token", responses={401: {"description": "Wrong username or password"}})
def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenModel:
    """
    Authentificate user from username/password and return token
    """
    try:
        orchestrator = get_orchestrator()
        user = orchestrator.users.authenticate_user(form_data.username, form_data.password)
        access_token = orchestrator.create_access_token(
            data={"sub": user.username}, expires_min=1440
        )
        return TokenModel(access_token=access_token, token_type="bearer", status=user.status)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@app.get("/logs", dependencies=[Depends(verified_user)], responses=COMMON_ERROR_RESPONSES)
def get_logs(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str = "all",
    limit: int = Query(100, ge=1, le=10_000),
) -> TableOutModel:
    """
    Get all logs for a username/project
    """
    if project_slug == "all":
        test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    else:
        test_rights(ProjectAction.GET, current_user.username, project_slug)
    df = get_orchestrator().get_logs(project_slug, limit)
    return TableOutModel(
        items=df.to_dict(orient="records"),
        total=limit,
    )


@app.post("/stop", dependencies=[Depends(verified_user)], responses=COMMON_ERROR_RESPONSES)
def stop_process(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    unique_id: str | None = None,
    project_slug: str | None = None,
    kind: str | None = None,
) -> None:
    """
    Stop processes either by unique_id or by kind for a user
    - unique_id: stop a specific process (only for administrator)
    - kind: stop all processes of a given kind for the user
    """
    if unique_id is None and kind is None:
        raise HTTPException(status_code=400, detail="You must provide a unique_id or a kind")
    try:
        orchestrator = get_orchestrator()
        if unique_id is not None:
            test_rights(ServerAction.MANAGE_SERVER, current_user.username)
            orchestrator.stop_process(unique_id, current_user.username)
        if project_slug is not None:
            # rights already checked
            orchestrator.stop_user_processes(current_user.username, project_slug, kind)
        orchestrator.log_action(
            current_user.username,
            f"STOP PROCESS: {kind if kind is not None else unique_id}",
            "general",
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
