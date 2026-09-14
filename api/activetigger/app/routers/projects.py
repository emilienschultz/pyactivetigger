import random
import shutil
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import FileResponse, Response

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    check_auth_exists,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    AvailableProjectsModel,
    DatasetModel,
    EvalSetDataModel,
    EvalSetImageModel,
    LexicometricsModel,
    ProjectAuthsModel,
    ProjectBaseModel,
    ProjectDescriptionModel,
    ProjectStateModel,
    ProjectUpdateModel,
    UserInDBModel,
    WaitingModel,
)
from activetigger.errors import APIError
from activetigger.functions import slugify
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project
from activetigger.uploads import get_upload_staging

router = APIRouter(tags=["projects"])

# Allowed data files per project kind. Files reach these routes as staged
# chunked uploads (see activetigger.uploads), never as multipart bodies.
_ALLOWED_DATA_EXTENSIONS = (".csv", ".parquet", ".xlsx")
_ALLOWED_IMAGE_EXTENSIONS = (".zip",)


@router.post("/projects/close/{project_slug}", dependencies=[Depends(verified_user)])
def close_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> None:
    """
    Close a project from memory
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    try:
        get_orchestrator().stop_project(project_slug)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_slug}/statistics", dependencies=[Depends(verified_user)])
def get_project_statistics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str | None = None,
) -> ProjectDescriptionModel:
    """
    Statistics for a scheme and a user
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    try:
        return project.get_statistics(scheme=scheme)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_slug}/lexicometrics", dependencies=[Depends(verified_user)])
def get_lexicometrics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> LexicometricsModel | None:
    """
    Lexicometry statistics of the train dataset (None if not computed yet)
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    try:
        return project.lexicometrics.get()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/{project_slug}/lexicometrics/compute", dependencies=[Depends(verified_user)]
)
def compute_lexicometrics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> WaitingModel:
    """
    Launch the computation of lexicometry statistics on the train dataset
    """
    test_rights(ProjectAction.ADD, current_user.username, project.project_slug)
    try:
        project.lexicometrics.compute(
            project_slug=project.name,
            username=current_user.username,
            language=project.params.language,
        )
        get_orchestrator().log_action(
            current_user.username, "COMPUTE LEXICOMETRICS", project.project_slug
        )
        return WaitingModel(detail="Lexicometrics are computing")
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/auth", dependencies=[Depends(verified_user)])
def get_project_auth(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], project_slug: str
) -> ProjectAuthsModel:
    """
    Users auth on a project
    """
    orchestrator = get_orchestrator()
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project doesn't exist")
    test_rights(ProjectAction.MONITOR, current_user.username, project_slug)
    try:
        return ProjectAuthsModel(auth=orchestrator.users.get_project_auth(project_slug))
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500) from e


@router.post("/projects/new", dependencies=[Depends(verified_user)])
def new_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: ProjectBaseModel,
) -> str:
    """
    Start the creation of a new project.

    The data file is referenced by `project.upload_id` (chunked-upload
    protocol, see activetigger.uploads) and moved here into the new project
    folder — unless the data comes from another project or a toy dataset
    (`from_project` / `from_toy_dataset`), placed by /files/copy/project.
    """
    orchestrator = get_orchestrator()
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)

    copies_existing_data = project.from_project is not None or project.from_toy_dataset
    if project.upload_id is None and not copies_existing_data:
        raise HTTPException(status_code=400, detail="No upload_id provided for the data file")

    # add a delay if projects are already being created
    if len(orchestrator.project_creation_ongoing) >= 3:
        time.sleep(random.randint(1, 4))

    try:
        # place the staged data file in the new project folder; strict mkdir
        # doubles as a guard against two concurrent creations of the same name
        if project.upload_id is not None:
            allowed = (
                _ALLOWED_IMAGE_EXTENSIONS if project.kind == "image" else _ALLOWED_DATA_EXTENSIONS
            )
            project_slug = orchestrator.check_project_name(project.project_name)
            project_path = orchestrator.path.joinpath(project_slug)
            try:
                project_path.mkdir(parents=True)
            except FileExistsError:
                raise HTTPException(
                    status_code=409,
                    detail="Project already exists, please choose another name",
                )
            try:
                target = get_upload_staging().move_to(
                    current_user.username, project.upload_id, project_path, allowed
                )
            except Exception:
                shutil.rmtree(project_path, ignore_errors=True)
                raise
            project.filename = target.name

        project_slug = orchestrator.starting_project_creation(
            project=project,
            username=current_user.username,
        )
        orchestrator.log_action(
            current_user.username, f"START CREATING PROJECT: {project_slug}", project_slug
        )
        return project_slug
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        orchestrator.clean_unfinished_project(project_name=project.project_name)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/update",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
def update_project(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    update: ProjectUpdateModel,
) -> None:
    """
    Update a project
    - change the name
    - change the language
    - change context cols
    - change text cols
    - expand the number of elements in the trainset
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)
    try:
        project.start_update_project(update, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO UPDATE PROJECT: {project.project_slug}",
            project.project_slug,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/duplicate", dependencies=[Depends(verified_user)])
def duplicate_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> str:
    """
    Kick off the duplication of an existing project (files + DB rows) under
    `<project_slug>-copy`. Returns the target slug immediately. Callers should poll
    /projects/status to know when the copy has finished.
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    test_rights(ProjectAction.GET, current_user.username, project_slug)
    orchestrator = get_orchestrator()
    try:
        new_slug = orchestrator.duplicate_project(project_slug, current_user.username)
        orchestrator.log_action(
            current_user.username,
            f"START DUPLICATING PROJECT: {project_slug} -> {new_slug}",
            new_slug,
        )
        return new_slug
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/delete",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
def delete_project(
    project_slug: str,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Delete a project
    """
    test_rights(ServerAction.DELETE_PROJECT, current_user.username, project_slug)
    try:
        orchestrator = get_orchestrator()
        orchestrator.delete_project(project_slug)
        orchestrator.log_action(
            current_user.username, f"DELETE PROJECT: {project_slug}", project_slug
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/status", dependencies=[Depends(verified_user)])
def get_project_status(
    project_name: str,
) -> str:
    """
    Get the status of a project
    - not existing
    - creating
    - duplicating
    - existing
    """
    try:
        orchestrator = get_orchestrator()
        slug = slugify(project_name)
        # if project is being duplicated (filesystem copy still running, or
        # done but DB clone not yet finalized by the update loop)
        if slug in orchestrator.project_duplication_ongoing:
            return "duplicating"
        # if project is in creation
        if slug in orchestrator.project_creation_ongoing:
            # Try to read creation progress for image projects
            project = orchestrator.project_creation_ongoing[slug]
            try:
                proj_dir = getattr(project, "params", None) and project.params.dir
                if proj_dir is not None:
                    progress_file = Path(proj_dir) / "creation_progress"
                    if progress_file.exists():
                        return f"creating:{progress_file.read_text().strip()}"
            except Exception:
                pass
            return "creating"
        # if creation failed, return the error (consumed once)
        if slug in orchestrator.creation_errors:
            error_msg = orchestrator.creation_errors.pop(slug)
            return f"error: {error_msg}"
        elif orchestrator.exists(project_name):
            return "existing"
        else:
            return "not existing"
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/evalset/delete", dependencies=[Depends(verified_user)])
def delete_evalset(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    dataset: str,
) -> None:
    """
    Delete an existing eval dataset
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)
    try:
        project.drop_evalset(dataset=dataset)
        get_orchestrator().log_action(
            current_user.username, f"DELETE EVALSET {dataset}", project.project_slug
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/projects/evalset/add", dependencies=[Depends(verified_user)])
def add_testdata(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    dataset: str,
    evalset: EvalSetDataModel | EvalSetImageModel,
) -> str | None:
    """
    Add a dataset for eval/test when there is none available.

    The data file(s) are referenced by upload id (chunked-upload protocol,
    see activetigger.uploads) and moved here into the project data folder,
    where the async task reads them back by filename.
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)

    # move the staged file(s) into the project data folder and record the
    # sanitized on-disk names for the task
    staging = get_upload_staging()
    if isinstance(evalset, EvalSetImageModel):
        allowed = _ALLOWED_IMAGE_EXTENSIONS + _ALLOWED_DATA_EXTENSIONS
    else:
        allowed = _ALLOWED_DATA_EXTENSIONS
    target = staging.move_to(
        current_user.username, evalset.upload_id, project.data.path_datasets, allowed
    )
    evalset.filename = target.name
    if isinstance(evalset, EvalSetImageModel) and evalset.labels_upload_id is not None:
        labels_target = staging.move_to(
            current_user.username,
            evalset.labels_upload_id,
            project.data.path_datasets,
            _ALLOWED_DATA_EXTENSIONS,
        )
        evalset.labels_filename = labels_target.name

    try:
        id = project.add_evalset(dataset, evalset, current_user.username, project.project_slug)
        get_orchestrator().log_action(
            current_user.username, f"ADD EVALSET {dataset}", project.project_slug
        )
        return id
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/projects")
def get_projects(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> AvailableProjectsModel:
    """
    Get general informations on the server
    depending of the status of connected user
    """
    try:
        orchestrator = get_orchestrator()
        return AvailableProjectsModel(
            projects=orchestrator.users.get_user_projects(current_user.username),
            storage_used=orchestrator.users.get_storage(current_user.username),
            storage_limit=orchestrator.users.get_storage_limit(current_user.username),
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", dependencies=[Depends(verified_user)])
def get_project_datasets(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    include_toy_datasets: bool = False,
) -> tuple[list[DatasetModel], list[DatasetModel] | None]:
    """
    Get all datasets already available for a specific user
    """
    try:
        orchestrator = get_orchestrator()
        toy_datasets = orchestrator.get_toy_datasets() if include_toy_datasets else []
        auth_datasets = orchestrator.users.get_auth_datasets(current_user.username)
        return auth_datasets, toy_datasets
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/projects/{project_slug}",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
def get_project_state(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
) -> ProjectStateModel:
    """
    Get the state of a specific project
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    try:
        return project.state()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Experimental image projects (see docs/image-projects-strategy.md) ---


@router.get(
    "/projects/{project_slug}/image_imagexp/{element_id}",
    dependencies=[Depends(verified_user)],
)
def get_project_image_imagexp(
    project_slug: str,
    element_id: str,
    request: Request,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
) -> Response:
    """
    Stream an image element from an image project.
    Safe caching: images are immutable once uploaded.
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    if getattr(project.params, "kind", "text") != "image":
        raise HTTPException(status_code=400, detail="Not an image project")
    if project.params.dir is None:
        raise HTTPException(status_code=500, detail="Project directory missing")
    images_dir = Path(project.params.dir) / "images"
    safe_id = Path(element_id).name  # strip any path component

    # Resolve the on-disk path via the in-memory DataFrames (train/valid/test).
    # The "text" column holds the file path for image projects.
    candidate: Path | None = None
    for df in (project.data.train, project.data.valid, project.data.test):
        if df is not None and safe_id in df.index and "text" in df.columns:
            p = Path(str(df.loc[safe_id, "text"]))
            if p.exists():
                candidate = p
                break

    # Fallback: try direct stem match in images_dir
    if candidate is None:
        for ext in (".png", ".jpg", ".jpeg"):
            p = images_dir / f"{safe_id}{ext}"
            if p.exists():
                candidate = p
                break

    # Confine to images_dir for safety
    if candidate is not None:
        try:
            candidate.resolve().relative_to(images_dir.resolve())
        except ValueError:
            candidate = None

    if candidate is None:
        raise HTTPException(status_code=404, detail="Image not found")

    stat = candidate.stat()
    etag = f'"{int(stat.st_mtime)}-{stat.st_size}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)
    return FileResponse(
        candidate,
        headers={
            "Cache-Control": "private, max-age=3600",
            "ETag": etag,
        },
    )


@router.get(
    "/projects/{project_slug}/thumbnail_imagexp/{element_id}",
    dependencies=[Depends(verified_user)],
)
def get_project_thumbnail_imagexp(
    project_slug: str,
    element_id: str,
    request: Request,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
) -> Response:
    """
    Stream a precomputed 256px JPEG thumbnail for an image element.
    Falls back to the original image if the thumbnail file is missing
    (e.g. ingest failure or older project), so the route is safe to deploy
    without a backfill migration.
    """
    test_rights(ProjectAction.GET, current_user.username, project.project_slug)
    if getattr(project.params, "kind", "text") != "image":
        raise HTTPException(status_code=400, detail="Not an image project")
    if project.params.dir is None:
        raise HTTPException(status_code=500, detail="Project directory missing")

    images_dir = Path(project.params.dir) / "images"
    thumbs_dir = images_dir / "thumbs"
    safe_id = Path(element_id).name

    # Thumbnails are named `{id_internal}.jpg` under images/thumbs/.
    # If missing (older project or ingest failure), fall back to the original.
    candidate: Path | None = None
    thumb_path = thumbs_dir / f"{safe_id}.jpg"
    if thumb_path.exists():
        try:
            thumb_path.resolve().relative_to(images_dir.resolve())
            candidate = thumb_path
        except ValueError:
            pass

    # Fallback: serve the original via in-memory DataFrames
    if candidate is None:
        for df in (project.data.train, project.data.valid, project.data.test):
            if df is not None and safe_id in df.index and "text" in df.columns:
                p = Path(str(df.loc[safe_id, "text"]))
                if p.exists():
                    candidate = p
                    break

    if candidate is None:
        for ext in (".png", ".jpg", ".jpeg"):
            p = images_dir / f"{safe_id}{ext}"
            if p.exists():
                candidate = p
                break

    if candidate is not None:
        try:
            candidate.resolve().relative_to(images_dir.resolve())
        except ValueError:
            candidate = None

    if candidate is None:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    stat = candidate.stat()
    etag = f'"{int(stat.st_mtime)}-{stat.st_size}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)
    return FileResponse(
        candidate,
        headers={
            "Cache-Control": "private, max-age=3600",
            "ETag": etag,
        },
    )
