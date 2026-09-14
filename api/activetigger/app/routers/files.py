import os
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    test_rights,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator

# Uploaded files are sent through the chunked-upload protocol
# Only server-side copies remain here.

router = APIRouter(tags=["files"])


@router.post("/files/copy/project")
def copy_existing_data(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    source_project: str,
    from_toy_dataset: bool = False,
) -> None:
    """
    Copy an existing project to create a new one
    if copy dataset from toy datasets: orchestrator.path_toy_datasets/NAME.parquet
    if copy from project: orchestrator.path/NAME/data_all.parquet
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)

    orchestrator = get_orchestrator()

    # check if the project does not already exist
    if orchestrator.exists(project_name):
        raise HTTPException(
            status_code=409, detail="Project already exists, please choose another name"
        )

    # Validate the source. `source_project` is user-supplied and used in a path,
    # so reduce it to a basename and require it to refer to a real, authorized
    # source before doing anything with the filesystem.
    source_name = Path(source_project).name
    if not source_name or source_name != source_project:
        raise HTTPException(status_code=400, detail="Invalid source_project")

    if from_toy_dataset:
        allowed = {d.project_slug for d in orchestrator.get_toy_datasets()}
        if source_name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown toy dataset")
        source_path = orchestrator.path_toy_datasets / f"{source_name}.parquet"
    else:
        if source_name not in orchestrator.existing_projects():
            raise HTTPException(status_code=404, detail="Unknown source project")
        # Confirm the caller is actually authorized on the source project.
        test_rights(ProjectAction.GET, current_user.username, source_name)
        source_path = Path(orchestrator.path) / source_name / config.data_all

    # try to copy the project
    try:
        # create a folder for the project to be created
        project_slug = orchestrator.check_project_name(project_name)
        project_path = Path(f"{orchestrator.path}/{project_slug}")
        os.makedirs(project_path)

        # copy the full dataset
        shutil.copyfile(
            source_path,
            project_path.joinpath(config.data_all),
        )

    except HTTPException:
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise
    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():  # ty: ignore[possibly-unresolved-reference]
            shutil.rmtree(project_path)  # ty: ignore[possibly-unresolved-reference]
        raise HTTPException(status_code=500, detail=str(e))
