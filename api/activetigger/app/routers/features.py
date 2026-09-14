from typing import Annotated

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Query,
)

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    FeatureDescriptionModelOut,
    FeatureModel,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project
from activetigger.uploads import get_upload_staging

router = APIRouter(tags=["features"])

_ALLOWED_IMPORT_EXTENSIONS = (".csv", ".parquet", ".xlsx")


@router.post("/features/add", dependencies=[Depends(verified_user)])
def post_embeddings(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    feature: FeatureModel,
):
    """
    Compute features :
    - same prcess
    - specific process : function + temporary file + update
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        # gather all text data to compute features on
        if project.data.train is None:
            raise HTTPException(
                status_code=400, detail="No training data available to compute features."
            )
        series_list = [project.data.train["text"]]
        if project.data.valid is not None:
            series_list.append(project.data.valid["text"])
        if project.data.test is not None:
            series_list.append(project.data.test["text"])
        df = pd.concat(series_list)

        # compute features
        project.features.compute(
            df,
            feature.name,
            feature.use_default_name,
            feature.type,
            feature.parameters,
            current_user.username,
        )
        get_orchestrator().log_action(
            current_user.username, f"COMPUTE FEATURE: {feature.type}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/delete", dependencies=[Depends(verified_user)])
def delete_feature(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
) -> None:
    """
    Delete a specific feature
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        project.features.delete(name)
        get_orchestrator().log_action(
            current_user.username, f"DELETE FEATURE: {name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/reset", dependencies=[Depends(verified_user)])
def reset_features(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Reset all features: delete and recreate the features parquet file
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        project.features.reset_features_file()
        get_orchestrator().log_action(current_user.username, "RESET FEATURES", project.name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/import", dependencies=[Depends(verified_user)])
def import_feature(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    upload_id: str,
    name: str = Form(...),
    id_column: str = Form(...),
    columns: str | None = Form(None),
) -> None:
    """
    Import a pre-computed feature from a staged upload (chunked-upload
    protocol, see activetigger.uploads). All target columns must be numeric.
    The stored feature is named `imported-<name>`.
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)

    staging = get_upload_staging()
    staged_path, filename = staging.claim(
        current_user.username, upload_id, _ALLOWED_IMPORT_EXTENSIONS
    )

    try:
        lower = filename.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(staged_path, sep=None, engine="python")
        elif lower.endswith(".parquet"):
            df = pd.read_parquet(staged_path)
        else:
            df = pd.read_excel(staged_path)

        selected = (
            [c.strip() for c in columns.split(",") if c.strip()]
            if columns is not None and columns.strip() != ""
            else None
        )

        full_name = project.features.import_from_dataframe(
            name=name,
            id_column=id_column,
            df=df,
            columns=selected,
            username=current_user.username,
            source_file=filename,
        )
        get_orchestrator().log_action(
            current_user.username, f"IMPORT FEATURE: {full_name}", project.name
        )
        # the staged file is consumed only on success so a failed import
        # can be retried without re-uploading
        staging.discard(current_user.username, upload_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/available", dependencies=[Depends(verified_user)])
def get_feature_info(
    project: Annotated[Project, Depends(get_project)],
) -> dict[str, FeatureDescriptionModelOut]:
    """
    Get feature info
    """
    try:
        return project.features.get_available()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
