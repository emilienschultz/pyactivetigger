from io import StringIO
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
)
from fastapi.responses import FileResponse
from fastapi.responses import Response as FastAPIResponse

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.config import config
from activetigger.datamodels import (
    ExportGenerationsParams,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.project import Project

router = APIRouter(tags=["export"])


@router.get("/export/data", dependencies=[Depends(verified_user)])
def export_data(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    format: str,
    dataset: str = "train",
) -> FileResponse:
    """
    Export labelled data
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.export_data(format=format, scheme=scheme, dataset=dataset)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/summary", dependencies=[Depends(verified_user)])
def export_summary(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict:
    """
    Lab-notebook style snapshot of the project (JSON).
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.export_summary()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/features", dependencies=[Depends(verified_user)])
def export_features(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    features: list = Query(),
    format: str = Query(),
) -> FileResponse:
    """
    Export features
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.export_features(features=features, format=format)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/projection", dependencies=[Depends(verified_user)])
def export_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    format: str = Query(),
    projection_name: str = Query(),
) -> FileResponse:
    """
    Export a named projection
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.projections.export(
            name=projection_name,
            format=format,
            col_id=project.params.col_id,
            id_mapping=project.data.index,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/prediction", dependencies=[Depends(verified_user)])
def export_prediction(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    format: str = Query(),
    name: str = Query(),
    dataset: str = Query("all"),
    kind: str = Query("bert"),
) -> FileResponse:
    """
    Export prediction file (parquet/csv/xlsx). `kind` selects which manager
    owns the file: BERT classifications under `languagemodels`, NER span
    predictions under `nermodels`, quickmodel predictions on the whole
    dataset under `quickmodels`.
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        if kind == "ner":
            if project.nermodels is None:
                raise HTTPException(
                    status_code=400, detail="NER models are not available for this project"
                )
            return project.nermodels.export_prediction(
                name=name,
                file_name=f"predict_{dataset}.parquet",
                format=format,
                col_id=project.params.col_id,
            )
        if kind == "quick":
            return project.quickmodels.export_prediction_file(
                name=name,
                dataset=dataset,
                format=format,
                col_id=project.params.col_id,
            )
        return project.languagemodels.export_prediction(
            name=name,
            file_name=f"predict_{dataset}.parquet",
            format=format,
            col_id=project.params.col_id,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bert", dependencies=[Depends(verified_user)])
def export_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
) -> Response:
    """
    Export fine-tuned BERT model.

    With sqlite (no nginx), FastAPI streams the file directly.
    With postgres (nginx), the X-Accel-Redirect header is intercepted by nginx
    which serves the file from the static volume.
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        file_path = project.languagemodels.export_bert(name=name)
        if "sqlite" in config.database_url:
            absolute_path = Path(config.data_path) / "projects" / "static" / file_path.path
            return FileResponse(
                path=absolute_path,
                filename=f"{name}.tar.gz",
                media_type="application/octet-stream",
            )
        return FastAPIResponse(
            status_code=200,
            headers={
                "X-Accel-Redirect": f"/privatefiles/{file_path.path}",
                "Content-Disposition": f'attachment; filename="{name}.tar.gz"',
                "Content-Type": "application/octet-stream",
            },
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/export/raw",
    dependencies=[Depends(verified_user)],
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def export_raw(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> Response:
    """
    Export raw data of the project.

    With sqlite (no nginx), FastAPI streams the file directly.
    With postgres (nginx), the X-Accel-Redirect header is intercepted by nginx
    which serves the file from the static volume.
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        file_path = project.export_raw(project.name)
        if "sqlite" in config.database_url:
            absolute_path = Path(config.data_path) / "projects" / "static" / file_path.path
            return FileResponse(
                path=absolute_path,
                filename=f"{project.name}.parquet",
                media_type="application/octet-stream",
            )
        return FastAPIResponse(
            status_code=200,
            headers={
                "X-Accel-Redirect": f"/privatefiles/{file_path.path}",
                "Content-Disposition": f'attachment; filename="{project.name}.parquet"',
                "Content-Type": "application/octet-stream",
            },
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/prompts/similarity", dependencies=[Depends(verified_user)])
def export_prompt_similarity(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt_id: str = Query(),
    dataset: str = Query("all"),
    format: str = Query("csv"),
) -> FileResponse:
    """
    Export the similarity file computed for a prompt on a dataset
    (see /prompts/similarity/compute).
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    if project.prompts is None:
        raise HTTPException(status_code=400, detail="Prompts are not available on this project.")
    try:
        path, file_name = project.prompts.export_similarity(
            prompt_id=prompt_id,
            dataset=dataset,
            format=format,
            col_id=project.params.col_id,
        )
        return FileResponse(path=path, filename=file_name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/export/generations",
    dependencies=[Depends(verified_user)],
    responses={200: {"content": {"text/csv": {}}}},
)
def export_generations(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    params: ExportGenerationsParams,
) -> Response:
    """
    Export annotations
    """
    try:
        table = project.export_generations(
            project_slug=project.name,
            username=current_user.username,
            params=params,
        )

        # convert to payload
        output = StringIO()
        table.to_csv(output, index=True)
        csv_data = output.getvalue()
        output.close()
        headers = {
            "Content-Disposition": 'attachment; filename="data.csv"',
            "Content-Type": "text/csv",
        }

        return Response(content=csv_data, media_type="text/csv", headers=headers)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/topics", dependencies=[Depends(verified_user)])
def export_bertopics_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_topics(name=name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/clusters", dependencies=[Depends(verified_user)])
def export_bertopics_clusters(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_clusters(name=name, col_id=project.params.col_id)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/report", dependencies=[Depends(verified_user)])
def export_bertopics_report(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_report(name=name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/bertopic/embeddings", dependencies=[Depends(verified_user)])
def export_bertopics_embeddings(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> FileResponse:
    """
    Export annotations
    """
    test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
    try:
        return project.bertopic.export_embeddings(name=name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/export/prediction/quickmodel", dependencies=[Depends(verified_user)])
# def export_quickmodel_predictions(
#     project: Annotated[Project, Depends(get_project)],
#     current_user: Annotated[UserInDBModel, Depends(verified_user)],
#     name: str,
#     format: str = "csv",
# ) -> StreamingResponse:
#     """
#     Export prediction quickmodel for the project/user/scheme if any
#     """
#     test_rights(ProjectAction.EXPORT_DATA, current_user.username, project.name)
#     try:
#         output, headers = project.quickmodels.export_prediction(name, format)
#         return StreamingResponse(output, headers=headers)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
