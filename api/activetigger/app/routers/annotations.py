from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.datamodels import (
    ActionModel,
    ActiveModel,
    AnnotationModel,
    AnnotationsDataModel,
    ElementInModel,
    ElementOutModel,
    NextInModel,
    ProjectionOutModel,
    ProjectionParametersModel,
    ReconciliateElementInModel,
    ReconciliationModel,
    TableAnnotationsModel,
    TableBatchInModel,
    TableOutModel,
    UserInDBModel,
    WaitingModel,
)
from activetigger.errors import APIError, InvalidInputError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project
from activetigger.uploads import get_upload_staging

router = APIRouter(tags=["annotations"])


@router.post("/elements/next", dependencies=[Depends(verified_user)])
def get_next(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    next: NextInModel,
) -> ElementOutModel:
    """
    Get next element
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        return project.get_next(
            next=next.model_copy(update={"n": 1}),
            username=current_user.username,
        )[0]
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/next/batch", dependencies=[Depends(verified_user)])
def get_next_batch(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    next: NextInModel,
) -> list[ElementOutModel]:
    """
    Get the next n elements to annotate in one call.
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        return project.get_next(
            next=next,
            username=current_user.username,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/elements/projection", dependencies=[Depends(verified_user)])
def get_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str = Query(),
    projection_name: str = Query(),
    model_name: str | None = Query(None),
    model_type: str | None = Query(None),
) -> ProjectionOutModel | None:
    """
    Get a named projection if computed
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        active_model: ActiveModel | None = None
        if model_name is not None and model_type is not None:
            active_model = ActiveModel(type=model_type, value=model_name, label=model_name)
        return project.get_projection(
            projection_name=projection_name, scheme=scheme, active_model=active_model
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/projection/compute", dependencies=[Depends(verified_user)])
def compute_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    projection: ProjectionParametersModel,
) -> WaitingModel:
    """
    Start projection computation using futures.
    Multiple named projections can coexist per project.
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    if not projection.name:
        raise HTTPException(status_code=400, detail="A projection name is required")
    if len(projection.features) == 0:
        raise HTTPException(status_code=400, detail="No features available")
    try:
        features = project.features.get(projection.features, dataset=["train"])
        project.projections.compute(
            project_slug=project.name,
            username=current_user.username,
            projection=projection,
            features=features,
            normalize_features=projection.normalize_features,
        )
        get_orchestrator().log_action(
            current_user.username,
            f"COMPUTE PROJECTION: {projection.name} ({projection.method})",
            project.project_slug,
        )
        return WaitingModel(detail=f"Projection {projection.name} is computing")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/projection/delete", dependencies=[Depends(verified_user)])
def delete_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    projection_name: str = Query(),
) -> WaitingModel:
    """
    Delete a named projection.
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.projections.delete(projection_name)
        get_orchestrator().log_action(
            current_user.username,
            f"DELETE PROJECTION: {projection_name}",
            project.project_slug,
        )
        return WaitingModel(detail=f"Projection {projection_name} deleted")
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/table", dependencies=[Depends(verified_user)])
def get_list_elements(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    batch: TableBatchInModel,
) -> TableOutModel:
    """
    Get a table of elements
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        return project.schemes.get_table(batch)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/annotation/table", dependencies=[Depends(verified_user)])
def post_list_elements(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    table: TableAnnotationsModel,
) -> None:
    """
    Update a table of annotations
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        errors = project.schemes.push_annotations_table(table, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"UPDATE BATCH ANNOTATIONS in project {project.name} N={len(table.annotations)} annotations ({len(errors or [])} errors)",
            project.name,
        )
        if errors is not None:
            raise InvalidInputError(f"Errors during annotations update: {errors}")
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/annotation/file", dependencies=[Depends(verified_user)])
def post_annotation_file(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    annotationsdata: AnnotationsDataModel,
) -> None:
    """
    Load annotations from a staged upload (chunked-upload protocol,
    see activetigger.uploads) referenced by annotationsdata.upload_id.
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    staging = get_upload_staging()
    staged_path, _ = staging.claim(current_user.username, annotationsdata.upload_id, (".csv",))
    try:
        project.schemes.add_file_annotations(
            annotationsdata=annotationsdata,
            file_path=staged_path,
            user=current_user.username,
        )
        get_orchestrator().log_action(
            current_user.username,
            f"LOAD ANNOTATION FROM FILE: scheme {annotationsdata.scheme}",
            project.name,
        )
        # consumed only on success so a failed import can be retried
        staging.discard(current_user.username, annotationsdata.upload_id)
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/id", dependencies=[Depends(verified_user)])
def get_element(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    element: ElementInModel,
) -> ElementOutModel:
    """
    Get specific element by id
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        return project.get_element(
            element=element,
            user=current_user.username,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/annotation/reconciliate", dependencies=[Depends(verified_user)])
def get_reconciliation_table(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str = Query(),
    dataset: str = Query("train"),
) -> ReconciliationModel:
    """
    Get the reconciliation table
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        df, users, agreement_stats = project.schemes.get_reconciliation_table(scheme, dataset)
        return ReconciliationModel(
            table=cast(
                list[dict[str, str | dict[str, str | None] | None]], df.to_dict(orient="records")
            ),
            users=users,
            **agreement_stats,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/annotation/reconciliate", dependencies=[Depends(verified_user)])
def post_reconciliation(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    element: ReconciliateElementInModel,
) -> None:
    """
    Post a label for all user in a list
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        project.schemes.reconciliate_element(element, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"RECONCILIATE ANNOTATION: in {element.scheme} element {element.element_id} as {element.label}",
            project.name,
        )
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/annotation/{action}", dependencies=[Depends(verified_user)])
def post_annotation(
    action: ActionModel,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    annotation: AnnotationModel,
) -> None:
    """
    Add, Update, Delete annotations
    Comment :
    - For the moment add == update
    - No information kept of selection process
    """

    if action in ["add", "update"]:
        if action == "add":
            test_rights(ProjectAction.ADD_ANNOTATION, current_user.username, project.name)
        else:
            test_rights(ProjectAction.UPDATE_ANNOTATION, current_user.username, project.name)
        try:
            project.schemes.push_annotation(
                annotation.element_id,
                annotation.label,
                annotation.scheme,
                current_user.username,
                annotation.dataset,
                annotation.comment,
                annotation.selection,
            )

            get_orchestrator().log_action(
                current_user.username,
                f"ADD ANNOTATION: in {annotation.scheme} element {annotation.element_id} as {annotation.label} ({annotation.dataset})",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    if action == "delete":
        test_rights(ProjectAction.UPDATE_ANNOTATION, current_user.username, project.name)
        try:
            project.schemes.delete_annotation(
                annotation.element_id,
                annotation.scheme,
                annotation.dataset,
                current_user.username,
            )

            get_orchestrator().log_action(
                current_user.username,
                f"DELETE ANNOTATION: in {annotation.scheme} id {annotation.element_id}",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    raise HTTPException(status_code=400, detail="Wrong action")
