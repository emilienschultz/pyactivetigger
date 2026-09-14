from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.datamodels import (
    ActionModel,
    CodebookModel,
    CompareSchemesModel,
    SchemeModel,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter()


@router.post("/schemes/label/rename", dependencies=[Depends(verified_user)])
def rename_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    former_label: str,
    new_label: str,
) -> None:
    """
    Rename a a label
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.project_slug)

    try:
        project.schemes.rename_label(former_label, new_label, scheme, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"RENAME LABEL: scheme:{scheme} before:{former_label} after:{new_label}",
            project.name,
        )
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemes/label/{action}", dependencies=[Depends(verified_user)])
def add_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    action: ActionModel,
    scheme: str,
    label: str,
) -> None:
    """
    Add a label to a scheme
    """
    if action == "add":
        test_rights(ProjectAction.ADD, current_user.username, project.name)
        try:
            project.schemes.add_label(label, scheme, current_user.username)

            get_orchestrator().log_action(
                current_user.username,
                f"ADD LABEL: scheme:{scheme} label:{label}",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if action == "delete":
        test_rights(ProjectAction.DELETE, current_user.username, project.name)
        try:
            project.schemes.delete_label(label, scheme, current_user.username)
            get_orchestrator().log_action(
                current_user.username,
                f"DELETE LABEL: scheme:{scheme} label:{label}",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=400, detail="Wrong action")


@router.post("/schemes/codebook", dependencies=[Depends(verified_user)])
def post_codebook(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    codebook: CodebookModel,
) -> None:
    """
    Add codebook
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)

    try:
        project.schemes.add_codebook(codebook.scheme, codebook.content, codebook.time)
        get_orchestrator().log_action(
            current_user.username,
            f"MODIFY CODEBOOK: scheme {codebook.scheme}",
            project.name,
        )
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemes/codebook", dependencies=[Depends(verified_user)])
def get_codebook(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
) -> CodebookModel:
    """
    Get the codebook of a scheme for a project
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        return project.schemes.get_codebook(scheme)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemes/rename", dependencies=[Depends(verified_user)])
def rename_scheme(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    old_name: str,
    new_name: str,
) -> None:
    """
    Rename a scheme
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.schemes.rename_scheme(old_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"RENAME SCHEME: {old_name} to {new_name}",
            project.name,
        )
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemes/duplicate", dependencies=[Depends(verified_user)])
def duplicate_scheme(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    scheme_name: str,
) -> None:
    """
    Duplicate a scheme
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        project.schemes.duplicate_scheme(scheme_name, scheme_name + "_copy", current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"DUPLICATE SCHEME: {scheme_name}",
            project.name,
        )
        return None
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemes/compare", dependencies=[Depends(verified_user)])
def compare_schemes(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    schemeA: str,
    schemeB: str,
    dataset: str,
) -> CompareSchemesModel:
    """
    Compare two schemes
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        return project.schemes.compare(schemeA, schemeB, dataset)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemes/{action}", dependencies=[Depends(verified_user)])
def post_schemes(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    action: ActionModel,
    scheme: SchemeModel,
) -> None:
    """
    Add, Update or Delete scheme
    """
    orchestrator = get_orchestrator()
    if action == "add":
        test_rights(ProjectAction.ADD, current_user.username, project.name)
        try:
            project.schemes.add_scheme(
                scheme.name, scheme.labels, scheme.kind, current_user.username
            )
            orchestrator.log_action(
                current_user.username,
                f"ADD SCHEME: {scheme.name}",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    if action == "delete":
        test_rights(ProjectAction.DELETE, current_user.username, project.name)
        try:
            project.schemes.delete_scheme(scheme.name)
            orchestrator.log_action(
                current_user.username,
                f"DELETE SCHEME: {scheme.name}",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if action == "update":
        test_rights(ProjectAction.UPDATE, current_user.username, project.name)
        try:
            project.schemes.update_scheme(scheme.name, scheme.labels)
            orchestrator.log_action(
                current_user.username,
                f"UPDATE SCHEME: {scheme.name}",
                project.name,
            )
            return None
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Wrong route")
