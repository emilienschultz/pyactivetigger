"""
Prompts router — prompt-based selection by cosine similarity against a bound
embedding feature (multimodal-embeddings on image projects, sentence-embeddings
on text projects). See docs/multimodal-prompt-selection.md.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import PromptInModel, PromptOutModel, UserInDBModel
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["prompts"])


def _require_prompts(project: Project):
    if project.prompts is None:
        raise HTTPException(
            status_code=400,
            detail="Prompts are not available on this project.",
        )
    return project.prompts


@router.post("/prompts/add", dependencies=[Depends(verified_user)])
def post_prompt(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt: PromptInModel,
) -> dict[str, str]:
    """Queue a prompt for embedding. Returns the task unique_id."""
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    prompts = _require_prompts(project)
    try:
        unique_id = prompts.add(prompt.text, prompt.feature_name, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"ADD PROMPT: {prompt.feature_name}",
            project.name,
        )
        return {"unique_id": unique_id}
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts/list", dependencies=[Depends(verified_user)])
def list_prompts(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    all_users: bool = Query(default=False),
) -> list[PromptOutModel]:
    prompts = _require_prompts(project)
    try:
        return prompts.list(user=None if all_users else current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/similarity/compute", dependencies=[Depends(verified_user)])
def compute_prompt_similarity(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt_id: str = Query(),
    dataset: str = Query("all"),
) -> dict[str, str | None]:
    """
    Compute the cosine similarity between a prompt and every element of a
    dataset, saved as an exportable file. For `dataset="all"` (the complete
    dataset) the bound embedding feature is recomputed by a queued task and
    the returned unique_id identifies it; for train/valid/test the file is
    written synchronously and unique_id is null.
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    prompts = _require_prompts(project)
    try:
        unique_id = prompts.compute_similarity(prompt_id, dataset, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            f"COMPUTE PROMPT SIMILARITY: {prompt_id} on {dataset}",
            project.name,
        )
        return {"unique_id": unique_id}
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/delete", dependencies=[Depends(verified_user)])
def delete_prompt(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt_id: str = Query(),
) -> None:
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    prompts = _require_prompts(project)
    try:
        prompts.delete(prompt_id)
        get_orchestrator().log_action(
            current_user.username, f"DELETE PROMPT: {prompt_id}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
