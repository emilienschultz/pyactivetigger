from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
)

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    ExportGenerationsParams,
    GenerationCreationModel,
    GenerationModel,
    GenerationModelApi,
    GenerationRequest,
    PromptInputModel,
    PromptModel,
    TableOutModel,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.generation.generations import Generations
from activetigger.generation.ollama import Ollama
from activetigger.generation.openapi import OpenAPI
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["generation"])


@router.get("/generate/models/available")
def list_generation_models() -> list[GenerationModelApi]:
    """
    Returns the list of the available GenAI models for generation
    API (not the models themselves)
    """
    try:
        return Generations.get_available_models()
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/ollama/models")
def list_ollama_models(endpoint: str) -> list[dict[str, str]]:
    """
    Query an Ollama server endpoint to list available models
    """
    try:
        return Ollama.list_models(endpoint)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/openai/models", dependencies=[Depends(verified_user)])
def list_openai_compatible_models(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    endpoint: str | None = None,
    credentials: str | None = None,
    saved_credentials: str | None = None,
) -> list[dict[str, str]]:
    """
    Query an OpenAI-compatible endpoint to list available models via /v1/models.
    If saved_credentials is given, the endpoint/secret saved in the user account are used.
    """
    try:
        if saved_credentials is not None:
            saved_endpoint, credentials = get_orchestrator().users.resolve_credentials(
                current_user.username, saved_credentials
            )
            endpoint = endpoint or saved_endpoint
        if endpoint is None:
            raise Exception("You should provide an endpoint")
        return OpenAPI.list_models(endpoint, credentials)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/models", dependencies=[Depends(verified_user)])
def list_project_generation_models(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
) -> list[GenerationModel]:
    """
    Returns the list of the available GenAI models configure for a project
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        return project.generations.available_models(project.name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/models", dependencies=[Depends(verified_user)])
def add_project_generation_models(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model: GenerationCreationModel,
) -> int:
    """
    Add a new GenAI model for the project
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        # resolve credentials saved in the user account, so the secret never transits by the client
        if model.saved_credentials is not None:
            endpoint, credentials = get_orchestrator().users.resolve_credentials(
                current_user.username, model.saved_credentials
            )
            model.endpoint = model.endpoint or endpoint
            model.credentials = credentials
        return project.generations.add_model(project.name, model, current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/generate/models/{model_id}",
    dependencies=[Depends(verified_user)],
)
def delete_project_generation_models(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_id: int = Path(ge=0),
) -> None:
    """
    Delete a GenAI model from the project
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.generations.delete_model(project.name, model_id)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/start", dependencies=[Depends(verified_user)])
def postgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenerationRequest,
) -> None:
    """
    Launch a call to generate from a prompt
    """

    try:
        project.generations.check_prompts(request.prompt, project.params.cols_context)
        project.start_generation(request, current_user.username)
        get_orchestrator().log_action(
            current_user.username,
            "START GENERATE",
            project.params.project_slug,
        )
        return None

    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/elements", dependencies=[Depends(verified_user)])
def getgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    params: ExportGenerationsParams,
) -> TableOutModel:
    """
    Get elements generated
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        table = project.get_generated(project.name, current_user.username, params)
        return TableOutModel(items=table.to_dict(orient="records"), total=len(table))
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error in loading generated data" + str(e))


@router.post("/generate/elements/drop", dependencies=[Depends(verified_user)])
def dropgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Drop all elements from prediction for a user
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        project.generations.drop_generated(project.name, current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/prompts", dependencies=[Depends(verified_user)])
def get_prompts(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[PromptModel]:
    """
    Get the list of prompts for the project
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        return project.generations.get_prompts(project.name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/prompts/add", dependencies=[Depends(verified_user)])
def add_prompt(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt: PromptInputModel,
) -> None:
    """
    Add a prompt to the project
    """
    test_rights(ProjectAction.GENERATE, current_user.username, project.name)
    try:
        project.generations.save_prompt(prompt, current_user.username, project.name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/prompts/delete", dependencies=[Depends(verified_user)])
def delete_prompt(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt_id: str,
) -> None:
    """
    Delete a prompt from the project
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.generations.delete_prompt(int(prompt_id))
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
