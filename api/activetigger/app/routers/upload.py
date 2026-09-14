from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from activetigger.app.dependencies import verified_user
from activetigger.datamodels import (
    UploadFinishedModel,
    UploadSessionModel,
    UploadStartModel,
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator
from activetigger.uploads import get_upload_staging

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/start")
def start_upload(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    params: UploadStartModel,
) -> UploadSessionModel:
    """
    Open a chunked-upload session. Large files are sent as a sequence of
    small requests so no single request outlives reverse-proxy timeouts.
    """
    # the declared size must fit in the user's storage quota
    users = get_orchestrator().users
    used_gb = users.get_storage(current_user.username)
    limit_gb = users.get_storage_limit(current_user.username)
    if used_gb + params.total_size / 1024**3 > limit_gb:
        raise HTTPException(
            status_code=413,
            detail=f"Upload would exceed your storage limit ({limit_gb} GB)",
        )

    upload_id, filename = get_upload_staging().start(
        current_user.username, params.filename, params.total_size, params.total_chunks
    )
    return UploadSessionModel(upload_id=upload_id, filename=filename)


@router.post("/chunk")
def upload_chunk(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    upload_id: str,
    index: int = Query(ge=0),
    file: UploadFile = File(...),
) -> None:
    """
    Store one chunk of a staged upload (idempotent per index, retryable).
    """
    get_upload_staging().write_chunk(current_user.username, upload_id, index, file.file)


@router.post("/finish")
def finish_upload(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    upload_id: str,
) -> UploadFinishedModel:
    """
    Assemble the chunks into the final staged file.
    """
    filename, size = get_upload_staging().finish(current_user.username, upload_id)
    return UploadFinishedModel(upload_id=upload_id, filename=filename, size=size)


@router.delete("/{upload_id}")
def cancel_upload(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    upload_id: str,
) -> None:
    """
    Cancel a staged upload and remove its data.
    """
    get_upload_staging().discard(current_user.username, upload_id)
