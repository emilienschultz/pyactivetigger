from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    test_rights,
    verified_user,
)
from activetigger.datamodels import MessagesInModel, MessagesOutModel, UserInDBModel
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator

router = APIRouter(tags=["messages"])


@router.get("/messages")
def get_messages(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    kind: Literal["system", "project", "user"],
    from_user: str | None = None,
    for_user: str | None = None,
    for_project: str | None = None,
) -> list[MessagesOutModel]:
    """
    Get messages
    - all if root
    - only for oneself
    """
    try:
        orchestrator = get_orchestrator()
        if kind == "project" and current_user.username != "root":
            test_rights(ProjectAction.GET, current_user.username, for_project)
        if current_user.username == "root":
            return orchestrator.messages.get_messages(kind, from_user, for_user, for_project)
        else:
            return orchestrator.messages.get_messages(
                kind, from_user, current_user.username, for_project
            )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/messages/inbox")
def get_inbox(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[MessagesOutModel]:
    """
    Personal inbox: DMs + project-distribution copies addressed to the caller.
    """
    try:
        return get_orchestrator().messages.get_inbox(current_user.username)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/messages/codebook")
def get_codebook_messages(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> list[MessagesOutModel]:
    """
    Project-distribution copies still owned by the caller for a given project.
    Used by the Codebook page to surface project messages.
    """
    try:
        return get_orchestrator().messages.get_codebook_messages(
            current_user.username, project_slug
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/messages")
def post_message(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    message: MessagesInModel,
) -> None:
    """
    Post a new message. Branches by `message.kind`:
    - "system": admin-only, single row (existing behavior).
    - "user":   any verified user can DM another user they share a project with.
                One row inserted with for_user=recipient.
    - "project": any member of the target project can post. One row inserted
                per project member (fanout); each recipient owns their copy.
    """
    orchestrator = get_orchestrator()
    sender = current_user.username

    if message.kind == "system":
        test_rights(ServerAction.MANAGE_SERVER, sender)
        try:
            orchestrator.messages.add_message(
                user_name=sender, kind="system", content=message.content
            )
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return

    if message.kind == "user":
        if not message.for_user:
            raise HTTPException(status_code=400, detail="for_user is required for a DM")
        if message.for_user == sender:
            raise HTTPException(status_code=400, detail="Cannot send a message to yourself")
        try:
            orchestrator.users.get_user(message.for_user)
        except Exception as e:
            raise HTTPException(status_code=404, detail="Recipient not found") from e
        try:
            orchestrator.messages.add_messages_bulk(
                user_name=sender,
                kind="user",
                content=message.content,
                recipients=[message.for_user],
            )
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return

    if message.kind == "project":
        if not message.for_project:
            raise HTTPException(
                status_code=400, detail="for_project is required for a project message"
            )
        try:
            members = list(orchestrator.users.get_project_auth(message.for_project).keys())
        except Exception as e:
            raise HTTPException(status_code=404, detail="Project not found") from e
        if sender != "root" and sender not in members:
            raise HTTPException(status_code=403, detail="You are not a member of this project")
        try:
            orchestrator.messages.add_messages_bulk(
                user_name=sender,
                kind="project",
                content=message.content,
                recipients=members,
                for_project=message.for_project,
            )
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return

    if message.kind == "all":
        # Root-only broadcast: fan out one inbox row per active user.
        test_rights(ServerAction.MANAGE_SERVER, sender)
        recipients = [u for u in orchestrator.users.existing_users().keys() if u != sender]
        try:
            orchestrator.messages.add_messages_bulk(
                user_name=sender,
                kind="user",
                content=message.content,
                recipients=recipients,
            )
        except (HTTPException, APIError, OverflowError):
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return

    raise HTTPException(status_code=400, detail=f"Unknown message kind: {message.kind}")


@router.post("/messages/delete")
def delete_message(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    message_id: int = Query(ge=0),
) -> None:
    """
    Delete a message.
    - root: can delete any message (admin path for system messages).
    - other users: can delete only rows addressed to them (for_user == self).
    """
    orchestrator = get_orchestrator()
    try:
        if current_user.username == "root":
            orchestrator.messages.delete_message(message_id)
            return
        ok = orchestrator.messages.delete_message_for_user(message_id, current_user.username)
        if not ok:
            raise HTTPException(
                status_code=403, detail="You can only delete messages addressed to you"
            )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
