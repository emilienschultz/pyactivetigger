import logging

from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict, Json
from task_manager.tasks.hello_world import hello_world_task

from activetigger.tasks import all_callbacks


class CallbackModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    results: any  # ty:ignore[invalid-type-form]
    task_name: str


router = APIRouter(tags=["tasks"])
logger = logging.getLogger("activetigger.fastapi.tasks")


@router.post(
    "/tasks/done",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_callback(
   task_result: CallbackModel = Body(...)
) :
    """
    Task success callback 
    """
    logger.info(f"task {task_result.task_id} succeed with result {task_result.results} {task_result.task_name}")
    
    # check if a completion callback is available
    callback = all_callbacks.task_callbacks[task_result.task_name]
    if callback:
        logger.info(f"execute callback for task {task_result.task_name} {task_result.task_id}")
        callback(task_result.task_id, task_result.results)
    return True



@router.post(
    "/tasks/hello_world",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_hello_world(
    name: str
) :
    """
    test task to show how to execute task from orchestrator
    """
    logger.info(f"create task name {name}")
    hello_world_task.s(name).apply_async()
    return True