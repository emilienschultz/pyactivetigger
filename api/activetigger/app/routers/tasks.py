import logging

from fastapi import APIRouter, Body
from pydantic import BaseModel
from task_manager.celery import tasks_callbacks
from task_manager.tasks.hello_world import hello_world_task


class CallbackModel(BaseModel):
    task_id: str
    results: str
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
    callback = tasks_callbacks[task_result.task_name]
    if callback:
        callback(task_result);
    return True



@router.post(
    "/tasks/hello_world",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_callback(
    name: str
) :
    """
    test task to show how to execute task from orchestrator
    """
    logger.info(f"create task name {name}")
    hello_world_task.s(name).apply_async()
    return True