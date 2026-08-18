import logging

from fastapi import APIRouter, Body
from pydantic import BaseModel, ConfigDict, Json
from task_manager.tasks.hello_world_task import hello_world_task
from task_manager.tasks.test_data_model_task import test_data_model_task

from activetigger.datamodels import ChangeEmailModel
from activetigger.tasks import all_callbacks


class GenericCallbackModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task_id: str
    task_name: str

class SuccessCallbackModel(GenericCallbackModel):
    results: any  # ty:ignore[invalid-type-form]

class FailureCallbackModel(GenericCallbackModel):
    report: any  # ty:ignore[invalid-type-form]


router = APIRouter(tags=["tasks"])
logger = logging.getLogger("activetigger.fastapi.tasks")


@router.post(
    "/tasks/done",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_success_callback(
   body: SuccessCallbackModel = Body(...)
) :
    """
    Task success callback 
    """
    logger.info(f"task {body.task_id} succeed with result {body.results} {body.task_name}")
    
    # check if a completion callback is available
    callback = all_callbacks.task_callbacks[body.task_name]
    if callback:
        logger.info(f"execute callback for task {body.task_name} {body.task_id}")
        callback.on_complete(body.task_id, body.results)
    return True

@router.post(
    "/tasks/failed",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_failure_callback(
   body: FailureCallbackModel = Body(...)
) :
    """
    Task success callback 
    """
    logger.info(f"task {body.task_id} failed with report {body.report} {body.task_name}")
    
    # check if a completion callback is available
    callback = all_callbacks.task_callbacks[body.task_name]
    if callback:
        logger.info(f"execute callback for task {body.task_name} {body.task_id}")
        callback.on_failure(body.task_id, body.report)
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




@router.post(
    "/tasks/test_data_model",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_test_data_model(
    email: str
) :
    """
    test task to show how to execute task from orchestrator
    """
    logger.info("create task name test data model")
    test_data_model_task.s(ChangeEmailModel(email=email, password="changeme").model_dump()).apply_async()
    return True