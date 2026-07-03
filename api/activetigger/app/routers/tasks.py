import logging
import os
from typing import Annotated
from activetigger.app.dependencies import verified_user
from activetigger.datamodels import UserInDBModel
from celery import chain, chord, uuid
from fastapi import APIRouter, Depends, Body
from h11 import Request, Response
from task_manager import celery
from task_manager.tasks.callback import callback_task
from task_manager.tasks.hello_world import hello_world_task
from pydantic import BaseModel


router = APIRouter(tags=["tasks"])
logger = logging.getLogger("activetigger.fastapi.tasks")

class CallbackModel(BaseModel):
    task_id: str
    data: str

@router.post(
    "/tasks/done",
    # TODO: add a Task API key verification
    # dependencies=[Depends(verified_user)],
)
def task_callback(
   task_result: CallbackModel = Body(...)
) :
    """
    Callback
    """
    logger.info(f"task {task_result.task_id} succeed with result {task_result.data}")
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
    test task
    """
    logger.info(f"create task name {name}")
    # todo: API host and port in config
    #hello_world_task.s(name).apply_async()
    (hello_world_task.s(name) | callback_task.s(f"http://api:{os.environ.get('API_PORT', 4000)}/tasks/done/")).apply_async()
    return True