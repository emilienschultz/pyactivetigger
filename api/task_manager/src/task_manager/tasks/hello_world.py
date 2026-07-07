from typing import TypedDict

from celery import Task
from celery.utils.log import get_task_logger

from task_manager.auto_callback_task import AutoCallbackTask, QueueName
from task_manager.celery import celery_app


class HelloWorldTaskResult(TypedDict):
    sentence:str

# Task definition using the auto callback generic parent task class
class HelloWorldTask(AutoCallbackTask):
    name = "hello world"
    queue = QueueName.CPU
    
    def run(self:Task, your_name: str | None)->HelloWorldTaskResult:   
        logger = get_task_logger(HelloWorldTask.name)
        logger.info(f"Start hello world {your_name}")
        results:HelloWorldTaskResult = {'sentence':f"Hello {your_name if your_name else 'World'}!"}
        return results
    
# Task registration
@celery_app.task(bind=True, name=HelloWorldTask.name, queue=HelloWorldTask.queue, base=HelloWorldTask)
def hello_world_task(self, your_name:str):
    return HelloWorldTask.run(self=self, your_name=your_name)

