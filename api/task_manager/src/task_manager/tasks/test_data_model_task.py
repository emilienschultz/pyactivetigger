from activetigger.datamodels import ChangeEmailModel
from celery import Task
from celery.utils.log import get_task_logger

from task_manager.auto_callback_task import AutoCallbackTask, QueueName
from task_manager.celery import celery_app

TestDataModelTaskResult = ChangeEmailModel



# Task definition using the auto callback generic parent task class
class TestDataModelTask(AutoCallbackTask):
    name = "Test data model"
    queue = QueueName.CPU
    
    def run(self:Task, input: ChangeEmailModel)->TestDataModelTaskResult:   
        logger = get_task_logger(TestDataModelTask.name)
        logger.info(f"Start change data model {input.email}")
        return input.model_dump()
    
# Task registration
@celery_app.task(bind=True, name=TestDataModelTask.name, queue=TestDataModelTask.queue, base=TestDataModelTask, pydantic=True)
def test_data_model_task(self, input:ChangeEmailModel):
    return TestDataModelTask.run(self=self, input=input)

