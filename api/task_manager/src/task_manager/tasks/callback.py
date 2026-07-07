

from celery import Task
from celery.utils.log import get_task_logger


from task_manager.celery import celery_app
from task_manager.utils import TaskResultForCallback, task_success_callback




logger = get_task_logger(__name__)

@celery_app.task( bind=True, name="callback")
def callback_task(self:Task, results:TaskResultForCallback):
    task_id = self.request.parent_id
    task_success_callback(task_id=task_id, results=results)