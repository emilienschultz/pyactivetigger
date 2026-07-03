
from celery.utils.log import get_task_logger
from task_manager.celery import celery_app




logger = get_task_logger(__name__)


@celery_app.task( bind=True, name="hello_world")
def hello_world_task(self, your_name: str | None):
    logger.info(f"Start hello world {your_name}")
    result = f"Hello {your_name if your_name else 'World'}!"
    return result