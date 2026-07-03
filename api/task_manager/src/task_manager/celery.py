import os
from celery import Celery
from task_manager.config import redis_url

celery_app = Celery('activetigger',
             broker=redis_url,
             backend=redis_url)

# Task discovery: automatically discover tasks in the "tasks" folder
tasks_path = os.path.join(os.path.dirname(__file__), "tasks")
tasks_files = [
    f"task_manager.tasks.{f}"
    for f in os.listdir(tasks_path)
    if os.path.isfile(os.path.join(tasks_path, f))
]
celery_app.autodiscover_tasks(tasks_files)

# Optional configuration, see the application user guide.
celery_app.conf.update(
    task_default_queue="CPU",
    result_expires=3600,
)

if __name__ == '__main__':
    celery_app.start()