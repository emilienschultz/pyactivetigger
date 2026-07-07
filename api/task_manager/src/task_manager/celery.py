
import os
from typing import Callable, Dict
from celery import Celery
from task_manager.ManagedTask import discover_tasks
from task_manager.config import redis_url

celery_app = Celery('activetigger',
             broker=redis_url,
             backend=redis_url)

# create a dict of on_complete callbacks
tasks_path = os.path.join(os.path.dirname(__file__), "tasks")
_tasks_classes = discover_tasks(tasks_path)
tasks_callbacks:Dict[str, Callable] = {}
for Task_class in _tasks_classes:
    tasks_callbacks[Task_class.name] = Task_class.on_complete
 
# Optional configuration, see the application user guide.
celery_app.conf.update(
    task_default_queue="CPU",
    result_expires=3600,
)

if __name__ == '__main__':
    celery_app.start()