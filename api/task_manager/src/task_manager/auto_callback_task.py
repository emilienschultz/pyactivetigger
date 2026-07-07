from enum import StrEnum

from celery import Task

from task_manager.utils import TaskResultForCallback, task_success_callback


# TODO: should we declare queue in celery_app directly?
class QueueName(StrEnum):
    CPU="CPU"
    GPU="GPU"

class AutoCallbackTask(Task):
    """Base contract for tasks which needs to callback orchestrator.

    Subclasses must implement:
      - `run(**kwargs)` — the actual work (registered as the Celery task)
      - `on_success()` — send a callback request to orchestrator API
    """
  
    # Callback to orchestrator on task success
    def on_success(self, retval, task_id, args, kwargs):
        results:TaskResultForCallback = {'results':retval, 'task_name': self.name}  # ty:ignore[invalid-argument-type]
        task_success_callback(task_id, results)
        return super().on_success(retval, task_id, args, kwargs)