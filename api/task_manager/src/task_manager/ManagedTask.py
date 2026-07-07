from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, List, TypedDict
from celery import Task
from task_manager.utils import task_success_callback

# TODO: should we declare queue in celery_app directly?
class QueueName(StrEnum):
    CPU="CPU"
    GPU="GPU"

class ManagedTask(Task):
    """Base contract for all tasks in the system.

    Subclasses must implement:
      - `run(**kwargs)` — the actual work (registered as the Celery task)
      - `on_complete(result, **kwargs)` — callback invoked by the orchestrator
    """
  
    # Callback to orchestrator on task success
    def on_success(self, retval, task_id, args, kwargs):
        task_success_callback(task_id, {'results':retval, 'task_name': self.name})
        return super().on_success(retval, task_id, args, kwargs)

    @classmethod
    @abstractmethod
    def on_complete(result: Any, **kwargs) -> Any:
        """Callback called by the orchestrator once the task succeeds."""
        raise NotImplementedError
    
    
# discover all ManagedTask implementations to collect on_complete methods
def discover_tasks(directory_path: str):
    discovered_tasks:List[ManagedTask] = []
    for file in Path(directory_path).glob("*.py"):
        if file.name.startswith("_"): continue
        
        module_name = file.stem
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ManagedTask) and obj is not ManagedTask:
                discovered_tasks.append(obj)
    return discovered_tasks