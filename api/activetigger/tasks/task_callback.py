import inspect
from abc import ABC, abstractmethod
from importlib import util
from pathlib import Path
from typing import Any, List


# TaskCallback is a generic class to specify task callback definitions
class TaskCallback(ABC):
    
    task_name:str
  
    @classmethod
    @abstractmethod
    def on_complete(cls, task_id:str, task_result: Any) -> Any:
        """Callback called by the orchestrator once the task succeeds."""
        raise NotImplementedError
    
    
# discover all TaskCallback implementations to collect on_complete methods
def discover_task_callbacks(directory_path: str):
    discovered_tasks:List[TaskCallback] = []
    for file in Path(directory_path).glob("*.py"):
        if file.name.startswith("_"): 
            continue
        
        module_name = file.stem
        spec = util.spec_from_file_location(module_name, file)
        if spec and spec.loader:
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, TaskCallback) and obj is not TaskCallback:
                    discovered_tasks.append(obj)  # ty:ignore[invalid-argument-type]
    return discovered_tasks
