import os
from typing import Callable

from activetigger.tasks.task_callback import TaskCallback, discover_task_callbacks

# task_callbacks collect all implemented on_complete callbacks found in `callbacks` folder
# we use the abstract class TaskCallback to identify those
tasks_path = os.path.join(os.path.dirname(__file__), "callbacks")
_tasks_classes = discover_task_callbacks(tasks_path)

task_callbacks:dict[str, TaskCallback] = {}
for Task_class in _tasks_classes:
    # task_callbacks are index by task names which is used by the API to pick up the right on_complete
    if Task_class.task_name in task_callbacks:
        raise Exception(f"A on_complete callback ({task_callbacks[Task_class.task_name]}) is already implemented for task {Task_class.task_name}. We can not use the one from {Task_class}")
    task_callbacks[Task_class.task_name] = Task_class
