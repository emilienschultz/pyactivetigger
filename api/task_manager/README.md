# Active Tigger Task manager

This package encapsulate tasks which are created by orchestrator but managed by Celery into pools of workers.

## How to create a new task?

### 1. create the task run method in task_manager

Tasks must be created in the `api/task_manager/src/task_manager/tasks` folder.

If it's a simple unitary task which will send one callback at success, write your task as a class deriving from `AutoCallbackTask`. In that case you just have to write:

- the task result type (must be JSON serializable) most likely as a `TypedDict`
- the task class which lists:
  - the task name
  - the worker queue
  - the run method
- don't forget to register the task into the celery_app

### 2. add a TaskCallback in orchestrator

To allow Orchestrator to pick up and manage the task result, create a TaskCallback class in `api/activetigger/tasks/callbaks/` folder.

The callback method `on_complete` must be wrapped into a class derived from `TaskCallback`.

Import the task class you created in task_manager to make sure you're using the right task name and task result type.

### 3. add task creation into orchestrator controllers

To use the task in Orchestrator code:

- import the decorated task method `from task_manager.tasks.hello_world_task import hello_world_task`
- then use it as a normal celery task `hello_world_task.s(name).apply_async()`

To specify complex input/results you can use Pydantic data models but with two conditions:

. Make sure to add `pydantic=True` in the `celery_app.task` decorator arguments
. Use `model_dump()` both when passing input to the task and when returning results

## Notes

Current Problems to solve:

1. share data model between tasks and orchestrator (multi-package with common package? remove workspace ?)
   => test_data_mode_task shows taht importing from activetigger.datamodels into task_manager works
2. make createProject Input model serializable: POSIX path are not.

Next ones:

- write data in common data folder
- dynamically change number of worker
