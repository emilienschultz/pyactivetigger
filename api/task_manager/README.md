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

To allow Orchestrator to pick up and manage the task result, create a TaskCallback class in `api/activetigger/tasks/callbacks/` folder.

The callback methods `on_complete` and `on_failure` must be wrapped into a class derived from `TaskCallback`.

Import the task class you created in task_manager to make sure you're using the right task name and task result type.

### 3. add task creation into orchestrator controllers

To use the task in Orchestrator code:

- import the decorated task method `from task_manager.tasks.hello_world_task import hello_world_task`
- then use it as a normal celery task `hello_world_task.s(name).apply_async()`

To specify complex input/results you can use Pydantic data models but with two conditions:

. Make sure to add `pydantic=True` in the `celery_app.task` decorator arguments
. Use `model_dump()` both when passing input to the task and when returning results

## Task monitoring

To monitor tasks from Celery, there are some caveats:

- to know ongoing tasks we must inspect workers state which is slow and does not give pending tasks state
- query the broker (redis) but that will only give pending tasks
- lookup a task if we know its ID but that means we need to store all tasks ID in activetigger
- monitor all tasks events and store a task state in activetigger

If there's any interest in keeping a log of created tasks by project (Celery will forget tasks history), then we should create that log through live monitoring and use that to track current state.

If not, I would still keep a list of current tasks ID in orchestrator memory through event monitoring rather than combining inspect and redis queries.

Monitoring events could remove the need for the API task callback routes. But doing so would open a risk of missing a task output. If something bad happens in activetigger before the end of a task, the end signal (either success of failure) would not be caught and we would lose this information. By using callback, the callback triggered by the task itself would fail and thus mark the task as failed.

## Notes

TODO:

- task monitoring
- add common data folder configurable to allow distant task manager with mounting point ?
- dynamically change number of worker
