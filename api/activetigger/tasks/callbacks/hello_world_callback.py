
# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.hello_world import HelloWorldTask, HelloWorldTaskResult

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class HelloWorldCallback(TaskCallback):

    # task_name is used to identify which callback to execute 
    task_name = HelloWorldTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete(cls, task_id:str, task_result:HelloWorldTaskResult):
        print(f"Completion of hello world task {task_id} from orchestrator")
        print(f"processing completion of task {task_id} with result {task_result}")
