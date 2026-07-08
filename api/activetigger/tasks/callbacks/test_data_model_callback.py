
# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.test_data_model_task import TestDataModelTask, TestDataModelTaskResult

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class TestDataModelCallback(TaskCallback):

    # task_name is used to identify which callback to execute 
    task_name = TestDataModelTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete(cls, task_id:str, task_result:TestDataModelTaskResult):
        print(f"Completion of {cls.task_name} task {task_id} from orchestrator")
        print(f"processing completion of task {task_id} with result {task_result}")
