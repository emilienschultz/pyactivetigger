
# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.create_project_task import CreateProjectTask, CreateProjectTaskResult

from activetigger.project import Project

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class CreateProjectCallback(TaskCallback):

    # task_name is used to identify which callback to execute 
    task_name = CreateProjectTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete(cls, task_id:str, task_result:CreateProjectTaskResult):
        print(f"Completion of Create Project task {task_id} in orchestrator {task_result}")
        # project = Project()
        # load DataFrame from filesystem
        
        #project.finish_project_creation(username, project, )

