import pandas as pd

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.create_project_task import CreateProjectTask, CreateProjectTaskResult

from activetigger.datamodels import ProjectModel
from activetigger.orchestrator import get_orchestrator

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
        # create project model from the dump version
        project = ProjectModel(**task_result['project'])
        # load DataFrames from filesystem
        trainset_import = pd.read_parquet(task_result['import_trainset_path']) if task_result['import_trainset_path'] else None
        testset_import = pd.read_parquet(task_result['import_testset_path']) if task_result['import_testset_path'] else None
        validset_import = pd.read_parquet(task_result['import_validset_path']) if task_result['import_validset_path'] else None
        
        orchestrator = get_orchestrator()
        try:
            project_manager = orchestrator.project_creation_ongoing[project.project_slug]
            project_manager.finish_project_creation(
                task_result['username'],
                project,
                trainset_import,
                testset_import,
                validset_import
                )
        except KeyError:
            raise Exception(f"Project {project.project_slug} not listed in orchestrator, we can't finish creation process")

