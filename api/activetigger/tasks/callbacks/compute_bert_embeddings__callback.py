from typing import cast

import pandas as pd

# use task specification to ensure name and return are in sync with task_manager
from task_manager.tasks.compute_bert_embeddings_task import (
    ComputeBertEmbeddingsTask,
    ComputeBertEmbeddingsTaskInput,
    ComputeBertEmbeddingsTaskResult,
)
from task_manager.utils import TaskFailureReportForCallback

from activetigger.datamodels import ProjectModel
from activetigger.orchestrator import get_orchestrator
from activetigger.tasks.compute_bert_embeddings import ComputeBertEmbeddings

# ensure the callback definition is what we expect by using the TaskCallback abstraction
from activetigger.tasks.task_callback import TaskCallback


class ComputeBertEmbeddingsCallback(TaskCallback):

    # task_name is used to identify which callback to execute 
    task_name = ComputeBertEmbeddingsTask.name

    # the on_complete method will be executed when a task succeeds
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_complete(cls, task_id:str, task_result:ComputeBertEmbeddingsTaskResult):
        print(f"Completion of Create Project task {task_id} in orchestrator {task_result}")
        
        # load DataFrame from filesystem
        results = pd.read_parquet(task_result['embeddings_path'])
        
        orchestrator = get_orchestrator()
        try:
            project_manager = orchestrator.project_creation_ongoing[task_result['project_slug']]
            project_manager.features.add(
                name="feature",
                kind="compute_bert_features",
                username= task_result['username'],
                parameters=cast(dict, task_result['parameters']),
                new_content=results,
            )
        except KeyError:
            raise Exception(f"Project {task_result['project_slug']} not listed in orchestrator, we can't finish creation process")

    # the on_failure method will be executed when a task fails
    # it will be executed from the orchestrator allowing using its dependencies (db and all)
    @classmethod
    def on_failure(cls, task_id:str, task_report:TaskFailureReportForCallback):
        print(f"Error on Create Project task {task_id} in orchestrator {task_report}")
        # cast first args as Task Input type
        orchestrator = get_orchestrator()
        try:
            task_input = ComputeBertEmbeddingsTaskInput(**task_report['task_args'][0])
            project_manager = orchestrator.project_creation_ongoing[task_input.project_slug]
            project_manager.status = 'error'
            # TODO: this message is generic on any GPU task
            message = (
                        f"Error for task {ComputeBertEmbeddingsTask.name} : GPU error — not enough GPU memory available. "
                        "Try reducing the batch size, the max sequence length, or using a smaller model. "
                        f"Details: {task_report['exception']}"
                    ) if any(
                    s in task_report['exception']
                    for s in [
                        "CUDA",
                        "CUDACachingAllocator",
                        "out of memory",
                        "NVML",
                        "cuda",
                    ]
                ) else f"Error for process {ComputeBertEmbeddingsTask.name} : {task_report['exception']}"
            project_manager.errors.add(message)
        except KeyError:
            raise Exception(f"Project {task_input.project_slug} not listed in orchestrator, we can't finish creation process")

