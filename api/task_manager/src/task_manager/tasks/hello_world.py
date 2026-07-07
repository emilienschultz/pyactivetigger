from celery import Task
from celery.utils.log import get_task_logger

from task_manager.celery import celery_app
from task_manager.ManagedTask import QueueName, ManagedTask


# Task definition encapsulating all Task logic including the on_complete callback
class HelloWorldTask(ManagedTask):
    name = "hello world"
    queue = QueueName.CPU
    
    @staticmethod
    def run(self:Task, your_name: str | None):   
        logger = get_task_logger(HelloWorldTask.name)
        logger.info(f"Start hello world {your_name}")
        results = f"Hello {your_name if your_name else 'World'}!"
        return results
    
    @staticmethod
    def on_complete(task_result):
        # code executed from orchestrator
        print(f"log from on complete: {task_result}")
        print(f"processing completion of task {task_result.task_id} with result {task_result.results}")

# Task registration
@celery_app.task(bind=True, name=HelloWorldTask.name, queue=HelloWorldTask.queue, base=HelloWorldTask)
def hello_world_task(self, your_name:str):
    return HelloWorldTask.run(self=self, your_name=your_name)

