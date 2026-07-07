#
# Worker entry points
#


from task_manager import config
from task_manager.auto_callback_task import QueueName
from task_manager.celery import celery_app


def start_cpu_worker_pool():
    """
    Start a worker for the CPU queue
    """
    args = [
        "worker",
        f"--loglevel={config.worker_loglevel}",
        f"--concurrency={config.cpu_worker_concurrency}",
        f"--queues={QueueName.CPU}",
        "--hostname=worker_cpu@%h",
        # "--purge",
    ]
    celery_app.worker_main(args)


# TODO
# def runGpuWorker():
#     """
#     Start a worker for the GPU queue
#     """
#     args = [
#         "worker",
#         f"--loglevel={worker_loglevel}",
#         f"--concurrency={gpu_worker_concurrency}",
#         "-Psolo",
#         "--queues=GPU",
#         "--hostname=worker_gpu@%h",
#         # "--purge",
#     ]
#     celery.worker_main(args)