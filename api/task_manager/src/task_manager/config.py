import os

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
worker_loglevel = os.environ.get("WORKER_LOGLEVEL", "DEBUG")
storage_path = os.environ.get("STORAGE_PATH", "/tmp")
cache_model_path = os.environ.get("TASKMANAGER_CACHE_MODEL_PATH", "/tmp")
cpu_worker_concurrency = os.environ.get("N_WORKERS_CPU", 1)
