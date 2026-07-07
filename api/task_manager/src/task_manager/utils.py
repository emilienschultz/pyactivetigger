import json
import os
from typing import TypedDict
from celery.utils.log import get_task_logger
import requests


class TaskResultForCallback(TypedDict):
    results: any
    task_name: str
    
# trigger orchestrator callback
def task_success_callback(task_id:str, results:TaskResultForCallback):
    logger = get_task_logger("success_callback")
    
    callback_url=f"http://api:{os.environ.get('API_PORT', 4000)}/tasks/done/"    
    logger.info(f"Start callback {task_id} {callback_url}")
    # TODO: use pydantic
    payload = {"task_id":task_id, "results":results['results'], "task_name": results['task_name']}
    logger.info(f"Callback payload {payload}")
    body = json.dumps(payload).encode()    
    headers = { "Content-Type": "application/json"}
    logger.info(f"Sending callback {callback_url} {headers} {body}")
    response = requests.post(callback_url, data=body, headers=headers)
    logger.info(f"Callback response: {response.status_code} - {response.text}")