
from celery.utils.log import get_task_logger
import json
import requests
from task_manager.celery import celery_app



logger = get_task_logger(__name__)

@celery_app.task( bind=True, name="callback")
def callback_task(self, results, callback_url: str):
    task_id = self.request.root_id
    logger.info(f"Start callback {task_id} {callback_url}")
    payload = {"task_id": task_id, "data": results}
    logger.info(f"Callback payload {payload}")
    body = json.dumps(payload).encode()    
    headers = { "Content-Type": "application/json"}
    logger.info(f"Sending callback {callback_url} {headers} {body}")
    response = requests.post(callback_url, data=body, headers=headers)
    logger.info(f"Callback response: {response.status_code} - {response.text}")
