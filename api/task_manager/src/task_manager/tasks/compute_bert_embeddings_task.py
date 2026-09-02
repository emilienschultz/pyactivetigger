import csv
import gc
import json
import math
import os
import sys
from os import unlink
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from activetigger.functions import get_device, release_device_memory

# from celery.utils.log import get_task_logger
from pydantic import BaseModel
from transformers import (
    AutoModel,  # ty: ignore[possibly-missing-import]
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
)

from task_manager.auto_callback_task import AutoCallbackTask, QueueName
from task_manager.celery import celery_app

# TODO: is that necessary? we don't use csv lib in this code
csv.field_size_limit(sys.maxsize)

class ComputeBertEmbeddingsTaskParameters(TypedDict):
    model: str
    pooling: Literal["mean", "cls"]
    name: str
    kind: Literal['bert-embeddings']
    username: str
    max_length_tokens: int
    batch_size: int

# Return type must be a dict, a BaseModel would not be serialized by Celery
class ComputeBertEmbeddingsTaskResult(TypedDict):
    username: str
    project_slug: str
    embeddings_path: str
    parameters: ComputeBertEmbeddingsTaskParameters



# input can be a BaseModel if pydantic is enabled in the Task decorator
class ComputeBertEmbeddingsTaskInput(BaseModel):
    feature_name: str
    username: str
    project_slug: str
    texts_path: Path
    path_process: Path
    model_name: str
    model_dir: Path
    pooling: Literal["mean", "cls"] = "mean"
    batch_size: int = 32
    max_tokens: int = 512
    min_gpu: int = 6
    path_progress: Path | None = None

# Task definition using the auto callback generic parent task class
class ComputeBertEmbeddingsTask(AutoCallbackTask):
    name = "compute bert embeddings"
    # GPU task unless CPU_only mode
    queue = QueueName.GPU if os.environ.get("GPU") == "true" else QueueName.CPU
    path_progress: Path | None

    def log_progress(self, progress:float, project_slug:str):
        self.update_state(state="PROGRESS", meta={'status':"computing", 'progress':0})
        print(f"{project_slug} Bert Embeddings {progress}")
        if self.path_progress:
            with open(self.path_progress, "w") as f:
                f.write(str(round(progress, 1)))
        #  TODO: add custom progress event in live monitoring
        # self.send_event(
        #     "task-progress",
        #     **meta
        # )

    def compute_bert_embeddings(self, props:ComputeBertEmbeddingsTaskInput) -> ComputeBertEmbeddingsTaskResult:

        self.path_progress = props.path_progress if props.path_progress else  props.path_process.joinpath(self.request.id)
        
        # load texts
        texts = pd.read_pickle(props.texts_path)
        model_path = props.model_dir.joinpath(props.model_name)
        if texts.isnull().sum() > 0:
            raise ValueError("There are missing values in the input data, so we can't proceed")

        device = get_device()
        if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory / (1024**3) <= props.min_gpu:
                print("Not enough GPU memory, fallback to CPU")
                device = torch.device("cpu")

        # tokenizer comes from the base HF model (same convention as predict_bert);
        # weights come from the local fine-tuned checkpoint.
        params_file = model_path.joinpath("parameters.json")
        base_model:str|None = None
        with open(params_file, "r") as f:
            params = json.load(f)
            base_model = params.get("base_model")
        if not base_model:
            raise ValueError(f"base_model missing from {params_file}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
        model.to(device)
        model.eval()

        # clamp request to what the model actually supports; HF tokenizers may
        # report 1e30 when unset, so we also fall back to the encoder config
        model_max = getattr(tokenizer, "model_max_length", None)
        if not model_max or model_max > 100000:
            model_max = getattr(model.config, "max_position_embeddings", props.max_tokens)
        max_length = int(min(props.max_tokens, model_max))

        try:
            self.log_progress(0, props.project_slug)
            embeddings: list[np.ndarray] = []
            total_batches = math.ceil(len(texts) / props.batch_size)
            for i, start in enumerate(range(0, len(texts), props.batch_size), 1):
                

                batch_texts = list(texts.iloc[start : start + props.batch_size])
                encoded = tokenizer(  # ty: ignore[call-non-callable]
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = model(**encoded)

                last_hidden = outputs.last_hidden_state  # [B, T, H]
                if props.pooling == "cls":
                    pooled = last_hidden[:, 0, :]
                else:
                    # mean pool over real tokens (ignore padding)
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    summed = (last_hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / counts

                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy())

                progress_percent = (i / total_batches) * 100
                
                self.log_progress(round(progress_percent, 1), props.project_slug)
                
                

            stacked = np.vstack(embeddings)
            emb = pd.DataFrame(
                stacked,
                index=texts.index,
                columns=["be%03d" % (x + 1) for x in range(stacked.shape[1])],
            )
            embeddings_path = props.model_dir.joinpath("embeddings.parquet")
            emb.to_parquet(embeddings_path)
            # remove texts pickle object from filesystem
            unlink(props.texts_path)
            # save parameters
            parameters = ComputeBertEmbeddingsTaskParameters({
                "model": props.model_name,
                "pooling": props.pooling,
                "name": props.feature_name,
                "kind": 'bert-embeddings',
                "username": props.username,
                "max_length_tokens": props.max_tokens,
                "batch_size": props.batch_size,
            })
            return {
                'project_slug': props.project_slug,
                'username': props.username,
                'embeddings_path': str(embeddings_path),
                'parameters': parameters
            }

        finally:
            # thos manual garbage collection seems unnecessary as this computations is wrapped in a task
            if model:
                del model
            if tokenizer:
                del tokenizer
            del texts
            gc.collect()
            release_device_memory()
   
# Task registration
@celery_app.task(
    bind=True,
    name=ComputeBertEmbeddingsTask.name,
    queue=ComputeBertEmbeddingsTask.queue,
    base=ComputeBertEmbeddingsTask,
    pydantic=True,
)
def compute_bert_embeddings(self, props: ComputeBertEmbeddingsTaskInput):
    return self.compute_bert_embeddings(props=props)
 