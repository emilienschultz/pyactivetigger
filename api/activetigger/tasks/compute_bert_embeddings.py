import gc
import json
import math
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from transformers import (
    AutoModel,  # ty: ignore[possibly-missing-import]
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
)

from activetigger.functions import get_device, release_device_memory
from activetigger.tasks.base_task import BaseTask


# TODO: deprecated task to remove once compute_feaeture_from_specs has been refacto with taskmanager
class ComputeBertEmbeddings(BaseTask):
    """
    Compute embeddings from a fine-tuned BERT model saved by `train_bert.py`.
    Loads the encoder (drops the classification head), runs a forward pass and
    pools token states into a single vector per text.
    """

    kind = "compute_feature_bert_embeddings"

    def __init__(
        self,
        texts: Series,
        path_process: Path,
        model_path: Path,
        pooling: str = "mean",
        batch_size: int = 32,
        max_tokens: int = 512,
        min_gpu: int = 6,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.texts = texts
        self.model_path = Path(model_path)
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.min_gpu = min_gpu
        self.path_process = path_process
        self.event = event
        if path_progress:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.path_progress = self.path_process.joinpath(self.unique_id)
            self.progress_file_temporary = True

    def _load_base_model_name(self) -> str:
        params_file = self.model_path.joinpath("parameters.json")
        with open(params_file, "r") as f:
            params = json.load(f)
        base = params.get("base_model")
        if not base:
            raise ValueError(f"base_model missing from {params_file}")
        return base

    def __call__(self) -> DataFrame:
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)

        if self.texts.isnull().sum() > 0:
            raise ValueError("There are missing values in the input data, so we can't proceed")

        device = get_device()
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).total_memory / (1024**3) <= self.min_gpu:
                print("Not enough GPU memory, fallback to CPU")
                device = torch.device("cpu")

        # prefer the tokenizer saved with the fine-tuned checkpoint
        if self.model_path.joinpath("tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        else:
            base_model = self._load_base_model_name()
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(str(self.model_path), trust_remote_code=True)
        model.to(device)
        model.eval()

        # clamp request to what the model actually supports; HF tokenizers may
        # report 1e30 when unset, so we also fall back to the encoder config
        model_max = getattr(tokenizer, "model_max_length", None)
        if not model_max or model_max > 100000:
            model_max = getattr(model.config, "max_position_embeddings", self.max_tokens)
        max_length = int(min(self.max_tokens, model_max))

        try:
            print("start computation")
            embeddings: list[np.ndarray] = []
            total_batches = math.ceil(len(self.texts) / self.batch_size)
            for i, start in enumerate(range(0, len(self.texts), self.batch_size), 1):
                if self.event is not None and self.event.is_set():
                    print("Process interrupted by user")
                    raise Exception("Process interrupted by user")

                batch_texts = list(self.texts.iloc[start : start + self.batch_size])
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
                if self.pooling == "cls":
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
                with open(self.path_progress, "w") as f:
                    f.write(str(round(progress_percent, 1)))
                print(progress_percent)

            stacked = np.vstack(embeddings)
            emb = DataFrame(
                stacked,
                index=self.texts.index,
                columns=["be%03d" % (x + 1) for x in range(stacked.shape[1])],
            )
            return emb
        finally:
            if self.progress_file_temporary:
                self.path_progress.unlink(missing_ok=True)
            try:
                del model, tokenizer
            except Exception:
                pass
            del self.texts
            gc.collect()
            release_device_memory()
