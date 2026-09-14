"""
Prompt management for embedding-based selection.

A prompt is a short text query that is embedded with the same model used for
the bound feature, so the query vector and the document/image vectors live in
the same space; `get_next` then ranks candidates by cosine similarity.

Supported feature kinds:
- `multimodal-embeddings` (image projects): query encoded with the feature's
  multimodal sentence-transformer (e.g. CLIP / BGE-VL / Qwen-VL).
- `sentence-embeddings` (text projects): query encoded with the same
  SentenceTransformer used to compute the feature.

Storage lives in a dedicated `prompts.parquet` (one row per prompt, with the
embedding columns inline), separate from `features.parquet`.
"""

import builtins
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from activetigger.datamodels import (
    PromptComputing,
    PromptOutModel,
    PromptSimilarityComputing,
    PromptsProjectStateModel,
)
from activetigger.errors import InvalidInputError, NotFoundError
from activetigger.features import Features
from activetigger.queue_manager import Queue
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.compute_multimodal_prompt import ComputeMultimodalPrompt
from activetigger.tasks.compute_sbert_prompt import ComputeSbertPrompt
from activetigger.tasks.compute_similarity_with_features import ComputeSimilarityWithFeatures

# Feature kinds whose vectors live in a space we can encode a text query into.
BINDABLE_FEATURE_KINDS = {"multimodal-embeddings", "sentence-embeddings"}

PROMPTS_FILE = "prompts.parquet"

# Per-prompt similarity files computed on a dataset, exportable from the
# export panel: <project_dir>/prompts_similarity/<prompt_id>/similarity_<dataset>.parquet
SIMILARITY_DIR = "prompts_similarity"

SIMILARITY_DATASETS = {"all", "train", "valid", "test"}

METADATA_COLUMNS = ["text", "feature_name", "user", "created_at"]


class Prompts:
    """Per-project store of embedded natural-language prompts."""

    def __init__(
        self,
        project_slug: str,
        path_dir: Path,
        queue: Queue,
        computing: list,
        features: Features,
    ) -> None:
        self.project_slug = project_slug
        self.path_dir = path_dir
        self.path_file = path_dir.joinpath(PROMPTS_FILE)
        self.queue = queue
        self.computing = computing
        self.features = features
        # Cache of sorted rankings keyed by (prompt_id, dataset). Each entry
        # is a pd.Series indexed by element_id with cosine similarity values,
        # sorted descending. Populated lazily by get_ranking and invalidated
        # on prompt delete / feature cascade.
        self._ranking_cache: dict[str, dict[str, pd.Series]] = {}

    # ---------- helpers ----------

    def _read(self) -> pd.DataFrame:
        if not self.path_file.exists():
            return pd.DataFrame(columns=METADATA_COLUMNS)
        return pd.read_parquet(self.path_file)

    def _write(self, df: pd.DataFrame) -> None:
        # Writing an empty DataFrame can produce a parquet file that pyarrow
        # later fails to read ("magic bytes not found in footer"). Treat an
        # empty result the same as "no file" — _read already handles that.
        if df.empty:
            if self.path_file.exists():
                self.path_file.unlink()
            return
        df.to_parquet(self.path_file, index=True)

    def _resolve_feature(self, feature_name: str) -> tuple[str, str]:
        """
        Return (model_name, kind) for a bindable feature.

        Model name is read from `parameters.hf_name` (multimodal-embeddings,
        set in features.py) or `parameters.model` (sentence-embeddings); both
        keys are checked so future kinds can use either convention.
        """
        available = self.features.get_available()
        feat = available.get(feature_name)
        if feat is None:
            raise NotFoundError(f"Feature '{feature_name}' does not exist")
        if feat.kind not in BINDABLE_FEATURE_KINDS:
            raise ValueError(
                f"Feature '{feature_name}' kind '{feat.kind}' is not bindable to a prompt "
                f"(expected one of {sorted(BINDABLE_FEATURE_KINDS)})"
            )
        params = feat.parameters or {}
        model_name = params.get("hf_name") or params.get("model")
        if not model_name:
            raise ValueError(f"Feature '{feature_name}' has no stored model name")
        return str(model_name), feat.kind

    # ---------- public API ----------

    def add(self, text: str, feature_name: str, user: str) -> str:
        """
        Queue the encoding of a new prompt. Returns the task unique_id.
        The actual parquet row is appended by `receive_result` once the
        GPU worker completes.
        """
        text = text.strip()
        if not text:
            raise InvalidInputError("Prompt text cannot be empty")
        model_name, kind = self._resolve_feature(feature_name)
        prompt_id = str(uuid.uuid4())

        task: BaseTask
        if kind == "multimodal-embeddings":
            task = ComputeMultimodalPrompt(
                text=text, model_name=model_name, path_process=self.path_dir
            )
        elif kind == "sentence-embeddings":
            task = ComputeSbertPrompt(text=text, model_name=model_name, path_process=self.path_dir)
        else:
            # Defensive: _resolve_feature already gates on BINDABLE_FEATURE_KINDS,
            # so this only fires if a new bindable kind is added without a task.
            raise ValueError(f"No prompt encoder for feature kind '{kind}'")

        unique_id = self.queue.add_task("prompt", self.project_slug, task, queue="gpu")

        self.computing.append(
            PromptComputing(
                user=user,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="prompt",
                prompt_id=prompt_id,
                text=text,
                feature_name=feature_name,
                hf_name=model_name,
            )
        )
        return unique_id

    def receive_result(self, computing: PromptComputing, vec: np.ndarray) -> None:
        """Persist a completed prompt to prompts.parquet."""
        # The feature might have been deleted while the task was queued.
        if not self.features.exists(computing.feature_name):
            return

        vec = np.asarray(vec).reshape(-1)
        row: dict[str, Any] = {
            "text": computing.text,
            "feature_name": computing.feature_name,
            "user": computing.user,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for i, v in enumerate(vec):
            row[f"dim_{i}"] = float(v)

        df = self._read()
        new_row = pd.DataFrame([row], index=pd.Index([computing.prompt_id], name="prompt_id"))

        # Align columns: existing rows may have fewer/more dim_* columns if
        # the feature changed. Union columns, fill missing with NaN.
        combined = pd.concat([df, new_row], axis=0)
        combined.index.name = "prompt_id"
        self._write(combined)

    def list(self, user: str | None = None) -> list[PromptOutModel]:
        df = self._read()
        if df.empty:
            return []
        if user is not None:
            df = df[df["user"] == user]
        out: list[PromptOutModel] = []
        for prompt_id, row in df.iterrows():
            out.append(
                PromptOutModel(
                    prompt_id=str(prompt_id),
                    text=str(row["text"]),
                    feature_name=str(row["feature_name"]),
                    user=str(row["user"]),
                    created_at=str(row["created_at"]),
                    computed_datasets=self.computed_datasets(str(prompt_id)),
                )
            )
        return out

    def get_embedding_and_feature(self, prompt_id: str) -> tuple[np.ndarray, str]:
        df = self._read()
        if prompt_id not in df.index:
            raise NotFoundError(f"Prompt '{prompt_id}' not found")
        row = df.loc[prompt_id]
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        if not dim_cols:
            raise ValueError(f"Prompt '{prompt_id}' has no embedding")
        vec = row[dim_cols].to_numpy(dtype=float)
        return vec, str(row["feature_name"])

    def delete(self, prompt_id: str) -> None:
        df = self._read()
        if prompt_id not in df.index:
            raise NotFoundError(f"Prompt '{prompt_id}' not found")
        df = df.drop(index=prompt_id)
        self._write(df)
        self._ranking_cache.pop(prompt_id, None)
        self._delete_similarity_files(prompt_id)

    def reset_all(self) -> None:
        """
        Drop every saved prompt and clear the ranking cache.
        Called from the Features on_reset cascade when all features are wiped.
        """
        if self.path_file.exists():
            try:
                self.path_file.unlink()
            except OSError as ex:
                print(f"Could not delete prompts file: {ex}")
        self._ranking_cache.clear()
        similarity_root = self.path_dir.joinpath(SIMILARITY_DIR)
        if similarity_root.exists():
            try:
                shutil.rmtree(similarity_root)
            except OSError as ex:
                print(f"Could not delete prompt similarity files: {ex}")

    def delete_by_feature(self, feature_name: str) -> int:
        """Drop every prompt bound to a feature (cascade from feature delete)."""
        df = self._read()
        if df.empty or "feature_name" not in df.columns:
            return 0
        mask = df["feature_name"] == feature_name
        n = int(mask.sum())
        if n:
            dropped_ids = [str(i) for i in df.index[mask]]
            self._write(df.loc[~mask])
            for pid in dropped_ids:
                self._ranking_cache.pop(pid, None)
                self._delete_similarity_files(pid)
        return n

    def get_ranking(self, prompt_id: str, dataset: str) -> pd.Series:
        """
        Return element_ids sorted by descending cosine similarity between the
        prompt embedding and the bound feature's element embeddings, over the
        full given dataset. Cached per (prompt_id, dataset) — subsequent
        get_next calls on the same prompt only do an index intersection.
        """
        cache_by_ds = self._ranking_cache.setdefault(prompt_id, {})
        cached = cache_by_ds.get(dataset)
        if cached is not None:
            return cached

        prompt_vec, feature_name = self.get_embedding_and_feature(prompt_id)
        feat_df = self.features.get([feature_name], dataset=[dataset])
        if feat_df.empty:
            raise ValueError(f"No embeddings for feature '{feature_name}' in dataset '{dataset}'")
        mat = feat_df.to_numpy(dtype=float)
        prompt_vec = np.asarray(prompt_vec, dtype=float).reshape(-1)
        if mat.shape[1] != prompt_vec.shape[0]:
            raise ValueError(
                f"Dimension mismatch between prompt ({prompt_vec.shape[0]}) "
                f"and feature '{feature_name}' ({mat.shape[1]}). "
                "The feature may have been recomputed with a different model."
            )
        row_norms = np.linalg.norm(mat, axis=1)
        prompt_norm = float(np.linalg.norm(prompt_vec))
        sims = (mat @ prompt_vec) / (row_norms * prompt_norm + 1e-12)
        ranked = pd.Series(sims, index=feat_df.index).sort_values(ascending=False)

        cache_by_ds[dataset] = ranked
        return ranked

    # ---------- full-dataset similarity (issue #1120) ----------

    def _similarity_dir(self, prompt_id: str) -> Path:
        return self.path_dir.joinpath(SIMILARITY_DIR).joinpath(prompt_id)

    def similarity_file(self, prompt_id: str, dataset: str) -> Path:
        return self._similarity_dir(prompt_id).joinpath(f"similarity_{dataset}.parquet")

    # builtins.list: bare `list` here would resolve to the `list` method
    # defined above in the class body
    def computed_datasets(self, prompt_id: str) -> builtins.list[str]:
        """Datasets for which a similarity file exists and can be exported."""
        folder = self._similarity_dir(prompt_id)
        if not folder.exists():
            return []
        out = [
            ds
            for ds in sorted(SIMILARITY_DATASETS)
            if folder.joinpath(f"similarity_{ds}.parquet").exists()
        ]
        return out

    def _delete_similarity_files(self, prompt_id: str) -> None:
        folder = self._similarity_dir(prompt_id)
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except OSError as ex:
                print(f"Could not delete similarity files for prompt {prompt_id}: {ex}")

    def _dataset_series(self, dataset: str) -> pd.Series:
        """
        Input series (texts for text projects, image paths for image
        projects — both live in the `text` column) consumed by feature
        recomputation for the requested dataset. Mirrors
        `QuickModels._dataset_texts`: for "all" we read `path_all`
        directly and index by `id_external` so the saved similarity
        parquet carries a meaningful row id.
        """
        if dataset == "all":
            df = pd.read_parquet(self.features.path_all, columns=["id_external", "text"])
            series = df["text"]
            series.index = df["id_external"].astype(str)
            series.index.name = "id_external"
            return series
        return self.features.concat_split_column("text", dataset=dataset)

    def compute_similarity(self, prompt_id: str, dataset: str, username: str) -> str | None:
        """
        Compute the cosine similarity between a prompt and every element
        of the requested dataset, persisted as a parquet exportable from
        the export panel.

        For train/valid/test the embeddings already live in the features
        file, so the file is written synchronously from `get_ranking`
        (returns None). For "all" the bound feature is recomputed on the
        complete dataset by a queued `ComputeSimilarityWithFeatures`
        task (returns the task unique_id).
        """
        if dataset not in SIMILARITY_DATASETS:
            raise InvalidInputError(f"Dataset must be one of {sorted(SIMILARITY_DATASETS)}")

        prompt_vec, feature_name = self.get_embedding_and_feature(prompt_id)

        if dataset != "all":
            ranked = self.get_ranking(prompt_id, dataset)
            out = pd.DataFrame({"similarity": ranked})
            out["rank"] = range(1, len(out) + 1)
            folder = self._similarity_dir(prompt_id)
            folder.mkdir(parents=True, exist_ok=True)
            out.to_parquet(self.similarity_file(prompt_id, dataset), index=True)
            return None

        # only one similarity computation per prompt at a time
        for e in self.computing:
            if (
                getattr(e, "kind", None) == "prompt_similarity"
                and getattr(e, "prompt_id", None) == prompt_id
            ):
                raise ValueError("A similarity computation is already running for this prompt")

        data = self._dataset_series(dataset)
        specs = self.features.build_compute_specs([feature_name], data)
        prompt_row = self._read().loc[prompt_id]

        task = ComputeSimilarityWithFeatures(
            specs=specs,
            path_process=self.features.path_all.parent,
            path_models=self.features.path_models,
            language=self.features.lang,
            prompt_vector=prompt_vec,
            path_output=self._similarity_dir(prompt_id),
            file_name=f"similarity_{dataset}.parquet",
        )
        unique_id = self.queue.add_task("prompt_similarity", self.project_slug, task, queue="gpu")

        progress_path = self.features.path_all.parent.joinpath(unique_id)

        def get_progress() -> float | None:
            try:
                if not progress_path.exists():
                    return None
                raw = progress_path.read_text().strip()
                return float(raw) if raw else 0.0
            except (OSError, ValueError):
                return None

        self.computing.append(
            PromptSimilarityComputing(
                user=username,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="prompt_similarity",
                prompt_id=prompt_id,
                text=str(prompt_row["text"]),
                feature_name=feature_name,
                dataset=dataset,
                get_progress=get_progress,
            )
        )
        return unique_id

    def export_similarity(
        self, prompt_id: str, dataset: str, format: str, col_id: str | None
    ) -> tuple[Path, str]:
        """
        Prepare the similarity file produced by `compute_similarity` for
        download and return (path, file_name) — the caller (export
        router) wraps it in a FileResponse. Same format conversion
        convention as `QuickModels.export_prediction_file`.
        """
        # prompt_id and dataset come from query params and are joined into
        # filesystem paths below: only accept ids present in the prompts
        # index and known dataset names, otherwise a crafted value could
        # escape the project directory.
        if prompt_id not in self._read().index:
            raise NotFoundError(f"Prompt '{prompt_id}' not found")
        if dataset not in SIMILARITY_DATASETS:
            raise InvalidInputError(f"Dataset must be one of {sorted(SIMILARITY_DATASETS)}")

        path = self.similarity_file(prompt_id, dataset)
        if not path.exists():
            raise FileNotFoundError(
                f"No similarity computed on dataset '{dataset}' for this prompt, "
                "please compute it first."
            )

        if format not in ("parquet", "csv", "xlsx"):
            raise ValueError("Format not supported")

        file_name = f"similarity_{dataset}.parquet"
        if format == "parquet":
            return path, file_name

        ext = "csv" if format == "csv" else "xlsx"
        out_name = f"{file_name}.{ext}"
        out_path = self._similarity_dir(prompt_id).joinpath(out_name)
        if not out_path.exists() or out_path.stat().st_mtime < path.stat().st_mtime:
            df = pd.read_parquet(path)
            target_col = col_id.removeprefix("dataset_") if col_id else "id_external"
            df = df.reset_index()
            first_col = df.columns[0]
            if first_col != target_col:
                df.rename(columns={first_col: target_col}, inplace=True)
            df = df[[target_col] + [c for c in df.columns if c != target_col]]
            if format == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_excel(out_path, index=False)

        return out_path, out_name

    def current_similarity_computing(self) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {}
        for e in self.computing:
            if e.kind != "prompt_similarity":
                continue
            progress = e.get_progress() if e.get_progress is not None else None
            out[e.prompt_id] = {
                "prompt_id": e.prompt_id,
                "text": e.text,
                "feature_name": e.feature_name,
                "dataset": e.dataset,
                "progress": str(progress) if progress is not None else None,
            }
        return out

    def current_computing(self) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {}
        for e in self.computing:
            if e.kind != "prompt":
                continue
            e = e  # type: PromptComputing
            progress_file = self.path_dir.joinpath(e.unique_id)
            progress: str | None = None
            try:
                if progress_file.exists():
                    progress = progress_file.read_text().strip()
            except OSError:
                progress = None
            out[e.prompt_id] = {
                "prompt_id": e.prompt_id,
                "text": e.text,
                "feature_name": e.feature_name,
                "progress": progress,
            }
        return out

    def state(self) -> PromptsProjectStateModel:
        try:
            available = self.features.get_available()
        except Exception:
            available = {}
        bindable = [name for name, feat in available.items() if feat.kind in BINDABLE_FEATURE_KINDS]
        return PromptsProjectStateModel(
            available=self.list(),
            bindable_features=bindable,
            training=self.current_computing(),
            similarity_computing=self.current_similarity_computing(),
        )
