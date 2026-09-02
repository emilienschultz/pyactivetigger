import threading
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import regex
from pandas import DataFrame, Series

from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.compute_bert_embeddings import ComputeBertEmbeddings
from activetigger.tasks.compute_clip_imagexp import ComputeClipImagexp
from activetigger.tasks.compute_dfm import ComputeDfm
from activetigger.tasks.compute_fasttext import ComputeFasttext
from activetigger.tasks.compute_multimodal import ComputeMultimodal
from activetigger.tasks.compute_sbert import ComputeSbert


class ComputeFeaturesFromSpecs(BaseTask):
    """
    Base for meta-tasks that (re)compute a batch of features described by
    `specs` and then do something with the resulting DataFrame (predict,
    write to the features file, ...). Subclasses implement __call__ and
    use `_compute_all()` to get the feature matrix.

    Each spec must have keys:
      - `name` (str)      : feature name, used as the output column prefix
        (same `<name>__<subcol>` scheme as the project features parquet).
      - `kind` (str)      : one of `Features.COMPUTABLE_KINDS`.
      - `parameters` (dict): as stored in the DB by `Features.add`.
      - `data` (Series)   : input series for this kind (text for text
        kinds, paths for image kinds).
      - `model_path` (Path, optional): required for `bert-embeddings`
        (resolved by the caller so this task stays independent of the
        `LanguageModels` manager).

    Progress reporting follows the same file-based convention as other
    tasks in this package: overall percentage is written to
    `path_progress` (defaults to `path_process/unique_id`) as a decimal
    string. When a sub-task exposes its own progress file, a watcher
    thread interpolates it into the overall percentage so the bar moves
    smoothly within a step; otherwise the bar advances only at step
    boundaries.
    """

    def __init__(
        self,
        specs: list[dict[str, Any]],
        path_process: Path,
        path_models: Path,
        language: str,
        path_progress: Path | None = None,
    ):
        super().__init__()
        self.specs = specs
        self.path_process = path_process
        self.path_models = path_models
        self.language = language
        if path_progress is not None:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.progress_file_temporary = True
            self.path_progress = self.path_process.joinpath(self.unique_id)

    def _refresh_progress_path(self) -> None:
        # `progress_file_temporary` means the caller relies on unique_id
        # naming — refresh in case unique_id changed since __init__.
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)

    def _cleanup_progress_file(self) -> None:
        if self.progress_file_temporary:
            self.path_progress.unlink(missing_ok=True)

    def _compute_all(self, total: int | None = None) -> DataFrame:
        """
        Run every spec and return the concatenated feature matrix with
        `<name>__<subcol>` columns. `total` is the number of progress
        units; pass `len(specs) + k` to reserve k units for follow-up
        steps in the subclass.
        """
        if total is None:
            total = max(len(self.specs), 1)
        frames: list[DataFrame] = []
        self._write_overall(0.0)
        self._raise_if_cancelled()
        for i, spec in enumerate(self.specs):
            self._raise_if_cancelled()
            name = spec["name"]
            sub_callable, watch_path = self._prepare(spec)
            # Forward the meta-task's cancellation event so sub-tasks
            # that cooperate (ComputeSbert/Bert/Clip/Multimodal/Fasttext
            # all poll self.event.is_set()) can abort mid-embedding.
            # Closures used for regex don't carry an event; the
            # per-iteration `_raise_if_cancelled` above covers them.
            # `setattr` sidesteps ty narrowing sub_callable to its
            # Callable-Protocol shape (which doesn't expose `event`).
            if hasattr(sub_callable, "event"):
                setattr(sub_callable, "event", self.event)

            stop: threading.Event | None = None
            watcher: threading.Thread | None = None
            if watch_path is not None:
                stop = threading.Event()
                watcher = threading.Thread(
                    target=self._watch,
                    args=(watch_path, i, total, stop),
                    daemon=True,
                )
                watcher.start()

            try:
                result = sub_callable()
            finally:
                if stop is not None:
                    stop.set()
                if watcher is not None:
                    watcher.join(timeout=2)

            if isinstance(result, Series):
                result = result.to_frame()
            result.columns = [f"{name}__{c}" for c in result.columns]
            frames.append(result)

            self._write_overall(((i + 1) / total) * 100)

        self._raise_if_cancelled()
        return pd.concat(frames, axis=1) if frames else DataFrame()

    def _prepare(self, spec: dict[str, Any]) -> tuple[Callable[[], Any], Path | None]:
        """
        Instantiate the sub-task (or wrap an inline computation) for one
        spec. Return `(callable, watch_path)` where `watch_path` is the
        sub-task's own progress file (to be watched by the meta-task) or
        `None` if the sub-task does not report progress.
        """
        kind = spec["kind"]
        parameters = spec["parameters"]
        data = spec["data"]

        if kind == "regex":
            pattern = regex.compile(parameters["regex"])
            count_mode = parameters.get("mode") == "count"

            def run_regex() -> Series:
                if count_mode:
                    return data.apply(lambda x: len(pattern.findall(x)))
                return data.apply(lambda x: bool(pattern.search(x)))

            return run_regex, None

        if kind == "sentence-embeddings":
            task = ComputeSbert(
                texts=data,
                path_process=self.path_process,
                model=parameters["model"],
                max_tokens=int(parameters.get("max_length_tokens", 1024)),
                batch_size=int(parameters.get("batch_size", 32)),
            )
            return task, task.path_progress

        if kind == "bert-embeddings":
            model_path = spec.get("model_path")
            if model_path is None:
                raise ValueError(
                    "bert-embeddings spec is missing 'model_path' "
                    "(caller must resolve it before invoking the task)"
                )
            # We can't use taskmanager here till all subtask are ported to taskmanager to be able to uuse a chord
            task = ComputeBertEmbeddings(
                texts=data,
                path_process=self.path_process,
                model_path=model_path,
                pooling=parameters.get("pooling", "mean"),
                batch_size=int(parameters.get("batch_size", 32)),
                max_tokens=int(parameters.get("max_length_tokens", 512)),
            )
            return task, task.path_progress

        if kind == "fasttext":
            task = ComputeFasttext(
                texts=data,
                language=self.language,
                path_process=self.path_process,
                path_models=self.path_models,
                model=parameters["model"],
            )
            # ComputeFasttext writes to path_process/unique_id; there is
            # no `path_progress` attribute, so derive it from unique_id.
            return task, self.path_process.joinpath(task.unique_id)

        if kind == "dfm":
            task = ComputeDfm(
                texts=data,
                tfidf=parameters.get("tfidf", False),
                ngrams=parameters.get("dfm_ngrams", 1),
                min_term_freq=parameters.get("dfm_min_term_freq", 5),
                max_term_freq=parameters.get("dfm_max_term_freq", 100),
                log=parameters.get("dfm_log", False),
                language=self.language,
                norm=parameters.get("dfm_norm", None),
            )
            # DFM is a one-shot vectoriser; no incremental progress.
            return task, None

        if kind == "image-embeddings":
            from activetigger.features import IMAGE_EMBEDDING_MODELS_IMAGEXP

            model_spec = IMAGE_EMBEDDING_MODELS_IMAGEXP.get(parameters["model"])
            if model_spec is None:
                raise ValueError(f"Unknown image embedding model: {parameters['model']}")
            task = ComputeClipImagexp(
                paths=data,
                path_process=self.path_process,
                model=model_spec["model"],
                pretrained=model_spec["pretrained"],
                batch_size=int(parameters.get("batch_size", 16)),
            )
            return task, task.path_progress

        if kind == "multimodal-embeddings":
            hf_name = parameters.get("hf_name")
            if not hf_name:
                from activetigger.features import MULTIMODAL_EMBEDDING_MODELS

                hf_name = MULTIMODAL_EMBEDDING_MODELS.get(parameters["model"])
                if hf_name is None:
                    raise ValueError(f"Unknown multimodal embedding model: {parameters['model']}")
            task = ComputeMultimodal(
                paths=data,
                path_process=self.path_process,
                model_name=hf_name,
                batch_size=int(parameters.get("batch_size", 8)),
            )
            return task, task.path_progress

        raise ValueError(f"Unsupported feature kind: {kind}")

    def _watch(
        self,
        watch_path: Path,
        step_index: int,
        total: int,
        stop: threading.Event,
    ) -> None:
        """
        Background poller: reads `watch_path` (sub-task's own progress
        file, values 0..100) and rewrites the meta progress as
        `(step_index + sub_pct/100) / total * 100`. Runs until `stop` is
        set. Silently ignores read/parse errors — the sub-task may not
        have written the file yet or may be mid-write.
        """
        while not stop.is_set():
            try:
                with open(watch_path) as f:
                    raw = f.read().strip()
                if raw:
                    sub_pct = float(raw)
                    self._write_overall(((step_index + sub_pct / 100.0) / total) * 100.0)
            except (FileNotFoundError, ValueError, OSError):
                pass
            stop.wait(0.5)

    def _write_overall(self, pct: float) -> None:
        pct = max(0.0, min(100.0, pct))
        try:
            with open(self.path_progress, "w") as f:
                f.write(str(round(pct, 1)))
        except OSError:
            pass

    def _raise_if_cancelled(self) -> None:
        """
        Bail out cooperatively when the queue signals a kill. The queue
        sets `self.event` on this task (see `Queue.add_task` / `kill`),
        so a running __call__ can spot the kill between feature steps.
        Sub-tasks that cooperate get the same event forwarded in
        `_compute_all` so they can abort mid-embedding too.
        """
        if self.event is not None and self.event.is_set():
            raise Exception("Process interrupted by user")
