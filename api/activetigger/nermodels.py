import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, cast

import pandas as pd
from fastapi.responses import FileResponse
from pandas import DataFrame

import activetigger.functions as functions
from activetigger.config import config
from activetigger.datamodels import (
    LMComputing,
    LMComputingOutModel,
    LMParametersDbModel,
    LMParametersModel,
    LMStatusModel,
    ModelInformationsModel,
    ModelScoresModel,
    NerModelsProjectStateModel,
    StaticFileModel,
    TextDatasetModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.functions import get_model_metrics
from activetigger.queue_manager import Queue
from activetigger.tasks.predict_ner import PredictNer
from activetigger.tasks.train_ner import TrainNer


class NerModels:
    """
    Manage fine-tuned token-classification (NER) models for span schemes.

    Mirrors LanguageModels but with a dedicated `ner/` subdirectory on
    disk and a dedicated "ner" DB kind so the two never collide — NER
    models can share a project with BERT classifiers without name clashes.

    TODO: if validated, mutualize code with languagemodel.py
    """

    project_slug: str
    path: Path
    queue: Queue
    computing: list
    models_service: ModelsService
    db_manager: DatabaseManager
    base_models: list[dict[str, Any]]
    params_default: LMParametersModel
    cache_predictions: dict[str, Tuple[datetime, pd.DataFrame]]

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        db_manager: DatabaseManager,
        list_models: str | None = None,
    ) -> None:
        self.params_default = LMParametersModel()
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.models_service = db_manager.language_models_service
        self.path = Path(path).joinpath("ner")
        self.cache_predictions = {}
        self._loss_cache: dict[str, Tuple[float, dict | None]] = {}
        self._loss_cache_interval: float = 5

        if list_models is not None and Path(list_models).exists():
            self.base_models = cast(
                list[dict[str, Any]], pd.read_csv(list_models).to_dict(orient="records")
            )
        else:
            # Reuse the BERT base-model defaults — token classification works
            # with the same encoders as sequence classification.
            self.base_models = [
                {
                    "name": "answerdotai/ModernBERT-base",
                    "priority": 10,
                    "comment": "",
                    "language": "en",
                },
                {
                    "name": "camembert/camembert-base",
                    "priority": 0,
                    "comment": "",
                    "language": "fr",
                },
            ]

        os.makedirs(self.path, exist_ok=True)

    def available(self) -> dict[str, dict[str, LMStatusModel]]:
        models = self.models_service.available_models(self.project_slug, "ner")
        r: dict = {}
        for m in models:
            if m.scheme not in r:
                r[m.scheme] = {}
            r[m.scheme][m.name] = LMStatusModel(
                predicted=m.parameters.get("predicted", False),
                predicted_all=m.parameters.get("predicted_all", False),
                tested=m.parameters.get("tested", False),
                predicted_external=m.parameters.get("predicted_external", False),
                name=m.name,
                time=m.time,
                exclude_labels=m.parameters.get("exclude_labels", []),
            )
        return r

    def exists(self, name: str) -> bool:
        return self.models_service.model_exists(self.project_slug, name)

    def training(self) -> dict[str, LMComputingOutModel]:
        return {
            e.user: LMComputingOutModel(
                name=e.model_name,
                status=e.status,
                progress=e.get_progress() if e.get_progress else None,
                loss=self.get_loss(e.model_name),
                epochs=e.params["epochs"] if e.params else None,
            )
            for e in self.computing
            if e.kind in ["ner", "train_ner", "predict_ner"]
        }

    def current_user_processes(self, user: str) -> list[LMComputing]:
        return [
            e
            for e in self.computing
            if e.user == user and e.kind in ["ner", "train_ner", "predict_ner"]
        ]

    def estimate_memory_use(self, model: str, kind: str = "train") -> int:
        if kind == "train":
            return 4
        if kind == "predict":
            return 3
        return 0

    def start_training_process(
        self,
        name: str,
        project: str,
        user: str,
        scheme: str,
        df: DataFrame,
        scheme_labels: list[str],
        col_text: str,
        col_label: str,
        params: LMParametersModel,
        base_model: str = "almanach/camembert-base",
        test_size: float = 0.2,
        num_min_annotations: int = 10,
        max_length: int = 512,
    ) -> str:
        if len(df.dropna()) < num_min_annotations:
            raise Exception(f"Less than {num_min_annotations} elements annotated")
        if self.models_service.model_exists(project, name):
            raise AlreadyExistsError("A model with this name already exists")
        if config.cpu_only:
            params.gpu = False
        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if self.estimate_memory_use(name, kind="train") > mem.available_memory:
                raise Exception("Not enough GPU memory available. Wait or reduce batch.")

        current_date = datetime.now(timezone.utc)
        unique_id = self.queue.add_task(
            "training",
            project,
            TrainNer(
                path=self.path,
                project_slug=project,
                model_name=name,
                df=df.copy(deep=True),
                scheme_labels=scheme_labels,
                col_label=col_label,
                col_text=col_text,
                base_model=base_model,
                params=params,
                test_size=test_size,
                max_length=max_length,
            ),
            queue="gpu",
        )
        del df
        params_db = LMParametersDbModel(**params.model_dump(), exclude_labels=[])
        self.computing.append(
            LMComputing(
                user=user,
                model_name=name,
                unique_id=unique_id,
                time=current_date,
                kind="train_ner",
                training_kind="ner",
                status="training",
                scheme=scheme,
                dataset=None,
                params=params_db.model_dump(),
                get_progress=self.get_progress(name, status="training"),
            )
        )
        return unique_id

    def start_predicting_process(
        self,
        project_slug: str,
        name: str,
        user: str,
        df: DataFrame | None,
        dataset: str,
        scheme_labels: list[str],
        col_label: str | None = None,
        batch_size: int = 16,
        status: str = "predicting",
        statistics: list | None = None,
        path_data: Path | None = None,
        external_dataset: TextDatasetModel | None = None,
        path_train: Path | None = None,
        path_valid: Path | None = None,
        path_test: Path | None = None,
    ) -> str:
        if not (self.path.joinpath(name)).exists():
            raise NotFoundError("The model does not exist")
        stale_progress = self.path.joinpath(name).joinpath("progress_predict")
        if stale_progress.exists():
            stale_progress.unlink()
        if df is None and dataset not in ["all", "external"]:
            raise Exception("Dataframe is required for this dataset")

        file_name = f"predict_{dataset}.parquet"
        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictNer(
                path=self.path.joinpath(name),
                dataset=dataset,
                df=df,
                scheme_labels=scheme_labels,
                col_text="text",
                col_label=col_label,
                col_id_external="id_external",
                col_datasets="dataset",
                file_name=file_name,
                batch=batch_size,
                statistics=statistics,
                path_data=path_data,
                external_dataset=external_dataset,
                path_train=path_train,
                path_valid=path_valid,
                path_test=path_test,
            ),
            queue="gpu",
        )
        self.computing.append(
            LMComputing(
                user=user,
                model_name=name,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="predict_ner",
                training_kind="ner",
                dataset=dataset,
                status=status,
                get_progress=self.get_progress(name, status=status),
            )
        )
        return unique_id

    def delete(self, name: str) -> None:
        if not name:
            raise ValueError("Model name is empty")
        model_dir = self.path.joinpath(name)
        tar_path = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"
        had_files = model_dir.exists() or os.path.exists(tar_path)
        errors = []
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                errors.append(f"model directory: {e}")
        try:
            os.remove(tar_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            errors.append(f"tar archive: {e}")
        db_removed = self.models_service.delete_model(self.project_slug, name)
        if not db_removed and not had_files:
            raise NotFoundError("Model does not exist")
        if errors:
            raise Exception(f"Problem to delete model files : {'; '.join(errors)}")

    def rename(self, former_name: str, new_name: str) -> None:
        model = self.models_service.get_model(self.project_slug, former_name)
        if model is None:
            raise NotFoundError("Model does not exist")
        if (Path(model.path) / "status.log").exists():
            raise Exception("Model is currently computing")
        self.models_service.rename_model(self.project_slug, former_name, new_name)
        model_path = Path(model.path)
        new_path = model_path.parent / model_path.name.replace(former_name, new_name)
        os.rename(model_path, new_path)

    def add(self, element: LMComputing) -> None:
        if element.status == "training":
            self.models_service.add_model(
                kind="ner",
                name=element.model_name,
                user=element.user,
                project=self.project_slug,
                scheme=element.scheme or "default",
                params=element.params or {},
                path=str(self.path.joinpath(element.model_name)),
                status="trained",
            )
            self.models_service.set_model_params(
                self.project_slug, element.model_name, "compressed", True
            )
            print("NER model trained")
        if element.status == "predicting":
            if element.dataset == "annotable":
                self.models_service.set_model_params(
                    self.project_slug, element.model_name, "predicted", True
                )
            if element.dataset == "all":
                self.models_service.set_model_params(
                    self.project_slug, element.model_name, "predicted_all", True
                )
            if element.dataset == "external":
                self.models_service.set_model_params(
                    self.project_slug, element.model_name, "predicted_external", True
                )
            print("NER prediction finished")

    def export_prediction(
        self,
        name: str,
        file_name: str = "predict.parquet",
        format: str = "parquet",
        col_id: str | None = None,
    ) -> FileResponse:
        path = self.path.joinpath(name).joinpath(file_name)
        if not path.exists():
            raise FileNotFoundError(
                f"The file {file_name} does not exist for this model, please run prediction again."
            )
        if format not in ("parquet", "csv", "xlsx"):
            raise Exception("Format not supported")
        if format == "parquet":
            return FileResponse(path=path, filename=file_name)
        ext = "csv" if format == "csv" else "xlsx"
        out_name = f"{file_name}.{ext}"
        out_path = self.path.joinpath(name).joinpath(out_name)
        if not out_path.exists() or out_path.stat().st_mtime < path.stat().st_mtime:
            df = pd.read_parquet(path)
            target_col = None
            meta_path = self.path.joinpath(name).joinpath(f"{file_name}.meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    target_col = json.load(f).get("col_id")
            elif col_id is not None:
                target_col = col_id.removeprefix("dataset_")
            if target_col is not None and "id_external" in df.columns:
                df.rename(columns={"id_external": target_col}, inplace=True)
                df = df[[target_col] + [c for c in df.columns if c != target_col]]
            if format == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_excel(out_path, index=False)
        return FileResponse(path=out_path, filename=out_name)

    def export_ner(self, name: str) -> StaticFileModel:
        file = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"
        if not Path(file).exists():
            raise NotFoundError("file does not exist")
        return StaticFileModel(name=f"{name}.tar.gz", path=f"{self.project_slug}/{name}.tar.gz")

    def get_progress(self, model_name: str, status: str) -> Callable[[], Optional[float]]:
        if status == "training":
            path_model = self.path.joinpath(model_name).joinpath("progress_train")
        elif status == "predicting":
            path_model = self.path.joinpath(model_name).joinpath("progress_predict")
        else:
            raise InvalidInputError("Status not recognized")

        def progress():
            if path_model.exists():
                r = path_model.read_text()
                if r.strip() == "":
                    r = 0
                return float(r)
            return None

        return progress

    def get_loss(self, model_name: str) -> dict | None:
        now = time.time()
        if model_name in self._loss_cache:
            cached_time, cached_loss = self._loss_cache[model_name]
            if (now - cached_time) < self._loss_cache_interval:
                return cached_loss
        try:
            with open(self.path.joinpath(model_name).joinpath("log_history.txt"), "r") as f:
                log = json.load(f)
            loss = pd.DataFrame(
                [
                    [
                        log[2 * i]["epoch"],
                        log[2 * i]["loss"],
                        log[2 * i + 1]["eval_loss"],
                    ]
                    for i in range(0, int(len(log) / 2))
                ],
                columns=["epoch", "val_loss", "val_eval_loss"],
            ).to_json()
            result = json.loads(loss)
        except Exception:
            result = None
        self._loss_cache[model_name] = (now, result)
        return result

    def get_parameters(self, model_name: str) -> dict | None:
        path = self.path.joinpath(model_name).joinpath("parameters.json")
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def get_informations(self, model_name: str) -> ModelInformationsModel:
        if not self.exists(model_name):
            raise NotFoundError(f"The model {model_name} does not exist")
        metrics = get_model_metrics(self.path.joinpath(model_name)) or {}
        db_params = self.models_service.get_model_db_parameters(self.project_slug, model_name) or {}
        return ModelInformationsModel(
            params=self.get_parameters(model_name),
            loss=self.get_loss(model_name),
            scores=ModelScoresModel(
                train_scores=metrics.get("train", None),
                internalvalid_scores=metrics.get("trainvalid", None),
                valid_scores=metrics.get("valid", None),
                test_scores=metrics.get("test", None),
                outofsample_scores=metrics.get("outofsample", None),
            ),
            predicted=bool(db_params.get("predicted", False)),
        )

    def get_base_model(self, model_name: str) -> str:
        with open(self.path.joinpath(model_name).joinpath("parameters.json"), "r") as f:
            data = json.load(f)
            if "base_model" in data:
                return data["base_model"]
            raise ValueError("No base_model found in parameters.json.")

    def get_prediction(self, model_name: str, cache_time: int = 120) -> pd.DataFrame:
        for key in list(self.cache_predictions.keys()):
            timestamp, _ = self.cache_predictions[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() > cache_time:
                del self.cache_predictions[key]
        if model_name in self.cache_predictions:
            _, cached_df = self.cache_predictions[model_name]
            return cached_df
        path = self.path.joinpath(model_name).joinpath("predict_annotable.parquet")
        if not path.exists():
            raise FileNotFoundError("Prediction file does not exist")
        df = pd.read_parquet(path)
        self.cache_predictions[model_name] = (datetime.now(timezone.utc), df)
        return df

    def state(self) -> NerModelsProjectStateModel:
        return NerModelsProjectStateModel(
            options=self.base_models,
            available=self.available(),
            training=self.training(),
            base_parameters=self.params_default,
        )
