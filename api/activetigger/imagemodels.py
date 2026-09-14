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
    ImageModelsProjectStateModel,
    LMComputing,
    LMComputingOutModel,
    LMParametersDbModel,
    LMParametersModel,
    LMStatusModel,
    ModelInformationsModel,
    ModelScoresModel,
    StaticFileModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.functions import get_model_metrics
from activetigger.queue_manager import Queue
from activetigger.tasks.predict_image import PredictImage
from activetigger.tasks.train_image import TrainImage


class ImageModels:
    """
    Module to manage fine-tuned image-classification models for image projects.

    Mirrors LanguageModels but operates on image paths (the `text` column of
    an image-project scheme dataframe holds the absolute path on disk) and
    delegates training/prediction to TrainImage / PredictImage. Works with
    any HuggingFace AutoModelForImageClassification (ViT, ConvNeXt,
    EfficientNet, Swin, BEiT, ...).
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
        self.params_default = LMParametersModel(
            batchsize=16,
            gradacc=8,
            epochs=10,
            lrate=1e-4,
            wdecay=1e-4,
            best=True,
            eval=10,
            gpu=True,
            adapt=False,
        )
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.models_service = db_manager.language_models_service
        self.path: Path = Path(path).joinpath("image").absolute()
        self.cache_predictions = {}
        self._loss_cache: dict[str, Tuple[float, dict | None]] = {}
        self._loss_cache_interval: float = 5

        if list_models is not None and Path(list_models).exists():
            self.base_models = cast(
                list[dict[str, Any]], pd.read_csv(list_models).to_dict(orient="records")
            )
        else:
            # Fallback list when no CSV is found. Mirror the main image_models.csv.
            self.base_models = [
                {
                    "name": "google/vit-large-patch16-384",
                    "priority": 10,
                    "comment": "ViT-Large (304M) at 384px — default",
                    "parameters": 304,
                    "image_size": 384,
                }
            ]

        # exist_ok handles the TOCTOU race when two requests load the same
        # project concurrently (both pass the exists() check, only one wins).
        os.makedirs(self.path, exist_ok=True)

    def available(self) -> dict[str, dict[str, LMStatusModel]]:
        """
        Return available models
        """
        models = self.models_service.available_models(self.project_slug, "image")
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
        """
        Test if the model exist
        """
        return self.models_service.model_exists(self.project_slug, name)

    def training(self) -> dict[str, LMComputingOutModel]:
        """
        Return models in training
        """
        return {
            e.user: LMComputingOutModel(
                name=e.model_name,
                status=e.status,
                progress=e.get_progress() if e.get_progress else None,
                loss=self.get_loss(e.model_name),
                epochs=e.params["epochs"] if e.params else None,
            )
            for e in self.computing
            if e.kind in ["image", "train_image", "predict_image"]
        }

    def current_user_processes(self, user: str) -> list[LMComputing]:
        return [
            e
            for e in self.computing
            if e.user == user and e.kind in ["image", "train_image", "predict_image"]
        ]

    def estimate_memory_use(self, model: str, kind: str = "train", batchsize: int = 16) -> int:
        """
        Rough GPU memory estimate (GB) for image-classification fine-tuning
        / prediction.
        """
        if kind == "train":
            if "large" in model and "384" in model:
                return max(12, int(0.75 * batchsize + 8))
            return max(6, int(0.4 * batchsize + 4))
        if kind == "predict":
            return 4
        return 0

    def start_training_process(
        self,
        name: str,
        project: str,
        user: str,
        scheme: str,
        df: DataFrame,
        training_kind: str,
        scheme_labels: list[str],
        col_text: str,
        col_label: str,
        params: LMParametersModel,
        base_model: str = "google/vit-large-patch16-384",
        test_size: float = 0.2,
        num_min_annotations: int = 10,
        loss: str = "cross_entropy",
        class_balance: bool = False,
        class_min_freq: int = 1,
        exclude_labels: list[str] | None = None,
        fp16: bool = True,
    ) -> str:
        """
        Launch a training process
        """
        if len(df.dropna()) < num_min_annotations:
            raise Exception(f"Less than {num_min_annotations} elements annotated")

        current_date = datetime.now(timezone.utc)
        model_name = name

        if self.models_service.model_exists(project, model_name):
            raise AlreadyExistsError("A model with this name already exists")

        if config.cpu_only:
            params.gpu = False

        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if (
                self.estimate_memory_use(base_model, kind="train", batchsize=int(params.batchsize))
                > mem.available_memory
            ):
                raise Exception(
                    "Not enough GPU memory available. Reduce batch size or use gradient accumulation."
                )

        if training_kind not in ["multilabel", "multiclass"]:
            raise Exception("training_kind must be multilabel or multiclass")

        unique_id = self.queue.add_task(
            "training",
            project,
            TrainImage(
                path=self.path,
                project_slug=project,
                model_name=model_name,
                df=df.copy(deep=True),
                training_kind=training_kind,
                scheme_labels=scheme_labels,
                col_label=col_label,
                col_text=col_text,
                base_model=base_model,
                params=params,
                test_size=test_size,
                loss=loss,
                class_balance=class_balance,
                class_min_freq=class_min_freq,
                fp16=fp16,
            ),
            queue="gpu",
        )
        del df

        params_db = LMParametersDbModel(**params.model_dump(), exclude_labels=exclude_labels or [])

        self.computing.append(
            LMComputing(
                user=user,
                model_name=model_name,
                unique_id=unique_id,
                time=current_date,
                kind="train_image",
                training_kind=training_kind,
                status="training",
                scheme=scheme,
                dataset=None,
                params=params_db.model_dump(),
                get_progress=self.get_progress(model_name, status="training"),
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
        training_kind: str,
        scheme_labels: list[str],
        col_label: str | None = None,
        batch_size: int = 32,
        status: str = "predicting",
        statistics: list | None = None,
        path_data: Path | None = None,
        path_train: Path | None = None,
        path_valid: Path | None = None,
        path_test: Path | None = None,
    ) -> None:
        """
        Initiate a prediction process with a model
        """

        if not (self.path.joinpath(name)).exists():
            raise NotFoundError("The model does not exist")

        if df is None and dataset != "all":
            raise Exception("Dataframe is required for this dataset")

        file_name = f"predict_{dataset}.parquet"

        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictImage(
                path=self.path.joinpath(name),
                dataset=dataset,
                df=df,
                training_kind=training_kind,
                scheme_labels=scheme_labels,
                col_text="text",
                col_label=col_label,
                col_id_external="id_external",
                col_datasets="dataset",
                basemodel=self.get_base_model(name),
                file_name=file_name,
                batch=batch_size,
                statistics=statistics,
                path_data=path_data,
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
                kind="predict_image",
                training_kind=training_kind,
                dataset=dataset,
                status=status,
                get_progress=self.get_progress(name, status=status),
            )
        )

    def delete(self, name: str) -> None:
        """
        Delete a model
        """
        if not self.models_service.delete_model(self.project_slug, name):
            raise NotFoundError("Model does not exist")
        try:
            if name and name != "":
                shutil.rmtree(self.path.joinpath(name))
                archive = (
                    Path(config.data_path)
                    / "projects"
                    / "static"
                    / self.project_slug
                    / f"{name}.tar.gz"
                )
                if archive.exists():
                    os.remove(archive)
        except Exception as e:
            raise Exception(f"Problem to delete model files : {e}")

    def rename(self, former_name: str, new_name: str) -> None:
        """
        Rename model
        """
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
        """
        Persist completed image-classification model events.
        """
        if element.status == "training":
            self.models_service.add_model(
                kind="image",
                name=element.model_name,
                user=element.user,
                project=self.project_slug,
                scheme=element.scheme or "default",
                params=element.params or {},
                path=str(self.path.joinpath(element.model_name)),
                status="trained",
            )
            self.models_service.set_model_params(
                self.project_slug,
                element.model_name,
                "compressed",
                True,
            )
            print("Image model trained")
        if element.status == "predicting":
            if element.dataset == "annotable":
                self.models_service.set_model_params(
                    self.project_slug, element.model_name, "predicted", True
                )
            if element.dataset == "all":
                self.models_service.set_model_params(
                    self.project_slug, element.model_name, "predicted_all", True
                )
            print("Image prediction finished")

    def export_prediction(
        self,
        name: str,
        file_name: str = "predict.parquet",
        format: str = "parquet",
        col_id: str | None = None,
    ) -> FileResponse:
        """
        Export prediction
        """
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
            if col_id is not None:
                target_col = col_id.removeprefix("dataset_")
            if target_col is not None and "id_external" in df.columns:
                df.rename(columns={"id_external": target_col}, inplace=True)
                df = df[[target_col] + [c for c in df.columns if c != target_col]]
            if format == "csv":
                df.to_csv(out_path, index=False)
            else:
                df.to_excel(out_path, index=False)
        return FileResponse(path=out_path, filename=out_name)

    def export_image(self, name: str) -> StaticFileModel:
        """
        export model
        """
        file = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"
        if not Path(file).exists():
            raise NotFoundError("file does not exist")
        return StaticFileModel(
            name=f"{name}.tar.gz",
            path=f"{self.project_slug}/{name}.tar.gz",
        )

    def get_labels(self, model_name: str) -> list:
        with open(self.path.joinpath(model_name).joinpath("config.json"), "r") as f:
            r = json.load(f)
        return list(r["id2label"].values())

    def get_progress(self, model_name, status: str) -> Callable[[], Optional[float]]:
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

    def get_loss(self, model_name) -> dict | None:
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
                    for i in range(0, int((len(log)) / 2))
                ],
                columns=["epoch", "val_loss", "val_eval_loss"],
            ).to_json()
            result = json.loads(loss)
        except Exception:
            result = None
        self._loss_cache[model_name] = (now, result)
        return result

    def get_parameters(self, model_name) -> dict | None:
        path = self.path.joinpath(model_name).joinpath("parameters.json")
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def get_informations(self, model_name) -> ModelInformationsModel:
        if not self.exists(model_name):
            raise NotFoundError(f"The model {model_name} does not exist")
        metrics = get_model_metrics(self.path.joinpath(model_name))
        if metrics is None:
            metrics = {}
        for key in metrics:
            if "training_kind" not in metrics[key]:
                metrics[key]["training_kind"] = "multiclass"
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
        )

    def get_base_model(self, model_name) -> str:
        with open(self.path.joinpath(model_name).joinpath("parameters.json"), "r") as f:
            data = json.load(f)
            if "base_model" in data:
                return data["base_model"]
            raise ValueError("No base_model found in parameters.json.")

    def get_eval_ids(self, model_name: str) -> list[str]:
        path = self.path.joinpath(model_name).joinpath("test_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Evaluation ids file does not exist")
        return [str(i) for i in pd.read_csv(path, index_col=0).index]

    def get_train_ids(self, model_name: str) -> list[str]:
        path = self.path.joinpath(model_name).joinpath("train_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Training ids file does not exist")
        return [str(i) for i in pd.read_csv(path, index_col=0).index]

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

    def state(self) -> ImageModelsProjectStateModel:
        return ImageModelsProjectStateModel(
            options=self.base_models,
            available=self.available(),
            training=self.training(),
            base_parameters=self.params_default,
        )
