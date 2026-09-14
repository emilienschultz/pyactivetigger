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
    LanguageModelsProjectStateModel,
    LMComputing,
    LMComputingOutModel,
    LMParametersDbModel,
    LMParametersModel,
    LMStatusModel,
    ModelInformationsModel,
    ModelScoresModel,
    StaticFileModel,
    TextDatasetModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.functions import get_model_metrics
from activetigger.queue_manager import Queue
from activetigger.tasks.predict_bert import PredictBertMultiClass
from activetigger.tasks.train_bert import TrainBert


class LanguageModels:
    """
    Module to manage languagemodels
    """

    project_slug: str
    path: Path
    queue: Queue
    computing: list
    language_models_service: ModelsService
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
        self.language_models_service = db_manager.language_models_service
        self.path: Path = Path(path).joinpath("bert")
        self.cache_predictions = {}
        self._loss_cache: dict[str, Tuple[float, dict | None]] = {}  # model_name -> (time, loss)
        self._loss_cache_interval: float = 5  # seconds

        # load the list of models; this runs inside Project.__init__, so a
        # missing or malformed CSV must fall back to the bundled defaults
        # instead of raising (which would break every project open)
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
            {
                "name": "flaubert/flaubert_base_cased",
                "priority": 7,
                "comment": "",
                "language": "fr",
            },
        ]
        if list_models is not None and Path(list_models).exists():
            try:
                df_models = pd.read_csv(list_models)
                if "name" not in df_models.columns:
                    raise ValueError("missing required column 'name'")
                self.base_models = cast(list[dict[str, Any]], df_models.to_dict(orient="records"))
            except Exception as ex:
                print(f"Could not read models list {list_models}, using defaults: {ex}")

        # create the directory for models; exist_ok handles the TOCTOU race
        # when two concurrent project loads both pass the exists() check.
        os.makedirs(self.path, exist_ok=True)

    def available(self) -> dict[str, dict[str, LMStatusModel]]:
        """
        Available models
        """
        models = self.language_models_service.available_models(self.project_slug, "bert")
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
        Check if a model exists
        """
        return self.language_models_service.model_exists(self.project_slug, name)

    def training(self) -> dict[str, LMComputingOutModel]:
        """
        Currently under training
        - name
        - progress if available
        - loss if available
        """

        r = {
            e.user: LMComputingOutModel(
                name=e.model_name,
                status=e.status,
                progress=e.get_progress() if e.get_progress else None,
                loss=self.get_loss(e.model_name),
                epochs=e.params["epochs"] if e.params else None,
            )
            for e in self.computing
            if e.kind in ["bert", "train_bert", "predict_bert"]
        }
        return r

    def delete(self, name: str) -> None:
        """
        Delete bert model.

        Cleans up filesystem first (model directory and optional tar.gz export),
        then removes the DB row. Each step is independent so a partial state
        (e.g. orphan files with no DB row, or a missing tar.gz) can still be
        recovered by a retry.
        """
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

        # remove from database last; treat a missing row as a no-op rather than
        # an error so orphan files can still be cleaned up by a retry.
        db_removed = self.language_models_service.delete_model(self.project_slug, name)

        if not db_removed and not had_files:
            raise NotFoundError("Model does not exist")

        if errors:
            raise Exception(f"Problem to delete model files : {'; '.join(errors)}")

    def current_user_processes(self, user: str) -> list[LMComputing]:
        """
        Get the user current processes
        """
        return [e for e in self.computing if e.user == user]

    def estimate_memory_use(self, model: str, kind: str = "train") -> int:
        """
        Estimate the GPU memory in Gb needed to train a model
        For the moment dummy values
        TODO : implement
        """
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
        training_kind: str,
        scheme_labels: list[str],
        col_text: str,
        col_label: str,
        params: LMParametersModel,
        use_dichotomization: bool,
        label_for_dichotomization: str | None = None,
        base_model: str = "almanach/camembert-base",
        test_size: float = 0.2,
        num_min_annotations: int = 10,
        num_min_annotations_per_label: int = 5,
        loss: str = "cross_entropy",
        max_length: int = 512,
        auto_max_length: bool = False,
        class_balance: bool = False,
        class_min_freq: int = 1,
        exclude_labels: list[str] | None = None,
    ) -> str:
        """
        Manage the training of a model from the API
        """

        # check the size of training data
        if len(df.dropna()) < num_min_annotations:
            raise Exception(f"Less than {num_min_annotations} elements annotated")

        # name integrating the scheme & user + date
        current_date = datetime.now(timezone.utc)
        model_name = name

        # check if a project not already exist
        if self.language_models_service.model_exists(project, model_name):
            raise AlreadyExistsError("A model with this name already exists")

        # force CPU when cpu_only mode
        if config.cpu_only:
            params.gpu = False

        # if GPU requested, test if enough memory is available (to avoid CUDA out of memory)
        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if self.estimate_memory_use(model_name, kind="train") > mem.available_memory:
                raise Exception("Not enough GPU memory available. Wait or reduce batch.")

        # launch as a independant process
        if training_kind not in ["multilabel", "multiclass"]:
            raise Exception("training_kind must be multilabel or multiclass")
        unique_id = self.queue.add_task(
            "training",
            project,
            TrainBert(
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
                max_length=max_length,
                auto_max_length=auto_max_length,
                class_balance=class_balance,
                class_min_freq=class_min_freq,
                use_dichotomization=use_dichotomization,
                label_for_dichotomization=label_for_dichotomization,
            ),
            queue="gpu",
        )
        del df

        # add flags in params
        params = LMParametersDbModel(**params.model_dump(), exclude_labels=exclude_labels or [])

        # Update the queue state
        self.computing.append(
            LMComputing(
                user=user,
                model_name=model_name,
                unique_id=unique_id,
                time=current_date,
                kind="train_bert",
                training_kind=training_kind,
                status="training",
                scheme=scheme,
                dataset=None,
                params=params.model_dump(),
                get_progress=self.get_progress(model_name, status="training"),
            )
        )
        return unique_id

    def clean_files_valid(self, model_name: str, dataset: str):
        """
        Clean previous files for validation or test dataset
        #TODO : CHECK STATISTICS
        """
        if dataset not in ["valid", "test"]:
            raise Exception("Dataset should be 'valid' or 'test'")
        if (self.path.joinpath(model_name).joinpath(f"predict_{dataset}.parquet")).exists():
            os.remove(self.path.joinpath(model_name).joinpath(f"predict_{dataset}.parquet"))
        if (self.path.joinpath(model_name).joinpath(f"metrics_{dataset}.json")).exists():
            os.remove(self.path.joinpath(model_name).joinpath(f"metrics_{dataset}.json"))

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
        external_dataset: None | TextDatasetModel = None,
        path_train: Path | None = None,
        path_valid: Path | None = None,
        path_test: Path | None = None,
    ) -> str:
        """
        Start predicting process
        """
        if not (self.path.joinpath(name)).exists():
            raise NotFoundError("The model does not exist")

        # remove stale progress left by a previous run whose worker died
        # before cleanup; otherwise the frontend shows the old percentage
        stale_progress = self.path.joinpath(name).joinpath("progress_predict")
        if stale_progress.exists():
            stale_progress.unlink()

        # case of external loading
        if df is None and dataset not in ["all", "external"]:
            raise Exception("Dataframe is required for this dataset")

        file_name = f"predict_{dataset}.parquet"

        # load the model
        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictBertMultiClass(
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
                kind="predict_bert",
                training_kind=training_kind,
                dataset=dataset,
                status=status,
                get_progress=self.get_progress(name, status=status),
            )
        )
        return unique_id

    def rename(self, former_name: str, new_name: str) -> None:
        """
        Rename a model (copy it)
        """
        # get model
        model = self.language_models_service.get_model(self.project_slug, former_name)
        if model is None:
            raise NotFoundError("Model does not exist")
        if (Path(model.path) / "status.log").exists():
            raise Exception("Model is currently computing")
        self.language_models_service.rename_model(self.project_slug, former_name, new_name)
        model_path = Path(model.path)
        new_path = model_path.parent / model_path.name.replace(former_name, new_name)
        os.rename(model_path, new_path)

    def export_prediction(
        self,
        name: str,
        file_name: str = "predict.parquet",
        format: str = "parquet",
        col_id: str | None = None,
    ) -> FileResponse:
        """
        Export predict file if exists
        """
        # get the prediction file
        path = self.path.joinpath(name).joinpath(file_name)
        if not path.exists():
            raise FileNotFoundError(
                f"The file {file_name} does not exist for this model, please run prediction again."
            )

        if format not in ("parquet", "csv", "xlsx"):
            raise Exception("Format not supported")

        # parquet: serve the source file directly (no conversion)
        if format == "parquet":
            return FileResponse(path=path, filename=file_name)

        # cache the converted file next to the source; regenerate only if stale
        ext = "csv" if format == "csv" else "xlsx"
        out_name = f"{file_name}.{ext}"
        out_path = self.path.joinpath(name).joinpath(out_name)
        if not out_path.exists() or out_path.stat().st_mtime < path.stat().st_mtime:
            df = pd.read_parquet(path)
            # rename id_external to original column name and move it to the first
            # position for consistency with other exports. For predictions on an
            # external dataset, prefer the id column from the sidecar metadata
            # (the external dataset's own id) over the project's internal col_id.
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

    def export_bert(self, name: str) -> StaticFileModel:
        """
        Export bert archive if exists
        """
        file = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"

        if not Path(file).exists():
            raise NotFoundError("file does not exist")
        return StaticFileModel(
            name=f"{name}.tar.gz",
            path=f"{self.project_slug}/{name}.tar.gz",
        )

    def add(self, element: LMComputing) -> None:
        """
        Manage computed process for model
        """
        if element.status == "training":
            # add in database
            self.language_models_service.add_model(
                kind="bert",
                name=element.model_name,
                user=element.user,
                project=self.project_slug,
                scheme=element.scheme or "default",
                params=element.params or {},
                path=str(self.path.joinpath(element.model_name)),  # TODO refactor
                status="trained",
            )

            # TODO test if the model is already compressed
            self.language_models_service.set_model_params(
                self.project_slug,
                element.model_name,
                "compressed",
                True,
            )
            print("Model trained")
        if element.status == "testing":
            self.language_models_service.set_model_params(
                self.project_slug,
                element.model_name,
                flag="tested",
                value=True,
            )
            print("Testing finished")
        if element.status == "predicting":
            # update flag if prediction was on the annotable dataset
            if element.dataset == "annotable":
                self.language_models_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted",
                    value=True,
                )
            # update flag if prediction was on the complete dataset
            if element.dataset == "all":
                self.language_models_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted_all",
                    value=True,
                )
            # update flag if there is a prediction in an external dataset
            if element.dataset == "external":
                self.language_models_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted_external",
                    value=True,
                )
            print("Prediction finished")

    def get_labels(self, model_name: str) -> list:
        """
        Get the labels of the model
        """
        with open(self.path.joinpath(model_name).joinpath("config.json"), "r") as f:
            r = json.load(f)
        return list(r["id2label"].values())

    def get_progress(self, model_name, status: str) -> Callable[[], Optional[float]]:
        """
        Get progress when training
        (different cases)
        """
        if status == "training":
            path_model = self.path.joinpath(model_name).joinpath("progress_train")
        elif status == "predicting":
            path_model = self.path.joinpath(model_name).joinpath("progress_predict")
        else:
            raise InvalidInputError("Status not recognized")

        def progress_predicting():
            if path_model.exists():
                r = path_model.read_text()
                if r.strip() == "":
                    r = 0
                return float(r)
            return None

        return progress_predicting

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
        """
        Get the parameters of the model
        """
        path = self.path.joinpath(model_name).joinpath("parameters.json")

        if not path.exists():
            return None

        with open(path, "r") as jsonfile:
            params = json.load(jsonfile)
        return params

    def get_informations(self, model_name) -> ModelInformationsModel:
        """
        Informations on the bert model from the files
        """

        if not self.exists(model_name):
            raise NotFoundError(f"The model {model_name} does not exist")

        metrics = get_model_metrics(self.path.joinpath(model_name))
        if metrics is None:
            metrics = {}

        # TODO: delete this hotfix to ensure that previously trained models will not trigger errors
        for key in metrics:
            if "training_kind" not in metrics[key]:
                metrics[key]["training_kind"] = "multiclass"

        db_params = (
            self.language_models_service.get_model_db_parameters(self.project_slug, model_name)
            or {}
        )

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

    def get_base_model(self, model_name) -> dict:
        """
        Get the base model for a model
        """
        with open(self.path.joinpath(model_name).joinpath("parameters.json"), "r") as jsonfile:
            data = json.load(jsonfile)
            if "base_model" in data:
                return data["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")

    def state(self) -> LanguageModelsProjectStateModel:
        """
        Get the state of the module
        """
        return LanguageModelsProjectStateModel(
            options=self.base_models,
            available=self.available(),
            training=self.training(),
            base_parameters=self.params_default,
        )

    def get_eval_ids(self, model_name: str) -> list[str]:
        """
        Get the evaluation ids from the eval dataset of the model
        """
        path = self.path.joinpath(model_name).joinpath("test_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Evaluation ids file does not exist")

        return [str(i) for i in pd.read_csv(path, index_col=0).index]

    def get_train_ids(self, model_name: str) -> list[str]:
        """
        Get the training ids from the train dataset of the model
        """
        path = self.path.joinpath(model_name).joinpath("train_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Training ids file does not exist")
        return [str(i) for i in pd.read_csv(path, index_col=0).index]

    def get_prediction(self, model_name: str, cache_time: int = 120) -> pd.DataFrame:
        """
        Get the prediction dataframe for a model with a cache to avoid reloading the file too often
        """

        # update cache
        for key in list(self.cache_predictions.keys()):
            timestamp, _ = self.cache_predictions[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() > cache_time:
                del self.cache_predictions[key]

        # return from cache
        if model_name in self.cache_predictions:
            _, cached_df = self.cache_predictions[model_name]
            return cached_df

        # load, cache and return prediction file
        path = self.path.joinpath(model_name).joinpath("predict_annotable.parquet")
        if not path.exists():
            raise FileNotFoundError("Prediction file does not exist")
        df = pd.read_parquet(path)
        self.cache_predictions[model_name] = (datetime.now(timezone.utc), df)
        return df
