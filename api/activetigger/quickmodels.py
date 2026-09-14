import os
import pickle
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fastapi.responses import FileResponse
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from activetigger.config import config
from activetigger.data import Data
from activetigger.datamodels import (
    KnnParams,
    LMComputingOutModel,
    LogisticL1Params,
    LogisticL2Params,
    ModelDescriptionModel,
    ModelInformationsModel,
    ModelScoresModel,
    Multi_naivebayesParams,
    PredictedLabel,
    QuickModelComputed,
    QuickModelComputing,
    QuickModelsProjectStateModel,
    RandomforestParams,
    TextDatasetModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.errors import InvalidInputError, NotFoundError
from activetigger.functions import concat_text_columns, get_model_metrics
from activetigger.queue_manager import Queue
from activetigger.tasks.predict_ml import PredictMLMultiClass
from activetigger.tasks.predict_with_features import PredictWithFeatures
from activetigger.tasks.train_ml import TrainMLMultiClass

if TYPE_CHECKING:
    from activetigger.features import Features


class QuickModels:
    """
    Module to manage quickmodels
    - define available models
    - save a quickmodel/user
    - train quickmodels
    """

    path: Path
    queue: Queue
    available_models: dict[str, Any]
    computing: list
    loaded: dict
    language_models_service: ModelsService
    features: "Features | None"

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        db_manager: DatabaseManager,
        features: "Features | None" = None,
    ) -> None:
        """
        Init Quickmodels class.

        `features` is the project-level `Features` manager. It is only
        required by `predict_on_dataset`, which delegates to it to build
        the per-feature compute specs and to source the text/paths
        series for the dataset being predicted on.
        """
        self.path: Path = path.joinpath("quickmodels")
        if not self.path.exists():
            os.mkdir(self.path)
        self.project_slug = project_slug
        self.language_models_service = db_manager.language_models_service
        self.computing = computing
        self.queue = queue
        self.features = features

        # Models and default parameters
        self.available_models = {
            "logistic-l1": LogisticL1Params(costLogL1=1),
            "logistic-l2": LogisticL2Params(costLogL2=1),
            "knn": KnnParams(n_neighbors=3),
            "randomforest": RandomforestParams(n_estimators=500, max_features=None),
            "multi_naivebayes": Multi_naivebayesParams(alpha=1, fit_prior=True, class_prior=None),
        }

        self.loaded = {}

    def compute_quickmodel(
        self,
        project_slug: str,
        user: str,
        scheme: str,
        features: list,
        name: str,
        model_type: str,
        df: DataFrame,
        col_labels: str,
        col_features: list,
        standardize: bool = True,
        model_params: dict | None = None,
        cv10: bool = False,
        balance_classes: bool = False,
        exclude_labels: list[str] = [],
        test_size: float = 0.2,
        retrain: bool = False,
        texts: pd.Series | None = None,
    ) -> str:
        """
        Add a new quickmodel for a user and a scheme
        """
        X, Y, labels = self.transform_data(df, col_labels, col_features, standardize)

        # default parameters
        if model_params is None:
            model_params = self.available_models[model_type].dict()

        # Select model
        if model_type == "knn":
            params_knn = KnnParams(**model_params)
            model = KNeighborsClassifier(n_neighbors=int(params_knn.n_neighbors), n_jobs=-1)
            balance_classes = False  # Force the parameter to be set as False
            model_params = params_knn.model_dump()

        if model_type == "logistic-l1":
            params_libL1 = LogisticL1Params(**model_params)
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=params_libL1.costLogL1,
                class_weight="balanced" if balance_classes else None,
                n_jobs=-1,
                random_state=config.random_seed,
            )
            model_params = params_libL1.model_dump()

        if model_type == "logistic-l2":
            params_libL2 = LogisticL2Params(**model_params)
            model = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=params_libL2.costLogL2,
                class_weight="balanced" if balance_classes else None,
                n_jobs=-1,
                random_state=config.random_seed,
            )
            model_params = params_libL2.model_dump()

        if model_type == "randomforest":
            # params  Num. trees mtry  Sample fraction
            # Number of variables randomly sampled as candidates at each split:
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            params_rf = RandomforestParams(**model_params)
            model = RandomForestClassifier(
                n_estimators=int(params_rf.n_estimators),
                max_features=(
                    int(params_rf.max_features) if params_rf.max_features is not None else None
                ),
                class_weight="balanced"
                if balance_classes
                else None,  # AM: Need to choose between balanced and balanced_subsample
                n_jobs=-1,
                random_state=config.random_seed,
            )
            model_params = params_rf.model_dump()

        if model_type == "multi_naivebayes":
            # small workaround for parameters
            params_nb = Multi_naivebayesParams(**model_params)
            if params_nb.class_prior is not None:
                class_prior = params_nb.class_prior
            else:
                class_prior = None
            # Only with dtf or tfidf for features
            # TODO: calculate class prior for docfreq & termfreq
            model = MultinomialNB(
                alpha=params_nb.alpha,
                fit_prior=params_nb.fit_prior,
                class_prior=class_prior,
            )
            balance_classes = False  # Force the parameter to be set as False
            model_params = params_nb.model_dump()

        # launch the compuation (model + statistics) as a future process
        args = {
            "model": model,  # ty: ignore[possibly-unresolved-reference]
            "X": X,
            "Y": Y,
            "labels": labels,
            "cv10": cv10,
            "balance_classes": balance_classes,
            "path": self.path,
            "name": name,
            "retrain": retrain,
            "scheme": scheme,
            "model_type": model_type,
            "user": user,
            "standardize": standardize,
            "features": features,
            "model_params": model_params,
            "texts": texts,
            "random_seed": config.random_seed,
            "exclude_labels": exclude_labels,
            "test_size": test_size,
        }
        unique_id = self.queue.add_task("train_quickmodel", project_slug, TrainMLMultiClass(**args))
        del args

        req = QuickModelComputing(
            status="training",
            user=user,
            unique_id=unique_id,
            time=datetime.now(timezone.utc),
            kind="train_quickmodel",
            scheme=scheme,
            model_type=model_type,
            name=name,
            features=features,
            labels=labels,
            model_params=model_params,
            standardize=standardize,
            cv10=cv10,
            balance_classes=balance_classes,
            exclude_labels=exclude_labels,
            test_size=test_size,
            retrain=retrain,
            dataset="train",
        )
        self.computing.append(req)
        return unique_id

    def add(self, element: QuickModelComputing) -> None:
        """
        Add computed model in the database
        """
        model_path = self.path.joinpath(element.name)
        self.language_models_service.add_model(
            kind="quickmodel",
            name=element.name,
            user=element.user,
            project=self.project_slug,
            scheme=element.scheme,
            params={**element.model_params, "exclude_labels": element.exclude_labels},
            path=str(model_path),
            status="trained",
            retrain=element.retrain,
        )

    def available(self) -> dict[str, list[ModelDescriptionModel]]:
        """
        Return available models per scheme
        """
        existing = self.language_models_service.available_models(self.project_slug, "quickmodel")
        r: dict[str, list[ModelDescriptionModel]] = {}
        for m in existing:
            # quickmodels are always attached to a scheme
            if m.scheme is None:
                continue
            r.setdefault(m.scheme, []).append(m)
        return r

    def get(self, name: str) -> QuickModelComputed:
        """
        Load the content of a specific model
        (cache in memory)
        """
        if not self.exists(name):
            raise NotFoundError("The model does not exist in database")
        if name in self.loaded:
            return self.loaded[name]
        else:
            path = self.path.joinpath(name)
            if not path.exists():
                raise NotFoundError("The model path does not exist")
            with open(path / "model.pkl", "rb") as file:
                sm: QuickModelComputed = pickle.load(file)
            return sm

    def get_prediction(self, name: str) -> DataFrame:
        """
        Get a specific quickmodel
        """
        sm = self.get(name)
        if sm.proba is None:
            raise ValueError("No probability available for this model")
        return sm.proba

    def get_prediction_element(self, model_name: str, element_id: Any) -> PredictedLabel:
        """
        Get prediction for a specific element
        """
        sm = self.get(model_name)
        if sm.proba is None:
            raise ValueError("No probability available for this model")
        if element_id not in sm.proba.index:
            raise NotFoundError("Element ID not found in the predictions")
        predicted_label = sm.proba.loc[element_id, "prediction"]
        predicted_proba = round(sm.proba.loc[element_id, predicted_label], 2)
        predicted_entropy = round(sm.proba.loc[element_id, "entropy"], 2)
        return PredictedLabel(
            label=predicted_label,
            proba=None if np.isnan(predicted_proba) else predicted_proba,
            entropy=None if np.isnan(predicted_entropy) else predicted_entropy,
        )

    def training(self) -> dict[str, LMComputingOutModel]:
        """
        Currently under training / predicting. Returns one entry per user
        so the frontend can render a progress bar for either kind of
        quickmodel process (train_quickmodel or predict_with_features).
        """
        kinds = {"train_quickmodel", "predict_with_features", "predict_quickmodel"}
        r: dict[str, LMComputingOutModel] = {}
        for e in self.computing:
            if getattr(e, "kind", None) not in kinds:
                continue
            get_progress = getattr(e, "get_progress", None)
            r[e.user] = LMComputingOutModel(
                name=getattr(e, "name", ""),
                status=getattr(e, "status", "training"),
                progress=get_progress() if get_progress else None,
            )
        return r

    def exists(self, name: str) -> bool:
        """
        Test if a quickmodel exists for a user/scheme
        """
        existing = self.language_models_service.available_models(self.project_slug, "quickmodel")
        return name in [m.name for m in existing]

    def transform_data(
        self, data, col_label, col_predictors, standardize
    ) -> tuple[DataFrame, pd.Series, list]:
        """
        Load data
        """
        f_na = data[col_predictors].isna().sum(axis=1) > 0
        if f_na.sum() > 0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        # normalize X data
        if standardize:
            scaler = StandardScaler()
            df = data[col_predictors]
            df_stand = scaler.fit_transform(df)
            df_pred = pd.DataFrame(df_stand, columns=df.columns, index=df.index)
        else:
            df_pred = data[col_predictors]

        # create global dataframe with no missing predictor
        df = pd.concat([data[~f_na][col_label], df_pred], axis=1)

        # data for training
        Y = df[col_label]
        X = df[col_predictors]
        labels = [str(i) for i in Y.unique() if pd.notna(i)]

        return X, Y, labels

    def export_prediction(self, name: str, format: str = "csv") -> tuple[BytesIO, dict[str, str]]:
        """
        Function to export the prediction of a quickmodel
        """
        # get data
        table = self.get_prediction(name)
        # convert to payload
        if format == "csv":
            output = BytesIO()
            pd.DataFrame(table).to_csv(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Type": "text/csv",
            }
            return output, headers
        elif format == "xlsx":
            output = BytesIO()
            pd.DataFrame(table).to_excel(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.xlsx"',
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
            return output, headers
        elif format == "parquet":
            output = BytesIO()
            pd.DataFrame(table).to_parquet(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.parquet"',
                "Content-Type": "application/octet-stream",
            }
            return output, headers
        else:
            raise ValueError("Format not supported")

    def export_prediction_file(
        self,
        name: str,
        dataset: str = "all",
        format: str = "parquet",
        col_id: str | None = None,
    ) -> FileResponse:
        """
        Serve the prediction parquet produced by `predict_on_dataset`.
        """
        file_name = f"predict_{dataset}.parquet"
        path = self.path.joinpath(name).joinpath(file_name)
        if not path.exists():
            raise FileNotFoundError(
                f"The file {file_name} does not exist for this model, please run prediction again."
            )

        if format not in ("parquet", "csv", "xlsx"):
            raise ValueError("Format not supported")

        if format == "parquet":
            return FileResponse(path=path, filename=file_name)

        ext = "csv" if format == "csv" else "xlsx"
        out_name = f"{file_name}.{ext}"
        out_path = self.path.joinpath(name).joinpath(out_name)
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

        return FileResponse(path=out_path, filename=out_name)

    def state(self) -> QuickModelsProjectStateModel:
        return QuickModelsProjectStateModel(
            options=self.available_models,
            available=self.available(),
            training=self.training(),
        )

    def delete(self, name: str) -> None:
        """
        Delete a specific quickmodel
        """
        if not self.exists(name):
            raise NotFoundError("The model does not exist")

        # delete from the database
        self.language_models_service.delete_model(self.project_slug, name)

        # delete from the filesystem
        model_path = self.path.joinpath(name)
        if model_path.exists():
            shutil.rmtree(model_path)

        # delete from the loaded cache
        if name in self.loaded:
            del self.loaded[name]

    def start_predicting_process(
        self,
        name: str,
        username: str,
        df: DataFrame,
        dataset: str,
        col_dataset: str,
        cols_features: list,
        col_label: str | None = None,
        col_text: str | None = None,
        statistics: list[str] | None = None,
    ) -> None:
        """
        Start the predicting process for a specific model
        """
        if not self.exists(name):
            raise NotFoundError("The model does not exist")
        sm = self.get(name)
        file_name = f"predict_{dataset}.parquet"
        unique_id = self.queue.add_task(
            "prediction",
            self.project_slug,
            PredictMLMultiClass(
                model=sm.model,
                df=df,
                col_dataset=col_dataset,
                col_features=cols_features,
                col_label=col_label,
                path=self.path.joinpath(name),
                file_name=file_name,
                statistics=statistics,
                col_text=col_text,
                exclude_labels=sm.exclude_labels,
            ),
            queue="cpu",
        )
        self.computing.append(
            QuickModelComputing(
                user=username,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="predict_quickmodel",
                status="predicting",
                name=name,
                dataset=dataset,
                features=sm.features,
                scheme=sm.scheme,
                model_type=sm.model_type,
                model_params=sm.model_params,
                labels=sm.labels,
            )
        )
        print("Predicting process started")

    def predict_on_dataset(
        self,
        name: str,
        username: str,
        dataset: str = "all",
    ) -> str:
        """
        Enqueue a `PredictWithFeatures` task that (re)computes the
        features the quickmodel was trained on for the requested dataset
        and applies the model to them. The proba DataFrame is saved to
        `<quickmodel_dir>/predict_<dataset>.parquet` and can be served
        by `export_prediction_file`.
        """
        sm = self._get_sm_for_predict(name)
        data = self._dataset_texts(dataset)
        return self._enqueue_predict_with_features(sm, username, dataset, data)

    def predict_on_external_dataset(
        self,
        name: str,
        username: str,
        external_dataset: TextDatasetModel,
    ) -> str:
        """
        Enqueue a `PredictWithFeatures` task on an uploaded external
        dataset. Mirrors the BERT flow: the file lives under the
        project's dataset dir, `cols_text` are concatenated into a
        single text column, and the resulting proba DataFrame is saved
        to `<quickmodel_dir>/predict_external.parquet`.
        """
        sm = self._get_sm_for_predict(name)
        assert self.features is not None  # invariant enforced above

        if not external_dataset.cols_text:
            raise ValueError("At least one text column must be selected")
        if not external_dataset.id:
            raise ValueError("An id column must be selected")
        if not external_dataset.filename:
            raise ValueError("No data file associated with the external dataset")

        path_data = self.features.data.get_path(external_dataset.filename)
        if not path_data.exists():
            raise FileNotFoundError(
                f"External dataset file '{external_dataset.filename}' not found. Upload it first."
            )

        df = Data.read_dataset(path_data)
        missing = [
            c for c in [*external_dataset.cols_text, external_dataset.id] if c not in df.columns
        ]
        if missing:
            raise NotFoundError(f"Columns not found in file: {missing}")

        texts = concat_text_columns(df, external_dataset.cols_text)
        texts.index = df[external_dataset.id].apply(str)
        texts.index.name = None
        texts = texts.dropna()
        if texts.index.duplicated().any():
            raise ValueError(
                f"Duplicate values in id column '{external_dataset.id}' — ids must be unique"
            )

        return self._enqueue_predict_with_features(sm, username, "external", texts)

    def mark_prediction_done(self, name: str, dataset: str) -> None:
        """
        Persist a `predicted_<dataset>` flag on the quickmodel row so
        the Prediction tab can warn before overwriting and the Export
        panel can gate the download button. Only tracked for datasets
        that ship a persistent file (`all` and `external`).
        """
        if dataset not in ("all", "external"):
            return
        try:
            self.language_models_service.set_model_params(
                self.project_slug, name, flag=f"predicted_{dataset}", value=True
            )
        except Exception as e:
            print(f"Could not set predicted_{dataset} flag for {name}: {e}")

    def _get_sm_for_predict(self, name: str) -> QuickModelComputed:
        if self.features is None:
            raise RuntimeError(
                "QuickModels was instantiated without a Features manager; "
                "prediction pipelines need it to build the compute specs."
            )
        sm = self.get(name)
        if not sm.features:
            raise ValueError(f"Quickmodel '{name}' has no features attached")
        return sm

    def _enqueue_predict_with_features(
        self,
        sm: QuickModelComputed,
        username: str,
        dataset: str,
        data: pd.Series,
    ) -> str:
        """
        Shared queue-insertion path for `predict_on_dataset` and
        `predict_on_external_dataset`. Builds the PredictWithFeatures
        task, wires the progress callable, and registers a
        `QuickModelComputing` entry so the frontend can render a bar.
        """
        assert self.features is not None
        specs = self.features.build_compute_specs(sm.features, data)

        task = PredictWithFeatures(
            specs=specs,
            path_process=self.features.path_all.parent,
            path_models=self.features.path_models,
            language=self.features.lang,
            quickmodel=sm,
            path_output=self.path.joinpath(sm.name),
            file_name=f"predict_{dataset}.parquet",
        )
        unique_id = self.queue.add_task(
            "predict_with_features", self.project_slug, task, queue="cpu"
        )

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
            QuickModelComputing(
                user=username,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="predict_with_features",
                status="predicting",
                name=sm.name,
                dataset=dataset,
                features=sm.features,
                scheme=sm.scheme,
                model_type=sm.model_type,
                model_params=sm.model_params,
                labels=sm.labels,
                get_progress=get_progress,
            )
        )
        return unique_id

    def _dataset_texts(self, dataset: str) -> pd.Series:
        """
        Assemble the input series consumed by feature computation for
        the requested dataset.

        For "all" we read `path_data_all` directly and index the series
        by `id_external` so the saved prediction parquet carries a
        meaningful row id (the CSV/xlsx exporter then promotes it to a
        first-position column named after `col_id`, matching the shape
        of the BERT prediction export).
        """
        assert self.features is not None
        if dataset == "all":
            df = pd.read_parquet(self.features.path_all, columns=["id_external", "text"])
            series = df["text"]
            series.index = df["id_external"].astype(str)
            series.index.name = "id_external"
            return series
        if dataset == "annotable":
            return self.features.concat_split_column("text", dataset="all")
        return self.features.concat_split_column("text", dataset=dataset)

    def get_informations(self, model_name) -> ModelInformationsModel:
        """
        Informations on the bert model from the files
        """

        if not self.exists(model_name):
            raise NotFoundError(f"The model {model_name} does not exist")
        sm = self.get(model_name)
        # params = self.get_parameters(model_name)
        metrics = get_model_metrics(self.path.joinpath(model_name))
        if metrics is None:
            metrics = {}

        return ModelInformationsModel(
            params={"exclude_labels": sm.exclude_labels},
            scores=ModelScoresModel(
                internalvalid_scores=metrics.get("trainvalid", None),
                valid_scores=metrics.get("valid", None),
                test_scores=metrics.get("test", None),
                outofsample_scores=metrics.get("outofsample", None),
                train_scores=metrics.get("train", None),
            ),
        )

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

    def drop_models(self, which: str = "all") -> None:
        """
        Drop all quickmodels of a specific kind
        """
        existing = self.language_models_service.available_models(self.project_slug, "quickmodel")
        if which == "all":
            for m in existing:
                self.delete(m.name)
        else:
            raise InvalidInputError("Kind not recognized")
