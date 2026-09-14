import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import pyarrow.parquet as pq
import regex
from pandas import DataFrame, Series

from activetigger.config import config
from activetigger.data import Data
from activetigger.datamodels import (
    FeatureComputing,
    FeatureDescriptionModelOut,
    FeaturesProjectStateModel,
)
from activetigger.db.projects import ProjectsService
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_bert_embeddings import ComputeBertEmbeddings
from activetigger.tasks.compute_clip_imagexp import ComputeClipImagexp
from activetigger.tasks.compute_dfm import ComputeDfm
from activetigger.tasks.compute_fasttext import ComputeFasttext
from activetigger.tasks.compute_multimodal import ComputeMultimodal
from activetigger.tasks.compute_sbert import ComputeSbert

# Experimental image projects: list of selectable image embedding models.
# Each entry maps a UI label to (open_clip model name, pretrained tag).
# See docs/image-projects-strategy.md.
IMAGE_EMBEDDING_MODELS_IMAGEXP: dict[str, dict[str, str]] = {
    "ViT-B-32/openai": {"model": "ViT-B-32", "pretrained": "openai"},
    "ViT-B-16/openai": {"model": "ViT-B-16", "pretrained": "openai"},
    "ViT-L-14/openai": {"model": "ViT-L-14", "pretrained": "openai"},
    "ViT-B-32/laion2b_s34b_b79k": {
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
    },
}
DEFAULT_IMAGE_EMBEDDING_MODEL_IMAGEXP = "ViT-B-32/openai"

# Experimental image projects: multimodal sentence-transformer models used
# for prompt-based image retrieval (§ docs/multimodal-prompt-selection.md).
# UI label -> full HF model name (loaded via SentenceTransformer(name)).
# The CLIP entries are native sentence-transformers models and work on CPU
# with no extra dependencies. BGE-VL and Qwen need trust_remote_code=True;
# BGE-VL additionally may require `pip install FlagEmbedding`.
MULTIMODAL_EMBEDDING_MODELS: dict[str, str] = {
    "CLIP-ViT-B-32": "sentence-transformers/clip-ViT-B-32",
    "CLIP-ViT-L-14": "sentence-transformers/clip-ViT-L-14",
    "BGE-VL-base": "BAAI/BGE-VL-base",
    "Qwen3-VL-Embedding-2B": "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen3-VL-Embedding-8B": "Qwen/Qwen3-VL-Embedding-8B",
}
DEFAULT_MULTIMODAL_EMBEDDING_MODEL = "CLIP-ViT-B-32"


class Features:
    """
    Manage project features
    Comment :
    - for the moment as a file
    - database for informations
    - use "__" as separator
    - add a column dataset to separate train, valid, test
    """

    # Feature kinds whose values can be reproduced from their stored parameters
    # (as opposed to `imported` / `prediction`, which come from external data
    # and cannot be recomputed by this project).
    COMPUTABLE_KINDS: frozenset[str] = frozenset(
        {
            "sentence-embeddings",
            "bert-embeddings",
            "fasttext",
            "dfm",
            "regex",
            "image-embeddings",
            "multimodal-embeddings",
        }
    )

    project_slug: str
    data: Data
    path_model: Path
    queue: Queue
    computing: list
    informations: dict
    content: DataFrame
    map: dict
    options: dict
    lang: str
    projects_service: ProjectsService
    n: int

    def __init__(
        self,
        project_slug: str,
        data: Data,
        models_path: Path,
        queue: Any,
        computing: list,
        db_manager,
        lang: str,
        kind: str = "text",
        languagemodels: Any = None,
    ) -> None:
        """
        Initit features
        """
        self.project_slug = project_slug
        self.projects_service = db_manager.projects_service
        self.data = data
        self.path_features = data.path_features
        self.path_all = data.path_data_all
        self.path_models = models_path
        self.queue = queue
        self.informations = {}

        self.lang = lang
        self.computing = computing
        self.kind = kind
        # Reference to the project's LanguageModels manager.
        self.languagemodels = languagemodels
        # Optional hook called as `on_delete(name, kind)` after a successful
        # feature delete. Project wires this up for image projects so that
        # deleting a multimodal-embeddings feature cascades to prompts.
        self.on_delete: Optional[Callable[[str, str | None], None]] = None
        # Optional hook called after a full reset_features_file() — all
        # features are gone, so any prompt/cache bound to them is stale.
        self.on_reset: Optional[Callable[[], None]] = None

        # Experimental image projects: replace the text embedding model list
        # with image embedding models, drop fasttext/dfm/regex (text-only).
        # See docs/image-projects-strategy.md.
        if kind == "image":
            self.options: dict = {
                "image-embeddings": {
                    "models": IMAGE_EMBEDDING_MODELS_IMAGEXP,
                    "default": DEFAULT_IMAGE_EMBEDDING_MODEL_IMAGEXP,
                },
                "multimodal-embeddings": {
                    "models": MULTIMODAL_EMBEDDING_MODELS,
                    "default": DEFAULT_MULTIMODAL_EMBEDDING_MODEL,
                },
                "dataset": {},
            }
        else:
            # load possible embeddings models
            fasttext_models = [f for f in os.listdir(self.path_models) if f.endswith(".bin")]
            # possibility to create a embeddings.yaml file to add models

            # options
            self.options = {
                "sentence-embeddings": {
                    "models": config.models_embeddings,
                    "default": "jinaai/jina-embeddings-v5-text-small",
                },
                "bert-embeddings": {
                    "models": [],
                    "pooling": ["mean", "cls"],
                },
                "fasttext": {"models": fasttext_models},
                "dfm": {
                    "tfidf": False,
                    "ngrams": 1,
                    "min_term_freq": 5,
                    "max_term_freq": 100,
                    "norm": None,
                    "log": None,
                },
                "regex": {"formula": None},
                "dataset": {},
            }

        # create the features file if not exist
        if not self.path_features.exists():
            self.create_features_file()
        self.map, self.n = self.get_map()

    def create_features_file(self):
        """
        Create the features file with the dataset information
        based on train, valid, test files
        """

        # clear database
        self.projects_service.delete_project_features(self.project_slug)

        # read file to create the structure
        train = pd.DataFrame(index=self.data.train.index)
        train["dataset"] = "train"

        to_concat = [train]

        if self.data.valid is not None:
            valid = pd.DataFrame(index=self.data.valid.index)
            valid["dataset"] = "valid"
            to_concat.append(valid)
        if self.data.test is not None:
            test = pd.DataFrame(index=self.data.test.index)
            test["dataset"] = "test"
            to_concat.append(test)

        df = pd.DataFrame(pd.concat(to_concat))
        df.to_parquet(self.path_features, index=True)
        del df

    def reset_features_file(self):
        """
        Reset the features file with only the dataset information
        based on train, valid, test files
        """
        if self.path_features.exists():
            self.path_features.unlink()
        self.projects_service.delete_project_features(self.project_slug)
        self.create_features_file()
        self.map, self.n = self.get_map()

        # cascade hook — every prompt/cache bound to a feature is now stale.
        on_reset = self.on_reset
        if on_reset is not None:
            try:
                on_reset()
            except Exception as ex:
                print(f"on_reset hook failed: {ex}")

    def get_map(self) -> tuple[dict, int]:
        """
        Get the structure of features from the parquet file
        """
        parquet_file = pq.ParquetFile(self.path_features)
        column_names = [i for i in parquet_file.schema.names if i != "dataset"]

        def find_strings_with_pattern(strings, pattern):
            matching_strings = [s for s in strings if regex.match(regex.escape(pattern), s)]
            return matching_strings

        var = set(
            [
                i.split("__")[0]
                for i in column_names
                if "__index" not in i and i not in ["id_internal", "id_external"]
            ]
        )
        dic = {i: find_strings_with_pattern(column_names, i) for i in var}
        num_rows = parquet_file.metadata.num_rows
        return dic, num_rows

    def exists(self, name: str) -> bool:
        """
        Check if a feature exists
        """
        return name in self.map

    def extension_running(self) -> bool:
        """
        True while an ExtendFeatures task (triggered by an eval set
        import) is rewriting the features file.
        """
        return any(getattr(c, "kind", None) == "extend_features" for c in self.computing)

    def _raise_if_extending(self) -> None:
        if self.extension_running():
            raise ValueError(
                "Features are being recomputed for a newly imported eval set; "
                "retry when this process is finished"
            )

    def add_predictions(self, predictions: dict) -> list:
        """
        Add BERT predictions to features
        """
        errors = []
        for f in predictions:
            try:
                # load and keep only label probability columns
                df = pd.read_parquet(predictions[f])
                cols_to_drop = [c for c in df.columns if c.startswith("entropy")]
                cols_to_drop += [
                    c
                    for c in df.columns
                    if c
                    in (
                        "prediction",
                        "text",
                        "dataset",
                        "id_external",
                        "GS-label",
                        "GS-label-non-dichotomized",
                    )
                    or not pd.api.types.is_numeric_dtype(df[c])
                ]
                df = df.drop(columns=set(cols_to_drop), errors="ignore")
                name = f.replace("__", "_")  # avoid __ in the name for features
                # if the feature already exists, delete it first
                if self.exists(name):
                    self.delete(name)
                # add it
                self.add(
                    name=name,
                    kind="prediction",
                    parameters={},
                    username="system",
                    new_content=df,
                )
            except Exception as ex:
                errors.append(f"Error in adding prediction : {str(ex)}")
        return errors

    def import_from_dataframe(
        self,
        *,
        name: str,
        id_column: str,
        df: DataFrame,
        columns: list[str] | None,
        username: str,
        source_file: str = "",
    ) -> str:
        """
        Import a pre-computed feature from a user-provided DataFrame.

        Matches rows on `id_external`. All target columns must be numeric.
        If `columns` is None or empty, every non-id column is imported as one
        multi-column embedding; otherwise only the selected columns are used.

        The stored feature is named `imported-<name>`. Returns that name.

        Raises ValueError on any validation issue.
        """
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Feature name is required")
        if not id_column:
            raise ValueError("ID column is required")
        if id_column not in df.columns:
            raise InvalidInputError(f"ID column '{id_column}' not found in file")

        if columns:
            mode = "select"
            target = [c for c in columns if c and c != id_column]
            missing_cols = [c for c in target if c not in df.columns]
            if missing_cols:
                raise NotFoundError(f"Columns not found in file: {missing_cols}")
        else:
            mode = "embedding"
            target = [c for c in df.columns if c != id_column]

        if not target:
            raise ValueError("No feature columns to import")

        non_numeric = [c for c in target if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns cannot be imported as features: {non_numeric}")

        ids = df[id_column].astype(str)
        if ids.duplicated().any():
            dups = ids[ids.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate values in ID column (e.g. {dups[:5]})")

        uploaded = df[target].copy()
        uploaded.index = ids
        uploaded.index.name = None

        project_ids = self.data.index["id_external"].astype(str)
        uploaded_ids = set(uploaded.index)
        missing = [i for i in project_ids if i not in uploaded_ids]
        if missing:
            raise ValueError(
                f"{len(missing)} project rows are missing from the file (e.g. {missing[:5]})"
            )

        aligned = uploaded.loc[pd.Index(project_ids)]
        aligned.index = self.data.index.index

        full_name = f"imported-{clean_name}"
        parameters: dict[str, Any] = {
            "id_column": id_column,
            "source_file": source_file,
            "mode": mode,
        }
        if mode == "embedding":
            parameters["embedding_size"] = len(target)
        else:
            parameters["columns"] = target

        self.add(
            name=full_name,
            kind="imported",
            username=username,
            parameters=parameters,
            new_content=aligned,
        )
        return full_name

    def add(
        self,
        name: str,
        kind: str,
        username: str,
        parameters: dict[str, Any],
        new_content: DataFrame | Series,
    ):
        """
        Add feature(s) after computing
        TODO : add in data
        """
        self._raise_if_extending()

        # test name
        if name in self.map:
            raise AlreadyExistsError("Feature already exists")

        # test length
        if len(new_content) != self.n:
            raise ValueError("Features don't have the right shape")

        # change type for series
        if type(new_content) is Series:
            new_content = pd.DataFrame(new_content)

        # change column name with a prefix
        new_content.columns = [  # ty: ignore[invalid-assignment]
            f"{name}__{i}" for i in new_content.columns
        ]

        # read data, add the feature to the dataset and save
        content = pd.read_parquet(self.path_features)
        content = pd.concat(
            [
                content[[i for i in content.columns if i not in new_content.columns]],
                new_content,
            ],
            axis=1,
        )
        content.to_parquet(self.path_features)
        del content

        # add informations to database
        self.projects_service.add_feature(
            project_slug=self.project_slug,
            kind=kind,
            name=name,
            parameters=parameters,
            user_name=username,
            data=list(new_content.columns),
        )

        # refresh the map
        self.map = self.get_map()[0]

    def delete(self, name: str):
        """
        Delete feature
        """
        self._raise_if_extending()

        if not self.exists(name):
            raise NotFoundError("Feature doesn't exist")

        # remember kind before the DB row is gone, so the on_delete hook
        # (e.g. prompts cascade) can dispatch on it.
        kind: str | None = None
        try:
            kind = self.projects_service.get_feature(self.project_slug, name).kind
        except Exception:
            kind = None

        col = self.map[name]
        # read data, delete columns and save
        content = pd.read_parquet(self.path_features)

        content[[i for i in content.columns if i not in col]].to_parquet(self.path_features)
        del content

        # delete from database
        self.projects_service.delete_feature(self.project_slug, name)

        # refresh the map
        self.map = self.get_map()[0]

        # cascade hook (set by Project for image projects to drop dependent prompts)
        on_delete = self.on_delete
        if on_delete is not None:
            try:
                on_delete(name, kind)
            except Exception as ex:
                print(f"on_delete hook failed for feature {name}: {ex}")

    def get(
        self, features: list | str, dataset: list | str = "all", keep_dataset_column: bool = False
    ) -> DataFrame:
        """
        Get content for specific features with a filter on dataset
        """
        features = [i for i in features if i is not None]
        if features == "all":
            features = list(self.map.keys())
        if type(features) is str:
            features = [features]

        # get needed columns
        cols = ["dataset"]
        missing = []
        for i in features:
            if i in self.map:
                cols += self.map[i]
            else:
                missing.append(i)
        if len(missing) > 0:
            # not necessary but to assure consistency
            raise NotFoundError(f"Missing features: {missing}. They may have been deleted.")

        # load only needed data from file
        data = pd.read_parquet(self.path_features, columns=cols)

        # filter on dataset
        if dataset == "annotable":
            if not keep_dataset_column:
                return data.drop(columns=["dataset"])
            return data

        if type(dataset) is str:
            raise Exception("Dataset should be a list of strings")

        data = data.loc[data["dataset"].isin(dataset)]
        if not keep_dataset_column:
            return data.drop(columns=["dataset"])
        return data

    def info(self, name: str):
        feature = self.projects_service.get_feature(self.project_slug, name)
        if feature is None:
            raise NotFoundError("Feature doesn't exist in database")
        return {
            "time": feature.time,
            "name": name,
            "kind": feature.kind,
            "username": feature.user,
            "parameters": feature.parameters,
            "columns": json.loads(feature.data),
        }

    def get_available(self) -> dict[str, FeatureDescriptionModelOut]:
        """
        Informations on features + update
        """
        return self.projects_service.get_project_features(self.project_slug)

    def get_column_raw(
        self, column_name: str, index: str = "annotable", add_real_index: bool = True
    ) -> Series:
        """
        Get column raw dataset
        """
        parquet_file = pq.ParquetFile(self.path_all)
        column_names = parquet_file.schema.names
        if column_name not in list(column_names):
            raise NotFoundError("Column doesn't exist")
        df = pd.read_parquet(self.path_all, columns=[column_name])
        if index == "annotable":  # filter only train id
            df_annotable = pd.read_parquet(self.path_features, columns=[])  # only the index
            return df.loc[df_annotable.index][column_name]
        elif index == "all":
            return df[column_name]
        else:
            raise InvalidInputError("Index not recognized")

    def current_user_processes(self, user: str):
        return [e for e in self.computing if getattr(e, "user", None) == user]

    def current_computing(self) -> dict[str, dict[str, str | None]]:
        computing = {
            e.name: {"progress": self.computing_progress(e.unique_id), "name": e.name}
            for e in self.computing
            if e.kind == "feature"
        }
        # eval set import/drop in progress: existing features are being
        # recomputed/rewritten for the new dataset shape. The `kind` key
        # lets the frontend render this entry differently.
        for e in self.computing:
            if e.kind == "extend_features":
                computing["extending features"] = {
                    "progress": self.computing_progress(e.unique_id),
                    "name": "extending features",
                    "kind": "extend_features",
                }
        return computing

    def __sbert_choose_model(self, parameters: dict):
        if (
            "model" not in parameters
            or parameters["model"] is None
            or parameters["model"] == "generic"
        ):
            return self.options["sentence-embeddings"]["default"]
        else:
            return parameters["model"]

    def __create_pretty_name(
        self, kind: str, name: str, use_default_name: bool, parameters: dict
    ) -> str:
        pretty_name = f"{kind}_{name}"
        if kind == "sentence-embeddings":
            if use_default_name:
                # use the model name
                model_name = self.__sbert_choose_model(parameters)
                pretty_name = f"SBERT_{model_name.split('/')[-1]}"
            else:
                pretty_name = f"SBERT_{name}"

            if "max_length_tokens" in parameters:
                pretty_name += f"-{int(parameters['max_length_tokens'])}tok"

        elif kind == "regex" and use_default_name and "value" in parameters:
            pretty_name = f"regex_{parameters['value']}"
        elif (
            kind == "dataset"
            and "dataset_col" in parameters
            and "dataset_type" in parameters
            and use_default_name
        ):
            pretty_name = f"{parameters['dataset_col']}_{parameters['dataset_type'].lower()}"
        elif kind == "bert-embeddings":
            model_name = str(parameters.get("model", "")) or "model"
            pooling = str(parameters.get("pooling", "mean"))
            short = model_name.split("/")[-1]
            if use_default_name:
                pretty_name = f"BERT_{short}_{pooling}"
            else:
                pretty_name = f"BERT_{name}_{pooling}"
        elif kind == "image-embeddings":
            ui_label = parameters.get("model") or DEFAULT_IMAGE_EMBEDDING_MODEL_IMAGEXP
            if use_default_name:
                pretty_name = f"CLIP_{ui_label.replace('/', '_')}"
            else:
                pretty_name = f"CLIP_{name}"
        elif kind == "multimodal-embeddings":
            ui_label = parameters.get("model") or DEFAULT_MULTIMODAL_EMBEDDING_MODEL
            if use_default_name:
                pretty_name = f"MM_{ui_label.replace('/', '_')}"
            else:
                pretty_name = f"MM_{name}"
        elif kind == "fasttext":
            if parameters["model"] is not None and parameters["model"] != "":
                short_model = (
                    parameters["model"].split("/")[-1]
                    if "/" in parameters["model"]
                    else parameters["model"]
                )
                name = f"fasttext_{name}_{short_model}"
        elif kind == "dfm":
            pass
        return pretty_name

    def compute(
        self,
        df: pd.Series,
        name: str,
        use_default_name: bool,
        kind: str,
        parameters: dict,
        username: str,
    ):
        """
        Compute new feature
        TODO : the parameters management is really bad
        """
        if len(self.current_user_processes(username)) > 0:
            raise ValueError("A process is already running")

        if kind not in {
            "sentence-embeddings",
            "bert-embeddings",
            "fasttext",
            "dfm",
            "regex",
            "dataset",
            "image-embeddings",
            "multimodal-embeddings",
        }:
            raise InvalidInputError("Kind not recognized")

        # Experimental image projects: gate text-only feature kinds.
        if self.kind == "image" and kind not in {
            "image-embeddings",
            "multimodal-embeddings",
            "dataset",
        }:
            raise ValueError(f"Feature kind '{kind}' is not supported on image projects")
        if self.kind != "image" and kind in {"image-embeddings", "multimodal-embeddings"}:
            raise ValueError(f"{kind} is only available on image projects")

        name = self.__create_pretty_name(kind, name, use_default_name, parameters)
        if self.exists(name):
            raise AlreadyExistsError("This name already exists")

        if kind == "regex":
            if "value" not in parameters:
                raise ValueError("No value for regex")

            pattern = regex.compile(parameters["value"])
            if parameters.get("regex_count", False):
                f = df.apply(lambda x: len(pattern.findall(x)))
            else:
                f = df.apply(lambda x: bool(pattern.search(x)))
            parameters = {
                "name": name,
                "kind": kind,
                "regex": parameters["value"],
                "mode": "count" if parameters.get("regex_count", False) else "presence",
                "count": int(f.sum()),
                "username": username,
            }
            self.add(name, kind, username, parameters, f)
            return None

        if kind == "dataset":
            # get the raw column for the train set
            column = self.get_column_raw(parameters["dataset_col"])

            # convert the column to a specific format
            if len(column.dropna()) != len(column):
                raise ValueError("Column contains null values")
            column_encoded: pd.DataFrame | pd.Series
            if parameters["dataset_type"] == "Numeric":
                try:
                    column_encoded = column.apply(float)
                except Exception:
                    raise Exception("The column can't be transform into numerical feature")
            else:
                column = column.apply(str)
                column_encoded = pd.get_dummies(
                    column, prefix=parameters["dataset_col"], drop_first=True
                )

            # add the feature to the project
            parameters = {
                "name": name,
                "kind": kind,
                "dataset_col": parameters["dataset_col"],
                "dataset_type": parameters["dataset_type"],
                "username": username,
            }
            self.add(name, kind, username, parameters, column_encoded)
            return None

        # features with queue
        unique_id = None

        if kind == "sentence-embeddings":
            model = self.__sbert_choose_model(parameters)

            if "max_length_tokens" not in parameters:
                parameters["max_length_tokens"] = 1024
            if "batch_size" not in parameters:
                parameters["batch_size"] = 32
            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeSbert(
                    texts=df,
                    path_process=self.path_all.parent,
                    model=model,
                    max_tokens=parameters["max_length_tokens"],
                    batch_size=int(parameters["batch_size"]),
                ),
                queue="gpu",
            )

            parameters = {
                "model": model,
                "name": name,
                "kind": kind,
                "username": username,
                "max_length_tokens": parameters["max_length_tokens"],
            }

        if kind == "bert-embeddings":
            if self.languagemodels is None:
                raise ValueError("No LanguageModels manager available for bert-embeddings")
            model_name = parameters.get("model")
            if not model_name:
                raise ValueError("A trained BERT model name must be provided")
            if not self.languagemodels.exists(model_name):
                raise ValueError(f"Trained BERT model '{model_name}' does not exist")

            pooling = parameters.get("pooling", "mean")
            if pooling not in {"cls", "mean"}:
                raise ValueError("pooling must be 'cls' or 'mean'")

            max_length_tokens = int(parameters.get("max_length_tokens", 512))
            batch_size = int(parameters.get("batch_size", 32))
            model_path = self.languagemodels.path.joinpath(model_name)

            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeBertEmbeddings(
                    texts=df,
                    path_process=self.path_all.parent,
                    model_path=model_path,
                    pooling=pooling,
                    batch_size=batch_size,
                    max_tokens=max_length_tokens,
                ),
                queue="gpu",
            )

            parameters = {
                "model": model_name,
                "pooling": pooling,
                "name": name,
                "kind": kind,
                "username": username,
                "max_length_tokens": max_length_tokens,
                "batch_size": batch_size,
            }

        if kind == "image-embeddings":
            # Resolve UI label -> (open_clip model, pretrained tag)
            ui_label = parameters.get("model") or DEFAULT_IMAGE_EMBEDDING_MODEL_IMAGEXP
            if ui_label == "generic":
                ui_label = DEFAULT_IMAGE_EMBEDDING_MODEL_IMAGEXP
            spec = IMAGE_EMBEDDING_MODELS_IMAGEXP.get(ui_label)
            if spec is None:
                raise InvalidInputError(f"Unknown image embedding model: {ui_label}")

            batch_size = int(parameters.get("batch_size", 16))
            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeClipImagexp(
                    paths=df,
                    path_process=self.path_all.parent,
                    model=spec["model"],
                    pretrained=spec["pretrained"],
                    batch_size=batch_size,
                ),
                queue="gpu",
            )

            parameters = {
                "model": ui_label,
                "name": name,
                "kind": kind,
                "username": username,
                "batch_size": batch_size,
            }

        if kind == "multimodal-embeddings":
            ui_label = parameters.get("model") or DEFAULT_MULTIMODAL_EMBEDDING_MODEL
            if ui_label == "generic":
                ui_label = DEFAULT_MULTIMODAL_EMBEDDING_MODEL
            hf_name = MULTIMODAL_EMBEDDING_MODELS.get(ui_label)
            if hf_name is None:
                raise InvalidInputError(f"Unknown multimodal embedding model: {ui_label}")

            batch_size = int(parameters.get("batch_size", 8))
            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeMultimodal(
                    paths=df,
                    path_process=self.path_all.parent,
                    model_name=hf_name,
                    batch_size=batch_size,
                ),
                queue="gpu",
            )

            parameters = {
                "model": ui_label,
                "hf_name": hf_name,
                "name": name,
                "kind": kind,
                "username": username,
                "batch_size": batch_size,
            }

        if kind == "fasttext":
            unique_id = self.queue.add_task(
                "feature",
                self.project_slug,
                ComputeFasttext(
                    texts=df,
                    language=self.lang,
                    path_process=self.path_all.parent,
                    path_models=self.path_models,
                    model=parameters["model"],
                ),
            )

            parameters = {
                "model": parameters["model"],
                "name": name,
                "kind": kind,
                "username": username,
            }

        if kind == "dfm":
            args = parameters.copy()
            args["texts"] = df
            args["language"] = self.lang
            unique_id = self.queue.add_task("feature", self.project_slug, ComputeDfm(**args))
            del args
            parameters = {
                "name": name,
                "kind": kind,
                "username": username,
                "tfidf": parameters.get("dfm_tfidf", False),
                "dfm_ngrams": parameters.get("dfm_ngrams", 1),
                "dfm_max_term_freq": parameters.get("dfm_max_term_freq", 100),
                "dfm_min_term_freq": parameters.get("dfm_min_term_freq", 5),
                "dfm_norm": parameters.get("dfm_norm", None),
                "dfm_log": parameters.get("dfm_log", None),
            }

        if unique_id:
            self.computing.append(
                FeatureComputing(
                    unique_id=unique_id,
                    kind="feature",
                    parameters=parameters,
                    type=kind,
                    user=username,
                    name=name,
                    time=datetime.now(timezone.utc),
                )
            )
            return None
        raise ValueError("Error in the process")

    def build_compute_specs(self, names: list[str], data: Series) -> list[dict[str, Any]]:
        """
        Assemble the `specs` list consumed by `PredictWithFeatures` for a
        set of existing feature names.

        For each name:
          - checks it exists in the project,
          - checks its `kind` is in `COMPUTABLE_KINDS` (imported /
            prediction features can't be recomputed → `ValueError`),
          - retrieves the stored parameters from the DB,
          - attaches the shared `data` series (texts or image paths),
          - resolves the trained-model path for `bert-embeddings` so
            the task stays independent of `LanguageModels`.
        """
        if not names:
            raise ValueError("No features to compute")

        specs: list[dict[str, Any]] = []
        for name in names:
            if not self.exists(name):
                raise ValueError(f"Feature '{name}' does not exist")
            feature = self.projects_service.get_feature(self.project_slug, name)
            if feature is None:
                raise ValueError(f"Feature '{name}' has no DB record")
            kind = feature.kind
            if kind not in self.COMPUTABLE_KINDS:
                raise ValueError(f"Feature '{name}' has kind '{kind}' which is not computable")

            spec: dict[str, Any] = {
                "name": name,
                "kind": kind,
                "parameters": feature.parameters,
                "data": data,
            }
            if kind == "bert-embeddings":
                if self.languagemodels is None:
                    raise ValueError(
                        f"Feature '{name}' needs a LanguageModels manager to resolve its model path"
                    )
                spec["model_path"] = self.languagemodels.path.joinpath(feature.parameters["model"])
            specs.append(spec)
        return specs

    def concat_split_column(self, column: str, dataset: str = "all") -> Series:
        """
        Concatenate `column` across the requested split(s).

        `dataset` ∈ {"train", "valid", "test", "all"}. "all" returns
        train + (valid) + (test) in the natural order — the same layout
        used by the features parquet file.
        """
        if dataset == "train":
            return self.data.train[column]
        if dataset == "valid":
            if self.data.valid is None:
                raise ValueError("No valid dataset available")
            return self.data.valid[column]
        if dataset == "test":
            if self.data.test is None:
                raise ValueError("No test dataset available")
            return self.data.test[column]
        if dataset != "all":
            raise InvalidInputError(f"Unknown dataset '{dataset}'")
        parts: list[Series] = [self.data.train[column]]
        if self.data.valid is not None:
            parts.append(self.data.valid[column])
        if self.data.test is not None:
            parts.append(self.data.test[column])
        return pd.concat(parts)

    def computing_progress(self, unique_id: str) -> str | None:
        """
        Get the progress of a computing feature
        """
        try:
            with open(self.path_all.parent.joinpath(unique_id), "r") as f:
                r = f.read()
            return r
        except Exception:
            return None

    def state(self) -> FeaturesProjectStateModel:
        # Refresh trained-BERT list each call: models are trained out-of-band
        # via LanguageModels, so the dropdown must reflect what's available now.
        if "bert-embeddings" in self.options and self.languagemodels is not None:
            self.options["bert-embeddings"]["models"] = self._available_bert_models()
        return FeaturesProjectStateModel(
            options=self.options,
            available=list(self.map.keys()),
            training=self.current_computing(),
        )

    def _available_bert_models(self) -> list[str]:
        """
        Flat list of trained BERT model names available as embedding sources.
        """
        if self.languagemodels is None:
            return []
        models: list[str] = []
        for _scheme, by_name in self.languagemodels.available().items():
            models.extend(by_name.keys())
        return sorted(set(models))
