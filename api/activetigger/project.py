import json
import math
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pandas import DataFrame
from task_manager.tasks.create_project_task import CreateProjectTaskInput, create_project_task

from activetigger.bertopic_manager import Bertopic
from activetigger.config import config
from activetigger.data import Data
from activetigger.datamodels import (
    ActiveModel,
    AuthUserModel,
    BertModelModel,
    BertopicComputing,
    ElementInModel,
    ElementOutModel,
    EvalSetDataModel,
    EvalSetImageModel,
    EventsModel,
    ExportGenerationsParams,
    FeatureComputing,
    GenerationComputing,
    GenerationModel,
    GenerationRequest,
    GenerationResult,
    ImageModelModel,
    LexicometricsComputing,
    LMComputing,
    NerModelModel,
    NextInModel,
    NextProjectStateModel,
    PredictedLabel,
    ProcessComputing,
    ProjectBaseModel,
    ProjectCreatingModel,
    ProjectDescriptionModel,
    ProjectionComputing,
    ProjectionOutModel,
    ProjectionOutModelNode,
    ProjectModel,
    ProjectStateModel,
    ProjectUpdateModel,
    PromptComputing,
    QuickModelComputing,
    QuickModelInModel,
    StaticFileModel,
    TextDatasetModel,
    UpdateComputing,
)
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.features import Features
from activetigger.functions import (
    clean_regex,
    dichotomize,
    get_dir_size,
    get_number_occurrences_per_label,
    regex_contains,
    remove_labels_without_enough_annotations,
    sanitize_query_expression,
)
from activetigger.generation.generations import Generations
from activetigger.imagemodels import ImageModels
from activetigger.languagemodels import LanguageModels
from activetigger.lexicometrics import Lexicometrics
from activetigger.messages import Messages
from activetigger.monitoring import Monitoring
from activetigger.nermodels import NerModels
from activetigger.projections import Projections
from activetigger.prompts import BINDABLE_FEATURE_KINDS, Prompts
from activetigger.queue_manager import Queue
from activetigger.quickmodels import QuickModels
from activetigger.schemes import Schemes
from activetigger.tasks.add_evalset import AddEvalSet, AddEvalSetImage
from activetigger.tasks.create_project import CreateProject, CreateProjectImagexp
from activetigger.tasks.extend_features import ExtendFeatures
from activetigger.tasks.generate_call import GenerateCall
from activetigger.tasks.update_datasets import UpdateDatasets
from activetigger.users import Users


class Errors:
    """
    Runtime error object
    """

    def __init__(self, timeout: int = 15) -> None:
        """
        Initialize the error stack
        """
        self.timeout = timeout
        self.__stack: list[list] = []

    def add(self, message: str) -> None:
        """
        Add an error to the stack
        """
        self.__stack.append([message, datetime.now(config.timezone)])

    def clean(self) -> None:
        """
        Clean old errors
        """
        now = datetime.now(config.timezone)
        self.__stack = [e for e in self.__stack if e[1] >= now - timedelta(minutes=self.timeout)]

    def state(self) -> list[list]:
        """
        Get the current stack
        """
        self.clean()
        return self.__stack


def _detect_span_scheme(values: pd.Series) -> tuple[bool, list[str]]:
    """Detect whether a label column holds span-style annotations.

    A span annotation is a JSON string parsing to a list of dicts with
    ``start`` / ``end`` / ``tag`` keys (the same shape that
    ``TextSpanPanel`` writes to the DB).

    The column is treated as a span scheme when:
    - at least one cell starts with ``[`` (so we don't run JSON on free text), AND
    - the majority of non-null cells parse to that shape (allowing empty
      ``[]`` lists, since "no entities" is a valid annotation).

    Returns ``(is_span, sorted_unique_tags)``. ``sorted_unique_tags`` may be
    empty if every row had ``[]`` — the caller can still create the scheme
    and let the user add tags later.
    """
    if values.empty:
        return False, []
    bracket_lead = values.str.lstrip().str.startswith("[")
    if not bracket_lead.any():
        return False, []
    tags: set[str] = set()
    matches = 0
    for raw in values:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, list):
            continue
        # Empty list still counts as a span cell (explicit "no spans")
        if not parsed:
            matches += 1
            continue
        if all(
            isinstance(item, dict) and {"start", "end", "tag"} <= item.keys() for item in parsed
        ):
            matches += 1
            for item in parsed:
                tag = item.get("tag")
                if isinstance(tag, str) and tag:
                    tags.add(tag)
    if matches > len(values) / 2:
        return True, sorted(tags)
    return False, []


class Project:
    """
    Project object
    """

    status: str
    starting_time: float
    name: str
    queue: Queue
    computing: list
    path_models: Path
    data: Data
    db_manager: DatabaseManager
    params: ProjectModel
    schemes: Schemes
    features: Features
    languagemodels: LanguageModels
    nermodels: NerModels | None
    quickmodels: QuickModels
    generations: Generations
    projections: Projections
    messages: Messages
    errors: Errors
    monitoring: Monitoring

    def __init__(
        self,
        project_slug: str,
        queue: Queue,
        db_manager: DatabaseManager,
        path_models: Path,
        users: Users,
        messages: Messages,
    ) -> None:
        """
        Initialize the project
        - load if it exits in database
        """
        self.status = "initialize"
        self.starting_time = time.time()
        self.queue = queue
        self.computing = []
        self.db_manager = db_manager
        self.path_models = path_models
        self.name = project_slug
        self.project_slug = project_slug
        self.errors = Errors()
        self.users = users
        self.messages = messages
        self.monitoring = Monitoring(db_manager, project_slug)

        # cached project directory size (refreshed every _memory_cache_interval seconds)
        self._memory_cache: float = 0.0
        self._memory_cache_time: float = 0.0
        self._memory_cache_interval: float = 60

        # cached project state (avoids recomputing for every polling user)
        self._state_cache: ProjectStateModel | None = None
        self._state_cache_time: float = 0.0
        self._state_cache_interval: float = 2  # seconds

        # load the project if exist
        if self.exists():
            self.status = "created"
            self.load_project(project_slug)

    def exists(self) -> bool:
        """
        Check if the project exists
        """
        if self.db_manager.projects_service.get_project(self.project_slug):
            return True
        return False

    def load_project(self, project_slug: str) -> None:
        """
        Load existing project
        """
        # get projet parameters
        existing_project = self.db_manager.projects_service.get_project(project_slug)

        if not existing_project:
            raise NotFoundError("This project does not exist")

        self.params = ProjectModel(**existing_project["parameters"])

        # check if directory exists
        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # Tabular data management
        self.data = Data(
            self.params.dir,
            self.params.dir.joinpath(config.data_all),
            self.params.dir.joinpath(config.features_file),
            self.params.dir.joinpath(config.train_file),
            self.params.dir.joinpath(config.valid_file),
            self.params.dir.joinpath(config.test_file),
        )

        # create specific management objets
        self.schemes = Schemes(
            project_slug,
            self.db_manager,
            self.data,
        )
        # LanguageModels is built first so Features can take a reference to
        # it (needed to list trained BERTs as a source for bert-embeddings).
        self.languagemodels = LanguageModels(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.db_manager,
            config.resolve_file(config.file_bert_models),
        )
        self.features = Features(
            project_slug,
            self.data,
            self.path_models,
            self.queue,
            cast(list[FeatureComputing], self.computing),
            self.db_manager,
            self.params.language,
            kind=getattr(self.params, "kind", "text"),
            languagemodels=self.languagemodels,
        )
        # Prompt-based selection — available on image projects (multimodal-
        # embeddings features) and text projects (sentence-embeddings features).
        self.prompts: Prompts | None = None
        if self.params.dir is not None:
            self.prompts = Prompts(
                project_slug,
                self.params.dir,
                self.queue,
                self.computing,
                self.features,
            )

            def _cascade_prompts(name: str, kind: str | None) -> None:
                if kind in BINDABLE_FEATURE_KINDS and self.prompts is not None:
                    self.prompts.delete_by_feature(name)

            def _reset_prompts() -> None:
                if self.prompts is not None:
                    self.prompts.reset_all()

            self.features.on_delete = _cascade_prompts
            self.features.on_reset = _reset_prompts
        # Image fine-tuning is only meaningful for image projects.
        self.imagemodels: ImageModels | None = None
        if getattr(self.params, "kind", "text") == "image":
            self.imagemodels = ImageModels(
                project_slug,
                self.params.dir,
                self.queue,
                self.computing,
                self.db_manager,
                config.resolve_file(config.file_image_models),
            )
        # NER fine-tuning is text-only (span schemes don't exist on image
        # projects). Always instantiated for text projects so the state /
        # routes can address it uniformly; whether the UI exposes it is
        # gated by experimental mode on the frontend.
        self.nermodels: NerModels | None = None
        if getattr(self.params, "kind", "text") != "image":
            self.nermodels = NerModels(
                project_slug,
                self.params.dir,
                self.queue,
                self.computing,
                self.db_manager,
                config.resolve_file(config.file_bert_models),
            )
        self.quickmodels = QuickModels(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.db_manager,
            features=self.features,
        )
        self.generations = Generations(
            self.db_manager, cast(list[GenerationComputing], self.computing)
        )
        self.projections = Projections(
            project_slug, self.params.dir, self.computing, self.queue, self.db_manager
        )
        self.lexicometrics = Lexicometrics(self.params.dir, self.computing, self.queue)
        self.bertopic = Bertopic(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.features,
            self.db_manager,
        )

        # Persist any generation checkpoints left behind by a crashed worker or
        # a previous server restart. At load time self.computing is empty, so
        # every gen_*.jsonl file in the project dir is an orphan.
        if self.params.dir is not None:
            for jsonl_path in self.params.dir.glob("gen_*.jsonl"):
                self._recover_generations_from_jsonl(jsonl_path)

    def start_project_creation(self, params: ProjectBaseModel, username: str, path: Path) -> None:
        """
        Manage process creation, sending the heavy process to the queue
        """
        self.status = "creating"

        # test if the name of the column is specified
        if params.col_id is None or params.col_id == "":
            raise InvalidInputError("No column selected for the id")
        if params.cols_text is None or len(params.cols_text) == 0:
            raise InvalidInputError("No column selected for the text")

        # add the dedicated directory
        params.dir = path.joinpath(self.project_slug)

        # start a create project task
        create_project_task.s(CreateProjectTaskInput(
                project_slug=self.project_slug,
                params=params,
                username=username,
                data_all=config.data_all,
                train_file=config.train_file,
                valid_file=config.valid_file,
                test_file=config.test_file,
                features_file=config.features_file,
                random_seed=config.random_seed,)
                # it's mandatory to dump the model to a JSON compatible dict
                .model_dump(mode='json')).apply_async()

    def start_project_creation_imagexp(
        self, params: ProjectBaseModel, username: str, path: Path
    ) -> None:
        """
        Experimental: enqueue creation of an image project.
        See docs/image-projects-strategy.md.
        """
        self.status = "creating"
        params.dir = path.joinpath(self.project_slug)

        unique_id = self.queue.add_task(
            "create_project",
            self.project_slug,
            CreateProjectImagexp(
                self.project_slug,
                params,
                username,
                data_all=config.data_all,
                train_file=config.train_file,
                valid_file=config.valid_file,
                test_file=config.test_file,
                features_file=config.features_file,
                random_seed=config.random_seed,
            ),
            queue="cpu",
        )

        self.computing.append(
            ProjectCreatingModel(
                username=username,
                project_slug=self.project_slug,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="create_project",
                status="training",
            )
        )

    def finish_project_creation(
        self,
        username: str,
        project: ProjectModel,
        import_trainset_labels: pd.DataFrame | None = None,
        import_validset_labels: pd.DataFrame | None = None,
        import_testset_labels: pd.DataFrame | None = None,
    ) -> None:
        """
        Get the result of the queue and finish the creation process
        """
        # add the project to the database
        self.db_manager.projects_service.add_project(
            project.project_slug, jsonable_encoder(project), username
        )

        # add the default scheme if needed
        if import_trainset_labels is None or len(import_trainset_labels.columns) == 0:
            self.db_manager.projects_service.add_scheme(
                self.project_slug, config.default_scheme, [], "multiclass", "system"
            )

        # if labels/schemes to import, add them to the database
        else:
            for col in import_trainset_labels.columns:
                scheme_name = col.replace("dataset_", "")
                non_null = import_trainset_labels[col].dropna().astype(str)
                span_kind, span_labels = _detect_span_scheme(non_null)
                if span_kind:
                    scheme_type = "span"
                    scheme_labels = span_labels
                else:
                    delimiters = non_null.str.contains("|", regex=False).sum()
                    if delimiters < 5:
                        scheme_type = "multiclass"
                        scheme_labels = list(non_null.unique())
                    else:
                        scheme_type = "multilabel"
                        scheme_labels = list(non_null.str.split("|").explode().unique())
                self.db_manager.projects_service.add_scheme(
                    self.project_slug,
                    scheme_name,
                    scheme_labels,
                    scheme_type,
                    "system",
                )
                elements = [
                    {"element_id": element_id, "annotation": label, "comment": ""}
                    for element_id, label in import_trainset_labels[col].dropna().items()
                ]
                self.db_manager.projects_service.add_annotations(
                    dataset="train",
                    user_name=username,
                    project_slug=self.project_slug,
                    scheme=scheme_name,
                    elements=elements,
                )
                if import_validset_labels is not None and col in import_validset_labels.columns:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in import_validset_labels[col].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="valid",
                        user_name=username,
                        project_slug=self.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )
                if import_testset_labels is not None and col in import_testset_labels.columns:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in import_testset_labels[col].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="test",
                        user_name=username,
                        project_slug=self.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )

        # add user authorizations
        self.users.set_auth(
            AuthUserModel(username=username, project_slug=project.project_slug, status="manager")
        )

        # pre-populate the project with the generative models declared in
        # generative.yaml (no-op if the file is missing or empty)
        try:
            Generations(self.db_manager, []).add_default_models(project.project_slug, username)
        except Exception as e:
            print(f"Failed to add default generative models for {project.project_slug}: {e}")

        self.status = "created"

    def delete(self) -> None:
        """
        Delete completely a project
        """

        if self.params.dir is None:
            raise ValueError("No directory for this project")

        # remove from database
        try:
            print("Deleting project from the database")
            self.db_manager.projects_service.delete_project(self.params.project_slug)
        except Exception as e:
            print(f"Problem with the database: {e}")
            raise ValueError("Problem with the database " + str(e))

        # remove folder of the project
        try:
            shutil.rmtree(self.params.dir)
        except Exception as e:
            raise ValueError("No directory to delete " + str(e))

        # remove static files
        if Path(f"{config.data_path}/projects/static/{self.name}").exists():
            shutil.rmtree(f"{config.data_path}/projects/static/{self.name}")

    def drop_evalset(self, dataset: str) -> None:
        """
        Clean all the test data of the project
        - remove the file
        - remove all the annotations in the database
        - set the flag to False
        """
        if not self.params.dir:
            raise Exception("No directory for project")
        path = self.params.dir.joinpath(f"{dataset}.parquet")
        if not path.exists():
            raise NotFoundError("No eval data available")
        os.remove(path)
        if getattr(self.params, "kind", "text") == "image":
            eval_images_dir = self.params.dir.joinpath("images", f"eval_{dataset}")
            if eval_images_dir.exists():
                shutil.rmtree(eval_images_dir, ignore_errors=True)
        self.db_manager.projects_service.delete_annotations_evalset(
            self.params.project_slug, dataset
        )
        if dataset == "test":
            self.data.test = None
            self.params.test = False
            self.params.n_test = 0
        if dataset == "valid":
            self.data.valid = None
            self.params.valid = False
            self.params.n_valid = 0
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )
        # add reload
        self.data.load_dataset("all")
        self._shrink_features_after_evalset_drop(dataset)

    def add_evalset(
        self,
        dataset,
        evalset: EvalSetDataModel | EvalSetImageModel,
        username: str,
        project_slug: str,
    ) -> str | None:
        """
        Add a eval dataset (test or valid)
        """
        if self.params.dir is None:
            raise Exception("Cannot add eval data without a valid dir")
        if dataset not in ["test", "valid"]:
            raise InvalidInputError("Dataset should be test or valid")
        if dataset == "test" and self.params.test:
            raise Exception("There is already a test dataset")
        if dataset == "valid" and self.params.valid:
            raise Exception("There is already a valid dataset")

        kind = getattr(self.params, "kind", "text")
        is_image = kind == "image"
        if is_image and not isinstance(evalset, EvalSetImageModel):
            raise Exception("Image projects require an EvalSetImageModel payload")
        if not is_image and not isinstance(evalset, EvalSetDataModel):
            raise Exception("Text projects require an EvalSetDataModel payload")

        # the evalset/add route filled the filename(s) from the staged
        # uploads it moved into the project data folder
        if not evalset.filename:
            raise Exception("No data file associated with the evalset")

        if isinstance(evalset, EvalSetDataModel):
            if not evalset.cols_text:
                raise Exception("No text column selected for the evalset")
            # each selected label column must match an existing scheme name
            available_schemes = self.schemes.available()
            unknown = [c for c in evalset.cols_label if c not in available_schemes]
            if unknown:
                raise Exception(
                    "Selected label column(s) do not match any existing scheme: "
                    + ", ".join(unknown)
                )
        else:
            if evalset.col_label == "":
                evalset.col_label = None
            if evalset.col_id == "":
                evalset.col_id = None

        # check existing task in the queue → if there is already an add_evalset task for this project and this dataset, we return the status of the task without adding a new one
        if self.queue.current:
            add_eval_task = next(
                (
                    t
                    for t in self.queue.current
                    if t.kind == "add_evalset"
                    and t.project_slug == project_slug
                    and t.task.dataset == dataset  # ty: ignore[unresolved-attribute]
                ),
                None,
            )
            if add_eval_task:
                raise Exception("this set is already being added")

        if isinstance(evalset, EvalSetImageModel):
            scheme_labels = (
                self.schemes.available()[evalset.scheme].labels if evalset.scheme else None
            )
            task = AddEvalSetImage(
                dataset=dataset,
                evalset=evalset,
                project=self.params,
                username=username,
                index=self.data.get_full_id().index,
                project_slug=project_slug,
                scheme=scheme_labels,
            )
        else:
            available_schemes = self.schemes.available()
            schemes_labels = {name: available_schemes[name].labels for name in evalset.cols_label}
            task = AddEvalSet(
                dataset=dataset,
                evalset=evalset,
                project=self.params,
                username=username,
                index=self.data.get_full_id().index,
                project_slug=project_slug,
                schemes=schemes_labels,
            )

        unique_id = self.queue.add_task(
            "add_evalset",
            project_slug,
            task,
            queue="cpu",
        )
        self.computing.append(
            ProcessComputing(
                user=username,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind=f"add_evalset_{dataset}",
            )
        )
        if username == "root":
            return unique_id
        else:
            return None

    def _refresh_features_after_evalset(self, eval_dataset: str, username: str) -> None:
        """
        Bring the features file to the new train+valid+test shape after a
        new eval set has been loaded. If every existing feature can be
        recomputed from its stored parameters, only compute them for the
        new rows in a background ExtendFeatures task — train features and
        quickmodels stay valid. Otherwise (imported/prediction/dataset
        features, image projects, or spec-building failure) fall back to
        a full reset.
        """
        if self._kill_running_feature_extensions():
            self.features.reset_features_file()
            self.quickmodels.drop_models(which="all")
            return

        features_meta = self.db_manager.projects_service.get_project_features(self.name)
        if (
            getattr(self.params, "kind", "text") != "image"
            and features_meta
            and all(f.kind in Features.COMPUTABLE_KINDS for f in features_meta.values())
        ):
            try:
                data = self.features.concat_split_column("text", eval_dataset)
                specs = self.features.build_compute_specs(list(features_meta), data)
                self._enqueue_extend_features(specs, eval_dataset, username)
                return
            except Exception as ex:
                print(f"Cannot extend features for the new {eval_dataset} set: {ex}")
        self.features.reset_features_file()
        self.quickmodels.drop_models(which="all")

    def _shrink_features_after_evalset_drop(self, eval_dataset: str) -> None:
        """
        Remove the dropped eval set's rows from the features file in a
        background task. Row removal needs no recomputation, so every
        feature kind survives (imported and prediction ones included)
        and the quickmodels stay valid.
        """
        if self._kill_running_feature_extensions():
            self.features.reset_features_file()
            self.quickmodels.drop_models(which="all")
            return
        self._enqueue_extend_features([], eval_dataset, "system")

    def _kill_running_feature_extensions(self) -> bool:
        running = [c for c in self.computing if getattr(c, "kind", None) == "extend_features"]
        for c in running:
            self.queue.kill(c.unique_id)
        return bool(running)

    def _enqueue_extend_features(self, specs: list[dict], eval_dataset: str, username: str) -> None:
        eval_data = getattr(self.data, eval_dataset)
        new_index = eval_data.index if eval_data is not None else pd.Index([])
        task = ExtendFeatures(
            specs=specs,
            path_process=self.features.path_all.parent,
            path_models=self.features.path_models,
            language=self.features.lang,
            path_features=self.features.path_features,
            eval_dataset=eval_dataset,
            new_index=new_index,
            full_index=self.data.index.index,
        )
        unique_id = self.queue.add_task("extend_features", self.name, task, queue="cpu")
        self.computing.append(
            ProcessComputing(
                user=username,
                unique_id=unique_id,
                time=datetime.now(timezone.utc),
                kind="extend_features",
            )
        )

    def _recover_features_after_failed_extension(self) -> None:
        """
        A failed or cancelled ExtendFeatures task leaves the features
        file with a shape that no longer matches the datasets (the eval
        set change is already applied) — reset to the safe empty state,
        matching what a non-extendable change does.
        """
        self.features.reset_features_file()
        self.quickmodels.drop_models(which="all")

    def train_quickmodel(
        self,
        quickmodel: QuickModelInModel,
        username: str,
        n_min_annotated: int = 3,
        retrain: bool = False,
    ) -> str:
        """
        Build all the information before calling the quickmodel computation
        retrain : if True, will delete the previous model with the same name
        """
        # Tests
        availabe_schemes = self.schemes.available()
        quickmodel.features = [i for i in quickmodel.features if i is not None]
        if quickmodel.features is None or len(quickmodel.features) == 0:
            raise Exception("No features selected")
        if quickmodel.model not in list(self.quickmodels.available_models.keys()):
            raise Exception("Model not available")
        if quickmodel.scheme not in availabe_schemes:
            raise NotFoundError("Scheme not available")
        if len(availabe_schemes[quickmodel.scheme].labels) < 2:
            raise Exception("Not enough labels in the scheme")
        exist = self.quickmodels.exists(quickmodel.name)
        if exist and not retrain:
            raise AlreadyExistsError("A quickmodel with this name already exists")
        if not exist and retrain:
            raise Exception("No quickmodel with this name to retrain")

        # only dfm feature for multi_naivebayes (FORCE IT if available else error)
        if quickmodel.model == "multi_naivebayes":
            dfm_features = [f for f in self.features.map if f.startswith("dfm")]
            if not dfm_features:
                raise Exception("No dfm features available")
            quickmodel.features = dfm_features
            quickmodel.standardize = False

        if quickmodel.params is None:
            params = None
        else:
            params = dict(quickmodel.params)
        # add information on the target of the model
        if quickmodel.dichotomize is not None and params is not None:
            params["dichotomize"] = quickmodel.dichotomize

        # get data
        df_features = self.features.get(quickmodel.features, dataset=["train"])
        df_scheme = self.schemes.get_scheme(scheme=quickmodel.scheme)

        # management for multilabels / dichotomize
        if quickmodel.dichotomize is not None:
            df_scheme, _ = dichotomize(df_scheme, "labels", quickmodel.dichotomize)

        # test for a minimum of annotated elements
        counts = df_scheme["labels"].value_counts()
        valid_categories = counts[counts >= n_min_annotated]
        if len(valid_categories) < 2:
            raise Exception(
                f"Not enough annotated elements (should be more than {n_min_annotated})"
            )

        col_features = list(df_features.columns)
        data = pd.concat([df_scheme, df_features], axis=1)
        process_id = self.quickmodels.compute_quickmodel(
            project_slug=self.params.project_slug,
            user=username,
            scheme=quickmodel.scheme,
            features=quickmodel.features,
            name=quickmodel.name,
            model_type=quickmodel.model,
            df=data,
            col_labels="labels",
            col_features=col_features,
            model_params=params,
            standardize=quickmodel.standardize or False,
            cv10=quickmodel.cv10 or False,
            balance_classes=quickmodel.balance_classes or False,
            exclude_labels=quickmodel.exclude_labels,
            test_size=quickmodel.test_size,
            retrain=retrain,
            texts=self.data.train["text"] if self.data.train is not None else None,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="train_quickmodel",
            parameters={},
            user_name=username,
        )
        return process_id

    def retrain_quickmodel(self, name: str, scheme: str, username: str) -> None:
        """
        Retrain a quickmodel
        """
        # Get old model parameters in a QuickModelInModel
        model = self.quickmodels.get(name)
        quickmodel = QuickModelInModel(
            name=name,
            scheme=scheme,
            model=model.model_type,
            features=model.features,
            params=model.model_params,
            standardize=model.standardize,
            dichotomize=model.model_params.get("dichotomize", None),
            cv10=model.cv10,
            balance_classes=model.balance_classes,
            exclude_labels=model.exclude_labels,
            test_size=model.test_size,
        )
        self.train_quickmodel(quickmodel, username, retrain=True)

    def get_model_prediction(self, type: str, name: str) -> pd.DataFrame:
        """
        Get prediction of a model or raise an error
        - quickmodel
        - languagemodel
        - imagemodel
        """
        if type == "quickmodel":
            if not self.quickmodels.exists(name):
                raise NotFoundError("Quickmodel doesn't exist")
            else:
                prediction = self.quickmodels.get_prediction(name)
        elif type == "languagemodel":
            if not self.languagemodels.exists(name):
                raise NotFoundError("Languagemodel doesn't exist")
            else:
                prediction = self.languagemodels.get_prediction(name)
        elif type == "imagemodel":
            if self.imagemodels is None or not self.imagemodels.exists(name):
                raise NotFoundError("Image model doesn't exist")
            prediction = self.imagemodels.get_prediction(name)
        else:
            raise InvalidInputError("Model type not recognized")
        return prediction

    def get_prediction_element(self, kind: str, name: str, element_id: str) -> PredictedLabel:
        """
        Get the prediction for a specific element
        - quickmodel
        - languagemodel
        """
        prediction = self.get_model_prediction(kind, name)
        predicted_label = str(prediction.loc[element_id, "prediction"])
        predicted_proba = float(cast(float, prediction.loc[element_id, predicted_label]))
        predicted_entropy = float(cast(float, prediction.loc[element_id, "entropy"]))
        return PredictedLabel(
            label=predicted_label,
            proba=round(predicted_proba, 2) if not math.isnan(predicted_proba) else None,
            entropy=round(predicted_entropy, 2) if not math.isnan(predicted_entropy) else None,
        )

    def get_next(
        self,
        next: NextInModel,
        username: str = "user",
    ) -> list[ElementOutModel]:
        """
        Get next item(s) for a specific scheme with a specific selection method
        - fixed
        - random
        - active (entropy or active LABEL)
        - maxprob
        - test

        next.n : number of elements to return (batch mode)
        history : previous selected elements
        frame is the use of projection coordinates to limit the selection
        filter is a regex to use on the corpus
        """

        if next.scheme not in self.schemes.available():
            raise NotFoundError("Scheme doesn't exist")

        # select the current dataset
        if next.dataset == "test":
            if self.data.test is None:
                raise ValueError("No test dataset available")
            df = self.schemes.get_scheme(next.scheme, complete=True, datasets=["test"])
        elif next.dataset == "valid":
            if self.data.valid is None:
                raise ValueError("No valid dataset available")
            df = self.schemes.get_scheme(next.scheme, complete=True, datasets=["valid"])
        elif next.dataset == "train":
            df = self.schemes.get_scheme(next.scheme, complete=True, datasets=["train"])
        else:
            raise InvalidInputError("Dataset should be test, valid or train")

        # check conditions for active learning and get proba
        proba = None
        if next.model_active is not None:
            prediction = self.get_model_prediction(next.model_active.type, next.model_active.value)
            proba = prediction.reindex(df.index)

        # filter based on the labels
        if next.sample == "untagged":
            f = df["labels"].isna()
        elif next.sample == "not_by_me":
            user_annotated_ids = self.db_manager.projects_service.get_elements_annotated_by_user(
                self.project_slug, next.scheme, [next.dataset], username
            )
            f = ~df.index.isin(user_annotated_ids)
        elif next.sample == "tagged":
            # on specific labels
            if next.on_labels is not None and len(next.on_labels) > 0:
                f = df["labels"].isin(next.on_labels)
            else:
                f = df["labels"].notna()

            # on specific users
            if next.on_users is not None and len(next.on_users) > 0:
                f_user = df["user"].isin(next.on_users)
                f = f & f_user
        elif next.sample == "commented":
            f = df["comment"].fillna("").str.len() > 0
        elif next.sample == "wrong":
            if next.dataset != "train":
                raise InvalidInputError(
                    "Wrong-prediction filter is only available on the train dataset"
                )
            if proba is None or "prediction" not in proba.columns:
                raise InvalidInputError(
                    "Wrong-prediction filter requires an active model with predictions"
                )
            f = df["labels"].notna() & (df["labels"] != proba["prediction"])
        else:
            f = pd.Series(True, index=df.index)

        # filter based on the text (field, context)

        if next.filter:
            # add context in the dataframe (it is ugly but ...)
            if next.dataset == "train":
                existing_cols_contexts = self.params.cols_context
                df = df.join(self.data.train[existing_cols_contexts])
            elif next.dataset == "valid" and self.data.valid is not None:
                existing_cols_contexts = [
                    i for i in self.params.cols_context if i in self.data.valid.columns
                ]
                df = df.join(self.data.valid[existing_cols_contexts])
            elif next.dataset == "test" and self.data.test is not None:
                existing_cols_contexts = [
                    i for i in self.params.cols_context if i in self.data.test.columns
                ]
                df = df.join(self.data.test[existing_cols_contexts])
            else:
                raise InvalidInputError("Dataset should be test, valid or train")

            # sanitize
            df["ID"] = df.index  # duplicate the id column
            filter_san = clean_regex(next.filter)
            if "CONTEXT=" in filter_san:  # case to search in the context
                f_regex = regex_contains(
                    df[existing_cols_contexts + ["ID"]].apply(
                        lambda row: " ".join(row.values.astype(str)), axis=1
                    ),
                    filter_san.replace("CONTEXT=", ""),
                    case=True,
                    na=False,
                )
            elif "QUERY=" in filter_san:  # case to use a query
                query_expr = sanitize_query_expression(
                    filter_san.replace("QUERY=", ""),
                    allowed_columns=existing_cols_contexts,
                )
                f_regex = cast(pd.Series, df[existing_cols_contexts].eval(query_expr))
            else:
                f_regex = regex_contains(df["text"], filter_san, case=True, na=False)
            f = f & f_regex

        # filter with a frame (projection coordinates)
        if next.frame and len(next.frame) == 4:
            if not next.projection_name:
                raise ValueError("No active projection selected for frame selection")
            projection = self.projections.get(next.projection_name)
            if projection is None:
                raise NotFoundError(f"Projection '{next.projection_name}' does not exist")
            projection_data = projection.data
            if projection_data is None:
                raise ValueError("No vizualisation data available")
            f_frame = (
                (projection_data[0] > next.frame[0])
                & (projection_data[0] < next.frame[1])
                & (projection_data[1] > next.frame[2])
                & (projection_data[1] < next.frame[3])
            )
            f = f & f_frame

        # test if there is at least one element available
        if f.sum() == 0:
            raise NotFoundError("No element available with this selection mode.")

        # filter by history
        ss = df[f].drop(next.history, errors="ignore")
        if len(ss) == 0:
            raise ValueError(
                "No element available with this selection mode and history. Clear the history to access previous elements."
            )
        n_sample = f.sum()  # use len(ss) for adding history

        n = min(max(1, next.n), 100)
        element_ids: list = []
        indicators: dict = {}
        similarities: dict = {}
        ranks: dict = {}

        # validate selection method
        valid_selections = {"fixed", "random", "maxprob", "active", "prompt"}
        if next.selection not in valid_selections:
            raise InvalidInputError(f"Unknown selection method: '{next.selection}'")

        # select an element based on the method

        if next.selection == "fixed":  # next rows
            element_ids = list(ss.index[:n])

        elif next.selection == "random":  # random rows
            element_ids = list(ss.sample(n=min(n, len(ss))).index)

        # be sure that the model has been trained
        if next.selection in ["maxprob", "active"] and next.model_active is None:
            raise Exception("An active model is required for this selection method")

        # maxprob: highest probability for a specific label
        if next.selection == "maxprob" and proba is not None:
            if next.label_prob is None:
                raise Exception("Label is required for maxprob selection")
            ss_maxprob = (
                proba[f][next.label_prob]
                .drop(next.history, errors="ignore")
                .sort_values(ascending=False)
            )
            element_ids = list(ss_maxprob.index[:n])
            for eid in element_ids:
                indicators[eid] = f"probability: {round(proba.loc[eid, next.label_prob], 2)}"

        # active: two modes depending on whether label_prob is set
        if next.selection == "active" and proba is not None:
            if next.label_prob is not None:
                # active LABEL: use entropy-LABEL defined as the entropy for the probabilities p(A)/1-p(A)
                entropy_col = f"entropy-{next.label_prob}"
                if entropy_col not in proba.columns:
                    raise ValueError(
                        f"Column '{entropy_col}' not found in model predictions. "
                        "The model may need to be retrained to support this selection method."
                    )
                ss_active = (
                    proba[f][entropy_col]
                    .drop(next.history, errors="ignore")
                    .sort_values(ascending=False)
                )
                element_ids = list(ss_active.index[:n])
                for eid in element_ids:
                    indicators[eid] = f"probability: {round(proba.loc[eid, next.label_prob], 2)}"
            else:
                # active (no label): higher entropy (uncertainty sampling)
                ss_active = (
                    proba[f]["entropy"]
                    .drop(next.history, errors="ignore")
                    .sort_values(ascending=False)
                )
                element_ids = list(ss_active.index[:n])
                for eid in element_ids:
                    indicators[eid] = f"entropy: {round(proba.loc[eid, 'entropy'], 2)}"

        # prompt: cosine similarity between a saved prompt embedding and the
        # element embeddings of its bound feature (multimodal-embeddings on
        # image projects, sentence-embeddings on text projects). The full
        # sorted ranking is cached on the Prompts object per (prompt_id,
        # dataset); each call then just does an index intersection with the
        # candidate set ss.
        if next.selection == "prompt":
            if self.prompts is None:
                raise ValueError("Prompt selection is not available on this project")
            if next.prompt_id is None:
                raise ValueError("prompt_id is required for prompt selection")
            ranked = self.prompts.get_ranking(next.prompt_id, next.dataset)
            # Index.intersection preserves the order of self, so candidates
            # stays in descending-similarity order.
            candidates = ranked.loc[ranked.index.intersection(ss.index)]
            if next.similarity_range is not None:
                lo, hi = next.similarity_range
                if lo > hi:
                    lo, hi = hi, lo
                candidates = candidates[(candidates >= lo) & (candidates <= hi)]
            if candidates.empty:
                raise ValueError("No candidate elements have embeddings for the prompt's feature.")
            element_ids = list(candidates.index[:n])
            for eid in element_ids:
                similarities[eid] = float(candidates.loc[eid])
                # rank in the full prompt ranking (1-based), independent of the
                # currently filtered candidate pool, so the user sees the
                # absolute position across the whole dataset.
                loc = ranked.index.get_loc(eid)
                if not isinstance(loc, int):
                    raise ValueError(
                        f"Expected unique position for {eid} in prompt ranking, got {type(loc).__name__}"
                    )
                ranks[eid] = loc + 1
                indicators[eid] = f"similarity: {round(similarities[eid], 3)}"
        if len(element_ids) == 0:
            raise NotFoundError("No element available with this selection mode.")

        # build the output for each selected element
        elements: list[ElementOutModel] = []
        for element_id in element_ids:
            # get prediction for the element selected
            predict = PredictedLabel(label=None, proba=None, entropy=None)
            if (
                next.model_active is not None
                and next.model_active.type is not None
                and next.model_active.value is not None
                and next.dataset == "train"
            ):
                predict = self.get_prediction_element(
                    next.model_active.type,
                    next.model_active.value,
                    element_id,
                )

            # get all tags already existing for the element selected
            previous = self.schemes.projects_service.get_annotations_by_element(
                self.params.project_slug,
                next.scheme,
                element_id,
            )

            if next.dataset in ["test", "valid"]:
                context = {}
            else:
                if self.data.train is None:
                    raise Exception("Train dataset is not defined")
                # get context for the selected element
                context = dict(
                    self.data.train.loc[element_id, self.params.cols_context]
                    .fillna("NA")
                    .apply(str)
                )

            text = df.loc[element_id, "text"]
            if pd.isna(text):
                text = "NA"

            elements.append(
                ElementOutModel(
                    element_id=element_id,
                    text=text,
                    context=context,
                    selection=next.selection,
                    info=indicators.get(element_id),
                    predict=predict,
                    frame=next.frame,
                    limit=None,
                    history=previous,
                    n_sample=n_sample,
                    similarity=similarities.get(element_id),
                    rank=ranks.get(element_id),
                )
            )
        return elements

    def get_element(
        self,
        element: ElementInModel,
        user: str | None = None,
    ) -> ElementOutModel:
        """
        Get an element of the database
        Separate train/test dataset

        TODO : get next and get element could be merged
        """

        text = None
        predict = PredictedLabel(label=None, proba=None, entropy=None)
        context = {}
        history = None
        if element.scheme is not None:
            history = self.schemes.projects_service.get_annotations_by_element(
                self.params.project_slug, element.scheme, element.element_id
            )

        if element.dataset == "valid":
            if self.data.valid is None:
                raise Exception("Valid dataset is not defined")
            if element.element_id not in self.data.valid.index:
                raise NotFoundError("Element does not exist.")
            text = str(self.data.valid.loc[element.element_id, "text"])

        if element.dataset == "test":
            if self.data.test is None:
                raise Exception("Test dataset is not defined")
            if element.element_id not in self.data.test.index:
                raise NotFoundError("Element does not exist.")
            text = str(self.data.test.loc[element.element_id, "text"])

        # case for train with more information
        if element.dataset == "train":
            if self.data.train is None:
                raise Exception("Train dataset is not defined")
            if element.element_id not in self.data.train.index:
                raise NotFoundError("Element does not exist.")

            text = str(self.data.train.loc[element.element_id, "text"])

            # get prediction if it exists
            predict = PredictedLabel(label=None, proba=None, entropy=None)
            try:
                if element.active_model is not None:
                    predict = self.get_prediction_element(
                        element.active_model.type, element.active_model.value, element.element_id
                    )
            except Exception as e:
                # TODO: warn user to retrain the model
                print(
                    (
                        f"No prediction found for element {element.element_id}."
                        f"Please retrain the model.\n"
                        f"Error: \n{e}"
                    )
                )

            # extract context
            row = self.data.train.loc[element.element_id]
            context = cast(
                dict[str, Any],
                row[self.params.cols_context].fillna("NA").astype(str).to_dict(),
            )
            context = {i.replace("dataset_", ""): str(context[i]) for i in context}

        if text is None:
            raise NotFoundError(
                f"Element {element.element_id} was not found in dataset {element.dataset}"
            )

        return ElementOutModel(
            element_id=element.element_id,
            text=text,
            context=context,
            selection="request",
            predict=predict,
            info="get specific",
            frame=None,
            limit=None,
            history=history,
        )

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params

    @staticmethod
    def compute_annotations_distribution(df: DataFrame, kind: str) -> dict[str, int]:
        if kind == "multiclass":
            return json.loads(df["labels"].value_counts().to_json())
        elif kind == "multilabel":
            return json.loads(df["labels"].str.split("|").explode().value_counts().to_json())
        elif kind == "span":
            r = (
                df["labels"]
                .apply(lambda x: json.loads(x) if pd.notna(x) else [])
                .explode()
                .apply(lambda x: x["tag"] if isinstance(x, dict) and "tag" in x else None)
            )
            return json.loads(r.value_counts().to_json())
        else:
            raise Exception("Not implemented for this kind of scheme")

    def get_statistics(self, scheme: str | None) -> ProjectDescriptionModel:
        """
        Generate a description of a current project/scheme/user
        """
        if scheme is None:
            raise InvalidInputError("Scheme is required")

        schemes = self.schemes.available()
        if scheme not in schemes:
            raise NotFoundError("Scheme not available")
        kind = schemes[scheme].kind

        users = self.db_manager.users_service.get_coding_users(scheme, self.params.project_slug)
        df_annotable = self.schemes.get_scheme(scheme, datasets=["train", "valid", "test"])

        # train
        df_train = df_annotable[df_annotable["dataset"] == "train"]
        train_annotated_distribution = self.compute_annotations_distribution(df_train, kind)
        train_annotated_n = len(df_train.dropna(subset=["labels"]))
        train_set_n = len(self.data.train) if self.data.train is not None else 0

        # valid
        if self.params.valid and (self.data.valid is not None):
            df_valid = df_annotable[df_annotable["dataset"] == "valid"]
            valid_set_n = len(self.data.valid)
            valid_annotated_n = len(df_valid.dropna(subset=["labels"]))
            valid_annotated_distribution = self.compute_annotations_distribution(df_valid, kind)
        else:
            valid_set_n = None
            valid_annotated_n = None
            valid_annotated_distribution = None

        # test
        if self.params.test and (self.data.test is not None):
            df_test = df_annotable[df_annotable["dataset"] == "test"]
            test_set_n = len(self.data.test)
            test_annotated_n = len(df_test.dropna(subset=["labels"]))
            test_annotated_distribution = self.compute_annotations_distribution(df_test, kind)
        else:
            test_set_n = None
            test_annotated_n = None
            test_annotated_distribution = None

        return ProjectDescriptionModel(
            users=users,
            train_set_n=train_set_n,
            train_annotated_n=train_annotated_n,
            train_annotated_distribution=train_annotated_distribution,
            valid_set_n=valid_set_n,
            valid_annotated_n=valid_annotated_n,
            valid_annotated_distribution=valid_annotated_distribution,
            test_set_n=test_set_n,
            test_annotated_n=test_annotated_n,
            test_annotated_distribution=test_annotated_distribution,
            sm_10cv=None,
        )

    def get_projection(
        self,
        projection_name: str,
        scheme: str,
        active_model: ActiveModel | None = None,
    ) -> ProjectionOutModel | None:
        """
        Get a projection by name if computed
        """
        projection = self.projections.get(projection_name)
        if projection is None:
            return None
        # get annotations - use copy to avoid mutating stored projection data
        df = self.schemes.get_scheme(scheme, complete=True, datasets=["train"])
        data = projection.data.copy()
        data["labels"] = df["labels"].reindex(data.index).fillna("NA")

        # get & add predictions if available
        if active_model is not None and active_model.type == "quickmodel":
            if not self.quickmodels.exists(active_model.value):
                raise NotFoundError("Quickmodel doesn't exist")
            data["prediction"] = self.quickmodels.get_prediction(active_model.value)["prediction"]
        elif active_model is not None and active_model.type == "languagemodel":
            if not self.languagemodels.exists(active_model.value):
                raise NotFoundError("Languagemodel doesn't exist")
            data["prediction"] = self.languagemodels.get_prediction(active_model.value)[
                "prediction"
            ]
        elif active_model is not None and active_model.type == "imagemodel":
            if self.imagemodels is None or not self.imagemodels.exists(active_model.value):
                raise NotFoundError("Image model doesn't exist")
            data["prediction"] = self.imagemodels.get_prediction(active_model.value)["prediction"]

        if "prediction" in data:
            predictions = data["prediction"].to_list()
        else:
            predictions = [None] * len(data)

        return ProjectionOutModel(
            nodes=[
                ProjectionOutModelNode(
                    node_id=node_id,
                    x=x,
                    y=y,
                    label=label,
                    predictions=[prediction] if prediction else None,
                )
                for node_id, x, y, label, prediction in zip(
                    data.index.to_list(),
                    data[0].to_list(),
                    data[1].to_list(),
                    data["labels"].to_list(),
                    predictions,
                )
            ],
            status=projection.id,
            parameters=projection.parameters,
            active_model=active_model,
        )

    def _get_cached_memory(self) -> float:
        """
        Return cached project directory size (MB).
        Refreshes at most every _memory_cache_interval seconds.
        """
        now = time.time()
        if (now - self._memory_cache_time) >= self._memory_cache_interval:
            self._memory_cache = get_dir_size(str(self.params.dir))
            self._memory_cache_time = now
        return self._memory_cache

    def state(self) -> ProjectStateModel:
        """
        State of the project
        Collecting states for submodules
        Cached with a short TTL so multiple users polling don't each trigger
        the full computation (DB queries, file reads, pandas work).
        """
        now = time.time()
        if (
            self._state_cache is not None
            and (now - self._state_cache_time) < self._state_cache_interval
        ):
            return self._state_cache

        # expose "prompt" selection only when the project has at least one
        # bindable feature (multimodal-embeddings for image, sentence-embeddings
        # for text) AND at least one saved prompt.
        methods = ["fixed", "random", "maxprob", "active"]
        if self.prompts is not None:
            try:
                available = self.features.get_available()
            except Exception:
                available = {}
            has_bindable_feature = any(f.kind in BINDABLE_FEATURE_KINDS for f in available.values())
            has_prompt = len(self.prompts.list()) > 0
            if has_bindable_feature and has_prompt:
                methods.append("prompt")

        result = ProjectStateModel(
            params=self.params,
            next=NextProjectStateModel(
                methods_min=["fixed", "random"],
                methods=methods,
                sample=["untagged", "all", "tagged", "not_by_me", "commented", "wrong"],
            ),
            schemes=self.schemes.state(),
            features=self.features.state(),
            prompts=self.prompts.state() if self.prompts is not None else None,
            quickmodel=self.quickmodels.state(),
            languagemodels=self.languagemodels.state(),
            imagemodels=self.imagemodels.state() if self.imagemodels is not None else None,
            nermodels=self.nermodels.state() if self.nermodels is not None else None,
            projections=self.projections.state(),
            lexicometrics=self.lexicometrics.state(),
            generations=self.generations.state(),
            bertopic=self.bertopic.state(),
            errors=self.errors.state(),
            memory=self._get_cached_memory(),
            last_activity=self.db_manager.logs_service.get_last_activity_project(
                self.params.project_slug
            ),
            users=self.users.state(self.params.project_slug),
        )
        self._state_cache = result
        self._state_cache_time = now
        return result

    def export_summary(self) -> dict:
        """
        Lab-notebook style snapshot of the project.

        Returns a plain dict (JSON-serializable) covering project parameters,
        users, schemes (with per-label / per-user / per-dataset annotation counts),
        features, language models, quick models, generations, projections, bertopic.
        Dates are ISO-8601 strings.

        This feature can be a bit heavy for the orchestrator ; in the future; move it somewhere else ?
        """

        # Remove keys linked to credentials

        def _iso(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.isoformat()
            return str(value)

        REDACT_KEYS = ("api_key", "apikey", "token", "secret", "password", "credentials")

        def _redact(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: ("***" if any(s in k.lower() for s in REDACT_KEYS) else _redact(v))
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_redact(v) for v in obj]
            return obj

        # --- project header --------------------------------------------------
        slug = self.params.project_slug
        project_row = self.db_manager.projects_service.get_project(slug) or {}
        time_created = project_row.get("time_created")
        time_modified = project_row.get("time_modified")
        created_by = project_row.get("user_name")

        project_block = {
            "slug": slug,
            "name": self.params.project_name,
            "kind": self.params.kind,
            "language": self.params.language,
            "created_at": _iso(time_created),
            "created_by": created_by,
            "modified_at": _iso(time_modified),
            "col_id": self.params.col_id,
            "cols_text": list(self.params.cols_text or []),
            "cols_context": list(self.params.cols_context or []),
            "n_total": self.params.n_total,
            "n_train": self.params.n_train,
            "n_test": self.params.n_test,
            "n_valid": self.params.n_valid,
            "parameters": _redact(jsonable_encoder(self.params, exclude={"dir"})),
        }

        # --- users -----------------------------------------------------------
        auths = self.db_manager.projects_service.get_project_auth(slug) or {}
        users_block = [
            {"username": username, "role": role} for username, role in sorted(auths.items())
        ]

        # --- schemes (with annotation rollups) -------------------------------
        available_schemes = self.schemes.available()
        scheme_datasets = ["train", "test", "valid"]
        schemes_block: list[dict] = []

        # query DB once per scheme via SQLAlchemy for time_created / time_modified
        # available() does not expose timestamps; reuse the same query other code
        # uses to read schemes from the DB.
        scheme_timestamps: dict[str, dict[str, Any]] = {}
        try:
            with self.db_manager.projects_service.Session() as session:
                from activetigger.db.models import Schemes as SchemesTable

                rows = session.query(SchemesTable).filter_by(project_slug=slug).all()
                for r in rows:
                    scheme_timestamps[r.name] = {
                        "time_created": r.time_created,
                        "time_modified": r.time_modified,
                        "user_name": r.user_name,
                    }
        except Exception:
            scheme_timestamps = {}

        for scheme_name, scheme in available_schemes.items():
            per_label: dict[str, int] = {}
            per_user: dict[str, int] = {}
            per_dataset: dict[str, int] = {d: 0 for d in scheme_datasets}
            distinct_elements: set[str] = set()
            total = 0

            for dataset in scheme_datasets:
                try:
                    rows = self.db_manager.projects_service.get_table_annotations_users(
                        slug, scheme_name, dataset
                    )
                except Exception:
                    rows = []
                # rows: [element_id, annotation, user_name, time, dataset]
                for element_id, annotation, user_name, _time, _ds in rows:
                    if annotation is None:
                        continue
                    total += 1
                    per_dataset[dataset] += 1
                    distinct_elements.add(element_id)
                    per_user[user_name] = per_user.get(user_name, 0) + 1
                    # multilabel uses "|"-separated labels
                    if scheme.kind == "multilabel" and "|" in annotation:
                        for lbl in annotation.split("|"):
                            per_label[lbl] = per_label.get(lbl, 0) + 1
                    else:
                        per_label[annotation] = per_label.get(annotation, 0) + 1

            ts = scheme_timestamps.get(scheme_name, {})
            schemes_block.append(
                {
                    "name": scheme_name,
                    "kind": scheme.kind,
                    "labels": list(scheme.labels),
                    "created_at": _iso(ts.get("time_created")),
                    "modified_at": _iso(ts.get("time_modified")),
                    "created_by": ts.get("user_name"),
                    "n_annotations_total": total,
                    "n_annotations_per_dataset": per_dataset,
                    "n_distinct_elements_annotated": len(distinct_elements),
                    "annotations_per_label": dict(sorted(per_label.items())),
                    "annotations_per_user": dict(sorted(per_user.items())),
                }
            )

        # --- features --------------------------------------------------------
        try:
            available_features = self.features.get_available()
        except Exception:
            available_features = {}
        features_block = [
            {
                "name": name,
                "kind": getattr(feat, "kind", None),
                "user": getattr(feat, "user", None),
                "time": _iso(getattr(feat, "time", None)),
                "parameters": _redact(getattr(feat, "parameters", {}) or {}),
            }
            for name, feat in available_features.items()
        ]

        # --- language models -------------------------------------------------
        language_models_block: list[dict] = []
        try:
            lm_available = self.languagemodels.available()
        except Exception:
            lm_available = {}
        for scheme_name, models in lm_available.items():
            for model_name, status in models.items():
                language_models_block.append(
                    {
                        "name": model_name,
                        "scheme": scheme_name,
                        "time": _iso(getattr(status, "time", None)),
                        "predicted": getattr(status, "predicted", False),
                        "predicted_all": getattr(status, "predicted_all", False),
                        "predicted_external": getattr(status, "predicted_external", False),
                        "tested": getattr(status, "tested", False),
                        "exclude_labels": list(getattr(status, "exclude_labels", []) or []),
                    }
                )

        # --- quick models ----------------------------------------------------
        quick_models_block: list[dict] = []
        try:
            qm_available = self.quickmodels.available()
        except Exception:
            qm_available = {}
        for scheme_name, models in qm_available.items():
            for m in models:
                quick_models_block.append(
                    {
                        "name": m.name,
                        "scheme": scheme_name,
                        "kind": m.kind,
                        "time": _iso(m.time),
                        "parameters": _redact(m.parameters or {}),
                    }
                )

        # --- generations / projections / bertopic / image models -------------
        def _state_dict(submodule) -> Any:
            try:
                return jsonable_encoder(submodule.state())
            except Exception:
                return None

        return {
            "format_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project": project_block,
            "users": users_block,
            "schemes": schemes_block,
            "features": features_block,
            "language_models": language_models_block,
            "quick_models": quick_models_block,
            "generations": _state_dict(self.generations),
            "projections": _state_dict(self.projections),
            "bertopic": _state_dict(self.bertopic),
            "imagemodels": _state_dict(self.imagemodels) if self.imagemodels is not None else None,
            "nermodels": _state_dict(self.nermodels) if self.nermodels is not None else None,
            "prompts": _state_dict(self.prompts) if self.prompts is not None else None,
        }

    def export_features(self, features: list, format: str = "parquet") -> FileResponse:
        """
        Export features data in different formats
        """
        if len(features) == 0:
            raise ValueError("No features selected")

        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        data = self.features.get(features, dataset="annotable")

        # expose the original id (id_external) as the index instead of id_internal,
        # for consistency with other exports
        column_name = (self.params.col_id or "id").removeprefix("dataset_")
        data = data.copy()
        data[column_name] = data.index.map(self.data.index["id_external"].to_dict())
        data = data.set_index(column_name)

        file_name = f"extract_schemes_{self.name}.{format}"

        # create files
        if format == "csv":
            data.to_csv(path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(path.joinpath(file_name))

        return FileResponse(path=path.joinpath(file_name), filename=file_name)

    def export_data(
        self, scheme: str, dataset: str = "train", format: str = "parquet", dropna: bool = True
    ) -> FileResponse:
        """
        Export annotation data in different formats
        - for a scheme & dataset
        - for all schemes & every annotation
        """
        # Row-level columns are shared across schemes (one value per row).
        # Annotation-level columns are produced per scheme and prefixed.
        SHARED_COLS = ["dataset", "text", "id_external"]
        SCHEME_COLS = ["labels", "user", "timestamp", "comment"]

        if format not in ["csv", "parquet", "xlsx"]:
            raise InvalidInputError("Format must be csv, parquet or xlsx")

        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        # test dataset availability
        if dataset == "valid":
            if self.data.valid is None:
                raise NotFoundError("No valid data available")
        if dataset == "test":
            if self.data.test is None:
                raise NotFoundError("No test data available")

        # for a specific scheme and dataset
        if scheme != "all" and dataset in ["train", "test", "valid"]:
            df = self.schemes.get_scheme(
                scheme=scheme, complete=True, datasets=[dataset], id_external=True
            )
            data = df[SHARED_COLS + SCHEME_COLS].copy()
            file_name = f"export_tags_{self.name}_{scheme}.{format}"
        # for all the annotated data in the project, need to concate
        elif scheme == "all":
            schemes = self.schemes.available()
            if not schemes:
                raise Exception("No scheme available to export")

            shared = None
            per_scheme = []
            for scheme_name in schemes:
                df = self.schemes.get_scheme(
                    scheme_name,
                    complete=True,
                    datasets=["train", "valid", "test"],
                    id_external=True,
                )
                if shared is None:
                    shared = df[SHARED_COLS].copy()
                per_scheme.append(
                    df[SCHEME_COLS].rename(columns=lambda c, s=scheme_name: f"{s}_{c}")
                )
            data = pd.concat([shared] + per_scheme, axis=1)
            file_name = f"export_tags_{self.name}_all.{format}"
            dropna = False
        else:
            raise InvalidInputError("Scheme or dataset not recognized")

        # transformation of the data
        if dropna:
            data = data.dropna(subset=["labels"])

        # rename the project's id column and put it first
        if self.params.col_id is not None:
            id_col = self.params.col_id.removeprefix("dataset_")
            data = data.rename(columns={"id_external": id_col})
        else:
            id_col = "id_external"
        ordered = [id_col] + [c for c in data.columns if c != id_col]
        data = data[ordered]

        # write file in the folder
        if format == "csv":
            data.to_csv(path.joinpath(file_name), index=False)
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name), index=False)
        if format == "xlsx":
            if "timestamp" in data.columns:
                data["timestamp"] = data["timestamp"].dt.tz_localize(None)
            data.to_excel(path.joinpath(file_name), index=False)

        return FileResponse(path.joinpath(file_name), filename=file_name)

    def _rename_generated_id_column(self, table: DataFrame) -> DataFrame:
        """
        Rename the generated id column and add the external id.

        The "index" column from the generation table holds the slugged
        id_internal. We expose it as id_internal and add a column with the
        original id, named after col_id with the "dataset_" prefix stripped.
        """
        col_name_id = self.params.col_id if self.params.col_id else "id"
        col_name_id = col_name_id.removeprefix("dataset_")
        table[col_name_id] = table["index"].map(self.data.index["id_external"])
        table = table.rename(columns={"index": "id_internal"})
        # put col_id first, then id_internal, then the rest
        ordered = [col_name_id, "id_internal"] + [
            c for c in table.columns if c not in (col_name_id, "id_internal")
        ]
        return table[ordered]

    def get_generated(
        self, project_slug: str, username: str, params: ExportGenerationsParams
    ) -> DataFrame:
        """
        Get generated elements with the original unslugged ids.
        """
        table = self.generations.get_generated(
            project_slug=project_slug,
            user_name=username,
            params=params,
        )
        table_with_id = self._rename_generated_id_column(table)

        return table_with_id

    def export_generations(
        self, project_slug: str, username: str, params: ExportGenerationsParams
    ) -> DataFrame:
        # get the elements
        table = self.generations.get_generated(
            project_slug=project_slug,
            user_name=username,
            params=params,
        )

        # apply filters on the generated
        table["answer"] = self.generations.filter(table["answer"], params.filters)

        # join the text on the internal id before we swap it out
        if self.data.train is None:
            raise Exception("No train data available")
        table = table.join(self.data.train["text"], on="index")

        # expose id_internal as the frame index so it lands as the CSV index
        return self._rename_generated_id_column(table).set_index("id_internal")

    def get_process(
        self, kind: str | list, user: str
    ) -> list[FeatureComputing | LMComputing | QuickModelComputing]:
        """
        Get current processes
        """
        if isinstance(kind, str):
            kind = [kind]
        return [e for e in self.computing if e.user == user and e.kind in kind]

    def export_raw(self, project_slug: str) -> StaticFileModel:
        """
        Export raw data
        To be able to export, need to copy in the static folder
        """
        target_dir = self.params.dir if self.params.dir is not None else Path(".")
        path_origin = target_dir.joinpath("data_all.parquet")
        folder_target = f"{config.data_path}/projects/static/{project_slug}"
        if not Path(folder_target).exists():
            os.makedirs(folder_target)
        files = [i for i in os.listdir(folder_target) if "_data_all_" in i]
        # file already exists
        if len(files) > 0:
            name = files[0]
            path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
        # create the file with a unique id
        else:
            name = f"{project_slug}_data_all_{uuid.uuid4()}.parquet"
            path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
            shutil.copyfile(path_origin, path_target)
        return StaticFileModel(name=name, path=f"{project_slug}/{name}")

    def start_update_project(self, update: ProjectUpdateModel, username: str) -> None:
        """
        Update project parameters

        For text/contexts/expand, it needs to draw from raw data
        - direct small modification
        - bigger modification (texts/contexts/expand) with the queue
        """

        if not self.params.dir:
            raise ValueError("No directory for project")
        if self.data.train is None:
            raise ValueError("No train data for project")

        # update the name
        if update.project_name and update.project_name != self.params.project_name:
            self.params.project_name = update.project_name

        # update the language
        if update.language and update.language != self.params.language:
            self.params.language = update.language

        # for other updates, add task to the queue
        unique_id = self.queue.add_task(
            kind="update_datasets",
            project_slug=self.name,
            task=UpdateDatasets(
                project_params=self.params,
                update=update,
            ),
            queue="cpu",
        )
        self.computing.append(
            UpdateComputing(
                unique_id=unique_id,
                user=username,
                time=datetime.now(timezone.utc),
                kind="update_datasets",
                update=update,
            )
        )

    def start_languagemodel_training(self, bert: BertModelModel, username: str) -> None:
        """
        Launch a training process
        """
        # Check if there is no other competing processes : 1 active process by user
        if len(self.languagemodels.current_user_processes(username)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )
        # get data
        df = self.schemes.get_scheme(bert.scheme, datasets=["train"], complete=True)
        df = df[["text", "labels"]].dropna()

        # Sort multilabel/multiclass
        scheme = self.schemes.available()[bert.scheme]
        scheme_labels = scheme.labels
        training_kind = scheme.kind  # "multiclass" or "multilabel"
        if training_kind not in ["multiclass", "multilabel"]:
            raise Exception(
                f"Training does not support this type of scheme (kind: {training_kind})"
            )

        # management for multilabels / dichotomize
        use_dichotomization = (
            bert.dichotomize is not None and bert.dichotomize != "No dichotomization"
        )

        if use_dichotomization:
            df, scheme_labels = dichotomize(df, "labels", str(bert.dichotomize))
            bert.name = f"{bert.name}_multilabel_on_{bert.dichotomize}"
            # Force training kind and scheme_labels
            training_kind = "multiclass"
            # Sanitize df
            df = df[df["labels"].notna()]

        # remove class under the threshold
        label_counts = get_number_occurrences_per_label(df["labels"], scheme_labels)
        if not use_dichotomization:
            for label_to_exclude in bert.exclude_labels:
                # force label counts to -1 to remove them  at the same time
                label_counts[label_to_exclude] = -1
        df, scheme_labels = remove_labels_without_enough_annotations(
            df, "labels", label_counts, bert.class_min_freq
        )
        df = df[df["labels"].notna()]

        # balance the dataset based on the min class
        if bert.class_balance and training_kind == "multiclass":
            # Specific behaviour for multiclass, balance classes is disabled for multilabels
            min_freq = df["labels"].value_counts().sort_values().min()
            df = df.groupby("labels").sample(n=min_freq)

        # launch training process
        process_id = self.languagemodels.start_training_process(
            name=bert.name,
            project=self.name,
            user=username,
            scheme=bert.scheme,
            df=df,
            training_kind=training_kind,
            scheme_labels=scheme_labels,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=bert.base_model,
            params=bert.params,
            test_size=bert.test_size,
            loss=bert.loss,
            max_length=bert.max_length,
            auto_max_length=bert.auto_max_length,
            class_balance=bert.class_balance,
            class_min_freq=bert.class_min_freq,
            use_dichotomization=use_dichotomization,
            label_for_dichotomization=bert.dichotomize if use_dichotomization else None,
            exclude_labels=bert.exclude_labels,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="train_languagemodel",
            parameters={},
            user_name=username,
        )

    def start_ner_training(self, ner: NerModelModel, username: str) -> None:
        """
        Launch fine-tuning of a token-classification model for a span scheme.

        Span schemes store their annotation as JSON ([{start,end,tag}, ...]);
        the dataframe column is forwarded as-is and parsed inside the task.
        Skips dichotomization / class-balance / exclude-labels, none of
        which apply to BIO tagging.
        """
        if self.nermodels is None:
            raise Exception("NER fine-tuning is not available for this project")
        if (
            len(self.languagemodels.current_user_processes(username))
            + len(self.nermodels.current_user_processes(username))
            > 0
        ):
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )

        df = self.schemes.get_scheme(ner.scheme, datasets=["train"], complete=True)
        df = df[["text", "labels"]].dropna(subset=["text"])
        # keep rows with explicit "[]" (no spans) but drop ones never annotated
        df = df[df["labels"].notna()]

        scheme = self.schemes.available()[ner.scheme]
        if scheme.kind != "span":
            raise Exception(f"NER training requires a span scheme (got kind: {scheme.kind})")

        process_id = self.nermodels.start_training_process(
            name=ner.name,
            project=self.name,
            user=username,
            scheme=ner.scheme,
            df=df,
            scheme_labels=scheme.labels,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=ner.base_model,
            params=ner.params,
            test_size=ner.test_size,
            max_length=ner.max_length,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="train_ner",
            parameters={},
            user_name=username,
        )

    def start_ner_prediction(
        self,
        username: str,
        dataset_type: str,
        datasets: list[str] | None,
        scheme_name: str,
        model_name: str,
        external_dataset: TextDatasetModel | None = None,
        batch_size: int = 16,
    ) -> None:
        """
        Launch a NER prediction run. Mirrors start_language_model_prediction
        but drops training_kind dispatch (always "ner") and routes to
        self.nermodels.
        """
        if self.nermodels is None:
            raise Exception("NER prediction is not available for this project")

        path_train = None
        path_valid = None
        path_test = None
        if dataset_type == "external":
            if external_dataset is None or external_dataset.filename is None:
                raise Exception("No external dataset available for prediction")
            df = None
            col_label = None
            datasets = None
            path_data = self.data.get_path(external_dataset.filename)
        elif dataset_type == "all":
            df = None
            col_label = None
            datasets = None
            path_data = self.data.path_data_all
            path_train = self.data.path_train
            path_valid = self.data.path_valid if self.data.path_valid.exists() else None
            path_test = self.data.path_test if self.data.path_test.exists() else None
        elif dataset_type == "annotable":
            if datasets is None:
                raise Exception("No dataset available for prediction")
            df = self.schemes.get_scheme(
                scheme=scheme_name, complete=True, datasets=datasets, id_external=True
            )
            col_label = "labels"
            path_data = None
        else:
            raise InvalidInputError(f"Dataset {dataset_type} not recognized")

        scheme = self.schemes.available()[scheme_name]
        if scheme.kind != "span":
            raise Exception(f"NER prediction requires a span scheme (got kind: {scheme.kind})")

        process_id = self.nermodels.start_predicting_process(
            project_slug=self.name,
            name=model_name,
            user=username,
            df=df,
            scheme_labels=scheme.labels,
            col_label=col_label,
            dataset=dataset_type,
            batch_size=batch_size,
            statistics=datasets,
            path_data=path_data,
            external_dataset=external_dataset,
            path_train=path_train,
            path_valid=path_valid,
            path_test=path_test,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="predict_ner",
            parameters={},
            user_name=username,
        )

    def start_image_model_training(self, image: ImageModelModel, username: str) -> None:
        """
        Launch an image-classification fine-tuning process for an image project.
        """
        if self.imagemodels is None:
            raise Exception("Image fine-tuning is only available for image projects")
        # Cross-manager check: forbid stacking an image training on top of an
        # existing BERT process (and vice versa) for the same user.
        already = self.languagemodels.current_user_processes(
            username
        ) + self.imagemodels.current_user_processes(username)
        if len(already) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )

        # Labelled rows from the train split. Schemes return a DataFrame whose
        # `text` column carries the image path for image projects.
        df = self.schemes.get_scheme(image.scheme, datasets=["train"], complete=True)
        df = df[["text", "labels"]].dropna()

        scheme = self.schemes.available()[image.scheme]
        scheme_labels = scheme.labels
        training_kind = scheme.kind
        if training_kind not in ["multiclass", "multilabel"]:
            raise Exception(
                f"Training does not support this type of scheme (kind: {training_kind})"
            )

        # Apply class_min_freq and exclude_labels (no dichotomization for images in v1).
        label_counts = get_number_occurrences_per_label(df["labels"], scheme_labels)
        for label_to_exclude in image.exclude_labels:
            label_counts[label_to_exclude] = -1
        df, scheme_labels = remove_labels_without_enough_annotations(
            df, "labels", label_counts, image.class_min_freq
        )
        df = df[df["labels"].notna()]

        if image.class_balance and training_kind == "multiclass":
            min_freq = df["labels"].value_counts().sort_values().min()
            df = df.groupby("labels").sample(n=min_freq)

        process_id = self.imagemodels.start_training_process(
            name=image.name,
            project=self.name,
            user=username,
            scheme=image.scheme,
            df=df,
            training_kind=training_kind,
            scheme_labels=scheme_labels,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=image.base_model,
            params=image.params,
            test_size=image.test_size,
            loss=image.loss,
            class_balance=image.class_balance,
            class_min_freq=image.class_min_freq,
            exclude_labels=image.exclude_labels,
            fp16=image.fp16,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="train_imagemodel",
            parameters={},
            user_name=username,
        )

    def start_image_model_prediction(
        self,
        username: str,
        dataset_type: str,
        datasets: list[str] | None,
        scheme_name: str,
        model_name: str,
        batch_size: int = 32,
    ) -> None:
        """
        Launch an image-classification prediction process.

        Only "annotable" and "all" are supported for v1; external-dataset
        prediction would need an image-upload flow we don't have yet.
        """
        if self.imagemodels is None:
            raise Exception("Image prediction is only available for image projects")
        if dataset_type == "external":
            raise Exception("External-dataset prediction is not supported for image models yet")

        path_train = None
        path_valid = None
        path_test = None
        if dataset_type == "all":
            df = None
            col_label = None
            datasets = None
            path_data = self.data.path_data_all
            path_train = self.data.path_train
            path_valid = self.data.path_valid if self.data.path_valid.exists() else None
            path_test = self.data.path_test if self.data.path_test.exists() else None
        elif dataset_type == "annotable":
            if datasets is None:
                raise Exception("No dataset available for prediction")
            df = self.schemes.get_scheme(
                scheme=scheme_name, complete=True, datasets=datasets, id_external=True
            )
            col_label = "labels"
            path_data = None
        else:
            raise InvalidInputError(f"Dataset {dataset_type} not recognized")

        scheme_ = self.schemes.available()[scheme_name]
        training_kind = scheme_.kind
        if training_kind not in ["multiclass", "multilabel"]:
            raise Exception(
                f"Prediction does not support this type of scheme (kind: {training_kind})"
            )
        scheme_labels = scheme_.labels
        self.imagemodels.start_predicting_process(
            project_slug=self.name,
            name=model_name,
            user=username,
            df=df,
            training_kind=training_kind,
            scheme_labels=scheme_labels,
            col_label=col_label,
            dataset=dataset_type,
            batch_size=batch_size,
            statistics=datasets,
            path_data=path_data,
            path_train=path_train,
            path_valid=path_valid,
            path_test=path_test,
        )

    def start_language_model_prediction(
        self,
        username: str,
        dataset_type: str,
        datasets: list[str] | None,
        scheme_name: str,
        model_name: str,
        external_dataset: TextDatasetModel | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Fetch all necessary data and launch a prediction process
        """
        # Retrieve relevant data
        path_train = None
        path_valid = None
        path_test = None
        if dataset_type == "external":
            if external_dataset is None or external_dataset.filename is None:
                raise Exception("No external dataset available for prediction")
            df = None
            col_label = None
            datasets = None
            path_data = self.data.get_path(external_dataset.filename)
        elif dataset_type == "all":
            df = None
            col_label = None
            datasets = None
            path_data = self.data.path_data_all
            path_train = self.data.path_train
            path_valid = self.data.path_valid if self.data.path_valid.exists() else None
            path_test = self.data.path_test if self.data.path_test.exists() else None
        elif dataset_type == "annotable":
            if datasets is None:
                raise Exception("No dataset available for prediction")
            df = self.schemes.get_scheme(
                scheme=scheme_name, complete=True, datasets=datasets, id_external=True
            )
            col_label = "labels"
            path_data = None
        else:
            raise InvalidInputError(f"Dataset {dataset_type} not recognized")

        scheme_ = self.schemes.available()[scheme_name]
        training_kind = scheme_.kind
        if training_kind not in ["multiclass", "multilabel"]:
            raise Exception(
                f"Prediction does not support this type of scheme (kind: {training_kind})"
            )
        scheme_labels = scheme_.labels
        process_id = self.languagemodels.start_predicting_process(
            project_slug=self.name,
            name=model_name,
            user=username,
            df=df,
            training_kind=training_kind,
            scheme_labels=scheme_labels,
            col_label=col_label,
            dataset=dataset_type,
            batch_size=batch_size,
            statistics=datasets,
            path_data=path_data,
            external_dataset=external_dataset,
            path_train=path_train,
            path_valid=path_valid,
            path_test=path_test,
        )
        self.monitoring.register_process(
            process_name=process_id,
            kind="predict_languagemodel",
            parameters={},
            user_name=username,
        )

    def start_quick_model_prediction(
        self,
        username: str,
        dataset_type: str,
        datasets: list[str] | None,
        scheme_name: str,
        model_name: str,
    ) -> None:
        """
        Fetch all necessary data and launch prediction process
        """
        if datasets is None:
            raise Exception("No dataset available for prediction")
        sm = self.quickmodels.get(model_name)
        if sm is None:
            raise NotFoundError(f"Quick model {model_name} not found")

        # build the X, y dataframe
        df = self.features.get(sm.features, dataset=dataset_type, keep_dataset_column=True)
        cols_features = [col for col in df.columns if col != "dataset"]
        labels = self.schemes.get_scheme(scheme=scheme_name, complete=True, datasets=datasets)
        df["labels"] = labels["labels"]
        df["text"] = labels["text"]

        # add the data for the labels
        self.quickmodels.start_predicting_process(
            name=model_name,
            username=username,
            df=df,
            dataset=dataset_type,
            col_dataset="dataset",
            cols_features=cols_features,
            col_label="labels",
            statistics=datasets,
            col_text="text",
        )

    def start_generation(self, request: GenerationRequest, username: str) -> None:
        """
        Start a generation process
        """
        extract = self.schemes.get_sample(
            request.scheme, request.n_batch, request.mode, dataset=request.dataset
        )
        if len(extract) == 0:
            raise Exception("No elements available for generation")
        model = self.generations.generations_service.get_gen_model(request.model_id)
        # add task to the queue
        unique_id = self.queue.add_task(
            "generation",
            self.name,
            GenerateCall(
                path_process=self.params.dir,
                username=username,
                project_slug=self.name,
                df=extract,
                prompt=request.prompt,
                model=GenerationModel(**model.__dict__),
                cols_context=self.params.cols_context,
                dataset=request.dataset,
                prompt_name=request.prompt_name if request.prompt_name else "",
                n_workers=request.n_workers,
            ),
        )
        self.computing.append(
            GenerationComputing(
                unique_id=unique_id,
                prompt_name=request.prompt_name if request.prompt_name else "",
                user=username,
                project=self.name,
                model_id=request.model_id,
                number=request.n_batch,
                dataset=request.dataset,
                time=datetime.now(timezone.utc),
                kind="generation",
                get_progress=GenerateCall.get_progress_callback(
                    self.params.dir.joinpath(unique_id) if self.params.dir is not None else None
                ),
            )
        )

    def clean_process(self, e: ProcessComputing) -> None:
        """
        Clean a process from computing and queue
        """
        self.computing.remove(e)
        self.queue.delete(e.unique_id)

    def _recover_generations_from_jsonl(self, path: Path) -> None:
        """
        Persist any results left behind in a generation recovery file.

        Rows are inserted into the DB and the file is deleted. Missing file is a no-op.
        """
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.generations.add(
                        user=data["user"],
                        project_slug=data["project_slug"],
                        element_id=data["element_id"],
                        model_id=data["model_id"],
                        prompt=data["prompt"],
                        answer=data["answer"],
                        batch=data.get("batch"),
                    )
            path.unlink()
        except Exception as ex:
            print(f"Failed to recover generation checkpoint {path}: {ex}")

    def update_processes(self) -> None:
        """
        Update completed processes and do specific operations regarding their kind
        - get the result from the queue
        - add the result if needed
        - manage error if needed
        """
        add_predictions = {}

        # loop on the current process
        for e in self.computing.copy():
            # get the process
            process = self.queue.get(e.unique_id)
            if process is None:
                self.clean_process(e)
                continue

            # check if the process is done, else continue
            if process.future is None or not process.future.done():
                continue

            # log error if exists in the process execution
            exception = process.future.exception()
            if exception:
                # User-initiated cancellations are not errors; clean up silently.
                if process.state == "cancelled":
                    print(f"Process {e.kind} cancelled by user")
                    if e.kind == "extend_features":
                        self._recover_features_after_failed_extension()
                    self.clean_process(e)
                    continue
                print(f"Error in {e.kind} : {exception}")
                exception_str = str(exception)
                if any(
                    s in exception_str
                    for s in [
                        "CUDA",
                        "CUDACachingAllocator",
                        "out of memory",
                        "NVML",
                        "cuda",
                    ]
                ):
                    message = (
                        f"Error for process {e.kind} : GPU error — not enough GPU memory available. "
                        "Try reducing the batch size, the max sequence length, or using a smaller model. "
                        f"Details: {exception_str}"
                    )
                else:
                    message = f"Error for process {e.kind} : {exception}"
                self.errors.add(message)

                # recover partial generation results from the checkpoint file
                if e.kind == "generation" and self.params.dir is not None:
                    self._recover_generations_from_jsonl(
                        self.params.dir.joinpath(f"gen_{e.unique_id}.jsonl")
                    )

                # specific case for project creation ; delete the project
                if e.kind == "create_project":
                    print("Error in project creation")
                    self.status = "error"

                # a failed extension leaves a features file with the
                # pre-eval-set shape; reset to the safe empty state
                if e.kind == "extend_features":
                    self._recover_features_after_failed_extension()

                self.clean_process(e)
                continue

            # get the result and do specific operations, if it fails, log the error
            try:
                results = process.future.result()
                match e.kind:
                    case "create_project":
                        e = cast(ProjectCreatingModel, e)
                        if results is None:
                            print("No result from project creation")
                            raise Exception("No result from project creation")
                        self.finish_project_creation(
                            e.username, results[0], results[1], results[2], results[3]
                        )
                    case "update_datasets":
                        e = cast(UpdateComputing, e)
                        self.db_manager.projects_service.update_project(
                            self.params.project_slug, jsonable_encoder(results[0])
                        )
                        self.params = results[0]
                        # reload the data in memory
                        self.data.load_dataset("all")
                        # reset the features file and load the dataset again
                        if results[1]:
                            self.features.reset_features_file()
                            self.bertopic.clear_bertopic()
                            self.projections.clear_projections()

                    case "train_bert":
                        model = cast(LMComputing, e)
                        events = cast(EventsModel, results)
                        self.languagemodels.add(model)
                        self.monitoring.close_process(model.unique_id, events)
                    case "predict_bert":
                        prediction = cast(LMComputing, e)
                        if (
                            results is not None
                            and results.path
                            and "predict_annotable.parquet" in results.path
                        ):
                            add_predictions["predict_" + prediction.model_name] = results.path
                        self.languagemodels.add(prediction)
                        if results is not None and results.events is not None:
                            self.monitoring.close_process(prediction.unique_id, results.events)
                    case "train_image":
                        if self.imagemodels is None:
                            continue
                        model = cast(LMComputing, e)
                        events = cast(EventsModel, results)
                        self.imagemodels.add(model)
                        self.monitoring.close_process(model.unique_id, events)
                    case "predict_image":
                        if self.imagemodels is None:
                            continue
                        prediction = cast(LMComputing, e)
                        if (
                            results is not None
                            and results.path
                            and "predict_annotable.parquet" in results.path
                        ):
                            add_predictions["predict_" + prediction.model_name] = results.path
                        self.imagemodels.add(prediction)
                    case "train_ner":
                        if self.nermodels is None:
                            continue
                        model = cast(LMComputing, e)
                        events = cast(EventsModel, results)
                        self.nermodels.add(model)
                        self.monitoring.close_process(model.unique_id, events)
                    case "predict_ner":
                        if self.nermodels is None:
                            continue
                        prediction = cast(LMComputing, e)
                        if (
                            results is not None
                            and results.path
                            and "predict_annotable.parquet" in results.path
                        ):
                            add_predictions["predict_" + prediction.model_name] = results.path
                        self.nermodels.add(prediction)
                        if results is not None and results.events is not None:
                            self.monitoring.close_process(prediction.unique_id, results.events)
                    case "train_quickmodel":
                        sm = cast(QuickModelComputing, e)

                        # Retrieve the additional events if they exist
                        events = cast(EventsModel, results)
                        self.quickmodels.add(sm)
                        self.monitoring.close_process(sm.unique_id, events)
                    case "predict_quickmodel":
                        sm = cast(QuickModelComputing, e)
                    case "predict_with_features":
                        pf = cast(QuickModelComputing, e)
                        self.quickmodels.mark_prediction_done(pf.name, pf.dataset)
                    case "feature":
                        feature_computation = cast(FeatureComputing, e)
                        self.features.add(
                            feature_computation.name,
                            feature_computation.type,
                            feature_computation.user,
                            feature_computation.parameters,
                            results,
                        )
                    case "prompt":
                        prompt_computation = cast(PromptComputing, e)
                        if self.prompts is not None and results is not None:
                            self.prompts.receive_result(prompt_computation, results)
                    case "prompt_similarity":
                        # the task already saved the similarity parquet;
                        # availability is derived from the file on disk
                        pass
                    case "projection":
                        projection = cast(ProjectionComputing, e)
                        self.projections.add(projection, results)
                    case "lexicometrics":
                        lexicometrics_computation = cast(LexicometricsComputing, e)
                        self.lexicometrics.add(lexicometrics_computation, results)
                    case "generation":
                        e = cast(GenerationComputing, e)
                        r = cast(
                            list[GenerationResult],
                            results,
                        )
                        batch = e.dataset + "_" + str(e.prompt_name) + "_" + e.unique_id
                        for row in r:
                            self.generations.add(
                                user=row.user,
                                project_slug=row.project_slug,
                                element_id=row.element_id,
                                model_id=row.model_id,
                                prompt=row.prompt,
                                answer=row.answer,
                                batch=batch,
                            )
                        if self.params.dir is not None:
                            jsonl = self.params.dir.joinpath(f"gen_{e.unique_id}.jsonl")
                            if jsonl.exists():
                                jsonl.unlink()
                    case "bertopic":
                        bertopic_model = cast(BertopicComputing, e)
                        events = cast(EventsModel, results)
                        self.bertopic.add(bertopic_model)
                        self.monitoring.close_process(bertopic_model.unique_id, events)
                    case "extend_features":
                        self.features.map, self.features.n = self.features.get_map()
                        if self.features.n != len(self.data.index):
                            self._recover_features_after_failed_extension()
                    case kind if kind.startswith("add_evalset_"):
                        e = cast(ProcessComputing, e)
                        if results is not None and len(results) > 0:
                            (
                                eval_dataset,
                                user_name,
                                proj_slug,
                                schemes_elements,
                            ) = results[0]
                            for scheme_name, elements in schemes_elements:
                                if elements:
                                    self.db_manager.projects_service.add_annotations(
                                        eval_dataset,
                                        user_name,
                                        proj_slug,
                                        scheme_name,
                                        elements,
                                    )
                            # update params with the new evalset
                            setattr(self.params, eval_dataset, getattr(results[1], eval_dataset))
                            setattr(
                                self.params,
                                f"n_{eval_dataset}",
                                getattr(results[1], f"n_{eval_dataset}"),
                            )
                            self.db_manager.projects_service.update_project(
                                self.params.project_slug, jsonable_encoder(self.params)
                            )
                            # load the new eval set before refreshing features so the
                            # features file covers the full train+valid+test index
                            self.data.load_dataset(eval_dataset)
                            self._refresh_features_after_evalset(eval_dataset, user_name)
            except Exception as ex:
                print(f"Error in {e.kind} : {ex}")
                self.errors.add(f"Error in {e.kind} : {str(ex)}")
                match e.kind:
                    case "create_project":
                        self.status = "error"
                    case "train_bert":
                        bert_task = cast(LMComputing, e)
                        self.db_manager.language_models_service.delete_model(
                            self.name, bert_task.model_name
                        )
                    case "train_image":
                        image_task = cast(LMComputing, e)
                        self.db_manager.language_models_service.delete_model(
                            self.name, image_task.model_name
                        )
                    case "train_ner":
                        ner_task = cast(LMComputing, e)
                        self.db_manager.language_models_service.delete_model(
                            self.name, ner_task.model_name
                        )
            # clean the process from the list and the queue
            finally:
                self.clean_process(e)  # ty: ignore[invalid-argument-type]

        # if there are predictions, add them
        if len(add_predictions) > 0:
            errors = self.features.add_predictions(add_predictions)
            for err in errors:
                self.errors.add(err)

    # def dump(self, with_files=True) -> None:
    #     """
    #     Dump the project in a archive
    #     - keep the files
    #     - do not keep the models

    #     Ideally, to be able to rerun everything
    #     """
    #     if self.params.dir is None:
    #         raise Exception("No directory for project")
    #     os.mkdir(self.params.dir.joinpath("dump"))

    #     # save the project parameters
    #     # - features computed

    #     # save the annotations

    #     # save the data (train + test + all)
    #     if with_files:
    #         shutil.copyfile(
    #             self.params.dir.joinpath("data_all.parquet"),
    #             self.params.dir.joinpath("dump").joinpath("data_all.parquet"),
    #         )
    #         shutil.copyfile(
    #             self.params.dir.joinpath("train.parquet"),
    #             self.params.dir.joinpath("dump").joinpath("data_train.parquet"),
    #         )
    #         if self.params.test:
    #             shutil.copyfile(
    #                 self.params.dir.joinpath("test.parquet"),
    #                 self.params.dir.joinpath("dump").joinpath("data_test.parquet"),
    #             )

    #     # save the codebook

    #     # create the archive
    #     shutil.make_archive(
    #         f"dump_{self.project_slug}", "zip", self.params.dir.joinpath("dump"), self.params.dir
    #     )

    #     # delete the dump folder
    #     shutil.rmtree(self.params.dir.joinpath("dump"))
    #     return None
