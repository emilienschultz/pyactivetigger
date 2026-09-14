from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.responses import FileResponse
from pandas import DataFrame

from activetigger.datamodels import (
    ProjectionComputing,
    ProjectionDataModel,
    ProjectionParametersModel,
    ProjectionsProjectStateModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, NotFoundError
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_projection import ComputeProjection


class Projections:
    """
    Manage projections.

    Projections are project-level and keyed by a user-provided name (like
    BERTopic models), so a project can hold several named visualisations
    computed with different parameters and users can switch between them.

    The database (models table, kind="projection") is the source of truth:
    a row is inserted only once a computation has completed. The
    coordinates DataFrame is stored as a parquet file referenced by the
    row's path.
    """

    project_slug: str
    path: Path
    models_service: ModelsService
    loaded: dict[str, ProjectionDataModel]
    options: dict[str, dict[str, Any]]
    computing: list
    queue: Queue

    def __init__(
        self,
        project_slug: str,
        path: Path,
        computing: list,
        queue: Queue,
        db_manager: DatabaseManager,
    ) -> None:
        self.project_slug = project_slug
        self.path = path.joinpath("projections")
        self.path.mkdir(parents=True, exist_ok=True)
        self.computing = computing
        self.queue = queue
        self.models_service = db_manager.language_models_service
        self.loaded = {}
        self.options = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": ["cosine", "euclidean"],
            },
            "tsne": {
                "n_components": 2,
                "learning_rate": "auto",
                "init": "random",
                "perplexity": 3,
            },
        }

    def current_computing(self):
        return [e.name for e in self.computing if e.kind == "projection"]

    def training(self) -> dict[str, str]:
        """
        Currently under training, keyed by projection name.
        """
        return {e.name: e.method for e in self.computing if e.kind == "projection"}

    def exists(self, name: str) -> bool:
        """
        Test if a projection is registered in the database.
        """
        model = self.models_service.get_model(self.project_slug, name)
        return model is not None and model.kind == "projection"

    def add(self, element: ProjectionComputing, results: DataFrame) -> None:
        """
        Register a computed projection: save the coordinates to a parquet
        file, then add the row in the database (file first, so a row
        never points to a missing file).
        """
        name = element.params.name or element.name
        # the file is keyed by the computation id to avoid deriving file
        # names from free-text projection names
        file_path = self.path.joinpath(f"{element.unique_id}.parquet")
        data = results.copy()
        # parquet requires string column names; axis columns are integers
        data.columns = data.columns.astype(str)
        data.to_parquet(file_path)
        self.models_service.add_model(
            kind="projection",
            project=self.project_slug,
            name=name,
            user=element.user,
            status="trained",
            scheme=None,
            params={"unique_id": element.unique_id, **element.params.model_dump()},
            path=str(file_path),
        )
        self.loaded.pop(name, None)

    def delete(self, name: str) -> None:
        """
        Delete a projection by name (database row first, then artifact).
        """
        model = self.models_service.get_model(self.project_slug, name)
        if model is None or model.kind != "projection":
            raise NotFoundError(f"Projection '{name}' does not exist")
        self.models_service.delete_model(self.project_slug, name)
        Path(model.path).unlink(missing_ok=True)
        self.loaded.pop(name, None)

    def compute(
        self,
        project_slug: str,
        username: str,
        projection: ProjectionParametersModel,
        features: DataFrame,
        normalize_features: bool = False,
    ) -> None:
        """
        Launch the projection computation in the queue.
        """
        if not projection.name:
            raise Exception("A projection name is required")
        if self.models_service.model_exists(self.project_slug, projection.name):
            raise Exception(f"A model named '{projection.name}' already exists")
        if projection.name in [e.name for e in self.computing if e.kind == "projection"]:
            raise AlreadyExistsError(
                f"A projection named '{projection.name}' is already being computed"
            )

        unique_id = self.queue.add_task(
            "projection",
            project_slug,
            ComputeProjection(
                kind=projection.method,
                features=features,
                params=projection.parameters,
                normalize_features=normalize_features,
            ),
        )
        self.computing.append(
            ProjectionComputing(
                unique_id=unique_id,
                name=projection.name,
                user=username,
                time=datetime.now(timezone.utc),
                kind="projection",
                method=projection.method,
                params=projection,
                normalize_features=normalize_features,
            )
        )

    def get(self, name: str) -> ProjectionDataModel | None:
        """
        Get a projection by name from the database
        (coordinates are read from the parquet file and cached in memory).
        """
        if name in self.loaded:
            return self.loaded[name]
        model = self.models_service.get_model(self.project_slug, name)
        if model is None or model.kind != "projection":
            return None
        file_path = Path(model.path)
        if not file_path.exists():
            return None
        data = pd.read_parquet(file_path)
        # restore the integer axis column names used by the consumers
        data.columns = [int(c) if isinstance(c, str) and c.isdigit() else c for c in data.columns]
        params = dict(model.parameters or {})
        unique_id = str(params.pop("unique_id", ""))
        element = ProjectionDataModel(
            id=unique_id,
            name=name,
            data=data,
            parameters=ProjectionParametersModel(**params),
        )
        self.loaded[name] = element
        return element

    def state(self) -> ProjectionsProjectStateModel:
        available: dict[str, str | int] = {
            m.name: str((m.parameters or {}).get("unique_id", ""))
            for m in self.models_service.available_models(self.project_slug, "projection")
        }
        return ProjectionsProjectStateModel(
            options=self.options,
            available=available,
            training=self.training(),
        )

    def export(
        self,
        name: str,
        format: str = "csv",
        col_id: str | None = None,
        id_mapping: DataFrame | None = None,
    ) -> FileResponse:
        """
        Export a named projection.

        The projection is internally indexed on id_internal (slugified). When
        an id_mapping is supplied (project.data.index), expose the original
        id values under col_id, for consistency with other exports.
        """
        projection = self.get(name)
        if projection is None:
            raise NotFoundError("No projection available")
        data = projection.data.copy()

        if id_mapping is not None and "id_external" in id_mapping.columns:
            column_name = (col_id or "id").removeprefix("dataset_")
            data[column_name] = data.index.map(id_mapping["id_external"].to_dict())
            data = data.set_index(column_name)

        data.columns = data.columns.astype(str)
        file_name = f"projection_{name}.{format}"
        if format == "csv":
            data.to_csv(self.path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(self.path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(self.path.joinpath(file_name))

        return FileResponse(
            path=self.path.joinpath(file_name),
            filename=file_name,
        )

    def clear_projections(self):
        """
        Clear the projections
        """
        for m in self.models_service.available_models(self.project_slug, "projection"):
            self.delete(m.name)
