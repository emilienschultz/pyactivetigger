import json
from datetime import datetime, timezone
from pathlib import Path

from activetigger.datamodels import (
    LexicometricsComputing,
    LexicometricsModel,
    LexicometricsParametersModel,
    LexicometricsProjectStateModel,
    LexicometricsStatisticsModel,
)
from activetigger.errors import AlreadyExistsError
from activetigger.queue_manager import Queue
from activetigger.tasks.compute_lexicometrics import ComputeLexicometrics


class Lexicometrics:
    """
    Manage lexicometry statistics of the annotable dataset.

    A single result per project, computed on demand and persisted as a JSON
    file so new statistics can be added later without a schema migration.
    """

    path: Path
    computing: list
    queue: Queue
    available: LexicometricsModel | None

    def __init__(self, path: Path, computing: list, queue: Queue) -> None:
        self.path = path
        self.computing = computing
        self.queue = queue
        self.available = None
        self.load()

    @property
    def file_path(self) -> Path:
        return self.path.joinpath("lexicometrics.json")

    @property
    def _legacy_file_path(self) -> Path:
        # results computed before the textometrics -> lexicometrics rename
        return self.path.joinpath("textometrics.json")

    def load(self) -> None:
        if not self.file_path.exists() and self._legacy_file_path.exists():
            try:
                self._legacy_file_path.rename(self.file_path)
            except Exception as e:
                print("Error migrating legacy textometrics file", e)
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r") as f:
                self.available = LexicometricsModel(**json.load(f))
        except Exception as e:
            print("Error in loading lexicometrics", e)

    def _save(self) -> None:
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.available.model_dump() if self.available else None, f)
        except Exception as e:
            print("Error in saving lexicometrics", e)

    def training(self) -> dict[str, str]:
        """
        Currently computing, keyed by username.
        """
        return {e.user: "computing" for e in self.computing if e.kind == "lexicometrics"}

    def compute(self, project_slug: str, username: str, language: str) -> None:
        """
        Launch the lexicometrics computation in the queue.

        The task reads the train dataset from the project directory itself.
        """
        if len(self.training()) > 0:
            raise AlreadyExistsError("Lexicometrics are already being computed")

        parameters = LexicometricsParametersModel(language=language)
        unique_id = self.queue.add_task(
            "lexicometrics",
            project_slug,
            ComputeLexicometrics(
                path_data=self.path,
                language=parameters.language,
                tokenizer_name=parameters.tokenizer,
                n_most_frequent=parameters.n_most_frequent,
                tfidf_n_words=parameters.tfidf_n_words,
                tfidf_n_docs_per_word=parameters.tfidf_n_docs_per_word,
                tfidf_n_words_per_doc=parameters.tfidf_n_words_per_doc,
                tfidf_min_term_freq=parameters.tfidf_min_term_freq,
                tfidf_max_documents=parameters.tfidf_max_documents,
            ),
        )
        self.computing.append(
            LexicometricsComputing(
                unique_id=unique_id,
                user=username,
                time=datetime.now(timezone.utc),
                kind="lexicometrics",
                parameters=parameters,
            )
        )

    def add(self, element: LexicometricsComputing, results: dict) -> None:
        """
        Store statistics after computation.
        """
        self.available = LexicometricsModel(
            computed_at=datetime.now(timezone.utc).isoformat(),
            user=element.user,
            parameters=element.parameters,
            statistics=LexicometricsStatisticsModel(**results),
        )
        self._save()

    def get(self) -> LexicometricsModel | None:
        return self.available

    def state(self) -> LexicometricsProjectStateModel:
        return LexicometricsProjectStateModel(
            available=self.available is not None,
            training=self.training(),
        )
