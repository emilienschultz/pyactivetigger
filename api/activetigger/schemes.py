import datetime
from datetime import timezone
from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import cohen_kappa_score

from activetigger.data import Data
from activetigger.datamodels import (
    AnnotationsDataModel,
    CodebookModel,
    CompareSchemesModel,
    ReconciliateElementInModel,
    SchemeModel,
    SchemesProjectStateModel,
    TableAnnotationsModel,
    TableBatchInModel,
    TableOutModel,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.db.projects import ProjectsService
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.functions import clean_regex, regex_contains


class SchemeCache:
    """
    Cache schemes for a project to performe whole dataset operations
    content is a view of scheme for a moment (timestamp, dataframe)
    dataframe : ['dataset', 'labels', 'user', 'timestamp', 'comment', 'text', 'id']
    It has a lifetime then is cleaned
    It is modified when an annotation is added/deleted to keep it up to date
    """

    content: dict[str, Tuple[datetime.datetime, DataFrame]] = {}

    def __init__(self, expiration_delay: int = 5) -> None:
        """
        Init
        """
        self.content = {}
        self.expiration_delay = expiration_delay

    def clean(self) -> None:
        """
        Clean old entries in the cache after a delay (no mater what)
        """
        now = datetime.datetime.now(timezone.utc)
        to_delete = []
        for k, v in self.content.items():
            if (now - v[0]).total_seconds() > self.expiration_delay:
                to_delete.append(k)
        for k in to_delete:
            del self.content[k]

    def get(self, scheme: str) -> DataFrame | None:
        """
        Get a scheme from cache
        """
        self.clean()
        r = self.content.get(scheme, None)
        if r is not None:
            return r[-1]
        return None

    def put(self, scheme: str, df: DataFrame) -> None:
        """
        Put scheme in cache
        """
        self.clean()
        self.content[scheme] = (datetime.datetime.now(timezone.utc), df)

    def update(self, scheme: str, id: str, label: str | None, user: str) -> None:
        """
        Update scheme in the cache with a change if exist
        """
        self.clean()
        if scheme in self.content:
            ts, df = self.content[scheme]
            df.loc[id, "labels"] = label
            df.loc[id, "user"] = user
            now = pd.Timestamp.now("UTC")
            # ensure tz matches the column dtype (may be tz-naive or tz-aware)
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                # column is object-typed (e.g. all NaN); convert to datetime first
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if df["timestamp"].dt.tz is None:
                now = now.tz_localize(None)
            # match the column's resolution (e.g. datetime64[s]) to avoid
            # "Invalid value ... for dtype datetime64[s]" errors in pandas 2.x.
            # numpy datetime64 dtypes don't expose `.unit`, so read it from the array.
            col_reso = getattr(df["timestamp"].array, "unit", None)
            if col_reso:
                now = now.as_unit(col_reso)
            df.loc[id, "timestamp"] = now


class Schemes:
    """
    Manage project schemes & tags

    Tables :
    - schemes
    - annotations
    """

    project_slug: str
    projects_service: ProjectsService
    db_manager: DatabaseManager
    content: DataFrame
    cache: SchemeCache

    def __init__(
        self,
        project_slug: str,
        db_manager: DatabaseManager,
        data: Data,
    ) -> None:
        """
        Init empty
        """
        self.project_slug = project_slug
        self.projects_service = db_manager.projects_service
        self.db_manager = db_manager
        self.data = data

        available = self.available()

        # create a default scheme if not available
        if len(available) == 0:
            self.add_scheme(name="default", labels=[])

        self.cache = SchemeCache()

    def get_scheme_data(
        self, scheme: str, user: str | None = None, force: bool = False
    ) -> DataFrame:
        """
        Get complete current label for a scheme
        Use the cache if possible

        TODO : replace the index structure by an index/corpus structure to replace the dataset information

        """

        df = self.cache.get(scheme)
        # if no cache, get from database
        if df is None or force:
            # annotations from the database
            results = self.projects_service.get_scheme_elements(
                self.project_slug, scheme, ["train", "test", "valid"]
            )
            results_df = pd.DataFrame(
                results,
                columns=["id", "dataset_annotation", "labels", "user", "timestamp", "comment"],
            ).set_index("id")
            # ensure timestamp is datetime
            results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], errors="coerce")
            # join the general index with the scheme data
            df = self.data.index.join(results_df, how="left")
            self.cache.put(scheme, df)
        return df

    def get_scheme(
        self,
        scheme: str,
        user: str | None = None,
        complete: bool = False,
        datasets: list[str] = ["train"],
        id_external: bool = False,
        force: bool = False,
    ) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        complete : add text from dataset & all the row
        """
        if scheme not in self.available():
            raise NotFoundError("Scheme doesn't exist")
        df = self.get_scheme_data(scheme, user, force)

        if id_external:
            cols = ["text", "id_external"]
        else:
            cols = ["text"]

        # add optionnaly the text content
        if complete:
            content = []
            for k in datasets:
                if k == "test":
                    if self.data.test is not None:
                        content.append(self.data.test[cols])
                elif k == "valid":
                    if self.data.valid is not None:
                        content.append(self.data.valid[cols])
                elif k == "train":
                    if self.data.train is not None:
                        content.append(self.data.train[cols])
            if len(content) == 0:
                raise InvalidInputError("Dataset not recognized")
            df_text = pd.concat(content)
            df = df.join(df_text, rsuffix="_content", how="right")
        df["id_internal"] = df.index
        return df[df["dataset"].isin(datasets)]

    def get_reconciliation_table(
        self, scheme: str, dataset: str = "train", no_label: str = "-----"
    ) -> tuple[DataFrame, list[str], dict]:
        """
        Get reconciliation table
        TODO : manage different dataset
        TODO : it is pretty ugly, should be refactored
        """
        if scheme not in self.available():
            raise NotFoundError("Scheme doesn't exist")

        results = self.projects_service.get_table_annotations_users(
            self.project_slug, scheme, dataset
        )
        if len(results) == 0:
            empty = pd.DataFrame(columns=["id", "annotations", "text", "current_label"])
            return empty, [], self._compute_agreement_stats(pd.DataFrame(), [])
        # Shape the data
        df = pd.DataFrame(
            results, columns=["id", "labels", "user", "time", "dataset"]
        )  # shape as a dataframe

        # keep the real labels
        current_labels = df.loc[df.groupby("id")["time"].idxmax()].set_index("id")["labels"]

        def agg(x):
            return list(x)[0] if len(x) > 0 else None  # take the label else None

        df = df.pivot_table(
            index="id", columns="user", values="labels", aggfunc=agg
        )  # pivot and keep the label

        # compute agreement metrics on elements annotated by 2+ users
        users = list(df.columns)
        agreement_stats = self._compute_agreement_stats(df, users)

        f_multi = df.apply(
            lambda x: len(set([i for i in x if pd.notna(i)])) > 1, axis=1
        )  # filter for disagreement
        df = pd.DataFrame(
            df.apply(lambda x: x.fillna(no_label).to_dict(), axis=1), columns=["annotations"]
        )
        # add the text
        if dataset == "train":
            df = df.join(self.data.train[["text"]], how="left")  # add the text
        elif dataset == "test":
            if self.data.test is not None:
                df = df.join(self.data.test[["text"]], how="left")  # add the text
        elif dataset == "valid":
            if self.data.valid is not None:
                df = df.join(self.data.valid[["text"]], how="left")  # add the text
        else:
            raise InvalidInputError("Dataset not recognized")

        df["current_label"] = current_labels
        df = df[f_multi].reset_index()
        # replace NaN with None so pydantic accepts Optional[str]
        df["current_label"] = df["current_label"].astype(object)
        df.loc[df["current_label"].isna(), "current_label"] = None

        # return the result
        return df, users, agreement_stats

    def _compute_agreement_stats(self, pivot_df: DataFrame, users: list[str]) -> dict:
        """
        Compute agreement metrics from a pivot table of user annotations.
        Only considers elements annotated by at least 2 users.
        """
        default = {
            "n_total": 0,
            "n_agreements": 0,
            "n_disagreements": 0,
            "agreement_percentage": None,
            "cohen_kappa": None,
        }
        if len(users) < 2 or len(pivot_df) == 0:
            return default

        # keep only rows where at least 2 users have annotated
        mask = pivot_df.notna().sum(axis=1) >= 2
        df_multi = pivot_df[mask]
        n_total = len(df_multi)

        if n_total == 0:
            return default

        # agreement: all non-null values are the same
        n_agreements = int(
            df_multi.apply(lambda x: len(set([i for i in x if pd.notna(i)])) == 1, axis=1).sum()
        )
        n_disagreements = n_total - n_agreements
        agreement_percentage = n_agreements / n_total

        # compute average pairwise Cohen's Kappa
        kappa_scores = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                pair = df_multi[[users[i], users[j]]].dropna()
                if len(pair) > 0:
                    try:
                        kappa = cohen_kappa_score(pair[users[i]], pair[users[j]])
                        kappa_scores.append(kappa)
                    except Exception:
                        pass

        avg_kappa = sum(kappa_scores) / len(kappa_scores) if kappa_scores else None

        return {
            "n_total": n_total,
            "n_agreements": n_agreements,
            "n_disagreements": n_disagreements,
            "agreement_percentage": agreement_percentage,
            "cohen_kappa": avg_kappa,
        }

    def reconciliate_element(self, element: ReconciliateElementInModel, username: str) -> None:
        """
        Reconciliate an element by adding the selected label for all users
        """
        for u in element.users:
            # unknown users would violate the annotations FK constraint
            self.db_manager.users_service.get_user(u)
        for u in element.users:
            self.push_annotation(
                element.element_id,
                element.label,
                element.scheme,
                u,
                element.dataset,
                "reconciliation",
                "disagreement",
            )

        # add a new tag for the reconciliator
        self.push_annotation(
            element_id=element.element_id,
            label=element.label,
            scheme=element.scheme,
            user=username,
            mode=element.dataset,
            comment="reconciliation",
            selection="disagreement",
        )

    def rename_label(self, former_label: str, new_label: str, scheme: str, username: str) -> None:
        """
        Convert tags from a specific label to another
        """
        available = self.available()
        if scheme not in available:
            raise NotFoundError("Scheme doesn't exist")
        if not self.exists_label(scheme, former_label):
            raise NotFoundError(f"Label {former_label} doesn't exist")
        # test if the new label exist, either create it
        if not self.exists_label(scheme, new_label):
            self.add_label(new_label, scheme, username)

        # convert the tags in the database (add new tag and delete old one)
        self.db_manager.projects_service.rename_label(
            project_slug=self.project_slug,
            scheme=scheme,
            former_label=former_label,
            new_label=new_label,
        )

        # update the scheme (no need to add empty annotation in the database)
        labels = available[scheme].labels
        labels.remove(former_label)
        if new_label not in labels:
            labels.append(new_label)
        self.update_scheme(scheme, labels)

    def get_total(self, dataset: str = "train") -> int:
        """
        Number of element in the dataset
        """
        if dataset == "test":
            # TODO: I think it should be tested way before now
            if self.data.test is None:
                raise Exception("Test dataset is not defined")
            return len(self.data.test)
        return len(self.data.train)

    def get_sample(
        self,
        scheme: str,
        n_elements: int,
        mode: str,
        dataset: str = "train",
        random: bool = False,
    ) -> DataFrame:
        """
        Get a sample of element following a method
        """
        if mode not in ["tagged", "untagged", "all"]:
            raise Exception("Mode not available")
        if scheme not in self.available():
            raise NotFoundError("Scheme doesn't exist")
        df = self.get_scheme(scheme, complete=True, datasets=[dataset])
        # build dataset
        if mode == "tagged":
            df = df[df["labels"].notnull()]

        if mode == "untagged":
            df = df[df["labels"].isnull()]

        if n_elements > len(df):
            n_elements = len(df)

        if random:
            return df.sample(n=n_elements)
        return df.head(n_elements)

    def get_table(
        self,
        batch: TableBatchInModel,
        user: str = "all",
    ) -> TableOutModel:
        """
        Get data table
        scheme : the annotations
        dataset: the dataset
        min, max: the range
        contains: matching regex
        on_users: list of users
        on_labels: list of labels

        Choice to order by index.
        """
        if batch.scheme not in self.available():
            raise NotFoundError(f"Scheme {batch.scheme} is not available")

        # get all data
        df: DataFrame = self.get_scheme(
            batch.scheme,
            complete=True,
            datasets=[batch.dataset],
            id_external=True,
        )

        # manage NaT to avoid problems with json
        df["timestamp"] = df["timestamp"].apply(lambda x: str(x) if pd.notna(x) else "")

        # case of recent annotations (no filter possible)
        if batch.recent:
            list_ids = self.projects_service.get_recent_annotations(
                self.project_slug, user, batch.scheme, batch.max - batch.min, batch.dataset
            )
            df_r = df.loc[list(list_ids)]
            table = df_r
            total = len(df_r)
        else:
            # filters for labels & users
            f_labels = pd.Series([True] * len(df), index=df.index)
            if batch.on_users:
                f_labels = f_labels & df["user"].isin(batch.on_users)
            if batch.on_labels:
                if self.available()[batch.scheme].kind == "multilabel":
                    on_set = set(batch.on_labels)
                    f_labels = f_labels & df["labels"].fillna("").apply(
                        lambda s: bool(on_set.intersection(s.split("|")))
                    )
                else:
                    f_labels = f_labels & df["labels"].isin(batch.on_labels)

            # filter for patterns
            f_contains = pd.Series([True] * len(df), index=df.index)
            if batch.contains:
                if batch.contains.startswith("ALL=") and len(batch.contains) > 4:
                    contains_f = batch.contains.replace("ALL=", "")
                    f_l = regex_contains(df["labels"], clean_regex(contains_f))
                    f_text = regex_contains(df["text"], clean_regex(contains_f))
                    f_contains = f_l | f_text
                elif batch.contains == "HAS_COMMENT":
                    f_contains = df["comment"].fillna("").str.len() > 0
                elif batch.contains.startswith("COMMENT=") and len(batch.contains) > 8:
                    contains_f = batch.contains.replace("COMMENT=", "")
                    f_contains = regex_contains(df["comment"], clean_regex(contains_f))
                else:
                    f_contains = regex_contains(df["text"], clean_regex(batch.contains))
            df = df[f_contains & f_labels]

            # normalize size
            if batch.max == 0:
                batch.max = 20
            if batch.max > len(df):
                batch.max = len(df)
            if batch.min > len(df):
                raise InvalidInputError(
                    f"Minimal value {batch.min} is too high. It should not exced the size of the data ({len(df)})"
                )

            table = df.iloc[int(batch.min) : int(batch.max)]
            total = len(df)

        return TableOutModel(
            items=table.fillna("")[
                ["id_internal", "id_external", "timestamp", "labels", "text", "comment", "user"]
            ].to_dict(orient="records"),
            total=total,
        )

    def add_scheme(
        self,
        name: str,
        labels: list[str],
        kind: str = "multiclass",
        user: str = "server",
    ) -> None:
        """
        Add new scheme
        """
        if self.exists(name):
            raise AlreadyExistsError("Scheme already exists")
        if len(set(labels)) < len(labels):
            raise InvalidInputError(f"Labels need to be unique. Received: {labels}")
        self.projects_service.add_scheme(self.project_slug, name, labels, kind, user)

    def add_label(self, label: str, scheme: str, user: str) -> None:
        """
        Add label in a scheme
        """
        available = self.available()
        if (label is None) or (label == ""):
            raise InvalidInputError("Label cannot be empty")
        if scheme not in available:
            raise NotFoundError("Scheme doesn't exist")
        if available[scheme] is None:
            raise Exception("Scheme is not defined")
        if label in available[scheme].labels:
            return None
        labels = available[scheme].labels
        labels.append(label)
        self.update_scheme(scheme, labels)

    def exists_label(self, scheme: str, label: str) -> bool:
        """
        Test if a label exist in a scheme
        """
        available = self.available()
        if scheme not in available:
            raise NotFoundError("Scheme doesn't exist")
        if label in available[scheme].labels:
            return True
        return False

    def delete_label(self, label: str, scheme: str, user: str) -> None:
        """
        Delete a label in a scheme
        """
        available = self.available()
        if scheme not in available:
            raise NotFoundError("Scheme doesn't exist")
        if label not in available[scheme].labels:
            raise NotFoundError("Label doesn't exist")
        labels = available[scheme].labels
        labels.remove(label)
        # push empty entry for tagged elements using batch insert
        for dataset in ["train", "test", "valid"]:
            df = self.get_scheme(scheme, datasets=[dataset])
            elements = list(df[df["labels"] == label].index)
            if elements:
                self.projects_service.add_annotations(
                    dataset=dataset,
                    user_name=user,
                    project_slug=self.project_slug,
                    scheme=scheme,
                    elements=[
                        {"element_id": e, "annotation": None, "comment": "delete"} for e in elements
                    ],
                    selection="delete",
                )
                # update cache
                for e in elements:
                    self.cache.update(scheme, e, None, user)
        # update scheme
        self.update_scheme(scheme, labels)

    def update_scheme(self, scheme: str, labels: list) -> None:
        """
        Update existing schemes from database
        """
        self.projects_service.update_scheme_labels(self.project_slug, scheme, labels)

    def duplicate_scheme(self, scheme_name: str, new_scheme_name: str, username: str) -> None:
        """
        Duplicate a scheme
        """

        schemes = self.available()

        if scheme_name not in schemes:
            raise NotFoundError("Scheme does not exist")
        if new_scheme_name in schemes:
            raise AlreadyExistsError("New name already exists")

        self.projects_service.duplicate_scheme(
            self.project_slug, scheme_name, new_scheme_name, username
        )

    def rename_scheme(self, old_name: str, new_name: str) -> None:
        """
        Rename a scheme
        """
        schemes = self.available()

        if old_name not in schemes:
            raise NotFoundError("Scheme does not exist")
        if new_name in schemes:
            raise AlreadyExistsError("New name already exists")

        self.projects_service.rename_scheme(self.project_slug, old_name, new_name)

    def delete_scheme(self, name) -> None:
        """
        Delete a scheme
        """
        schemes = self.available()
        if name not in schemes:
            raise NotFoundError("Scheme does not exist")
        if len(schemes) == 1:
            raise Exception("Cannot delete the last scheme")

        self.projects_service.delete_scheme(self.project_slug, name)

    def exists(self, name: str) -> bool:
        """
        Test if scheme exist
        """
        if name in self.available():
            return True
        return False

    def available(self) -> dict[str, SchemeModel]:
        """
        Available schemes {scheme:[labels]}
        """
        r = self.projects_service.available_schemes(self.project_slug)
        return {
            i["name"]: SchemeModel(
                name=i["name"], labels=i["labels"], kind=i["kind"], project_slug=self.project_slug
            )
            for i in r
        }

    def get(self) -> dict:
        """
        state of the schemes
        """
        return {"project_slug": self.project_slug, "availables": self.available()}

    def delete_annotation(
        self, element_id: str, scheme: str, dataset: str, user: str = "server"
    ) -> None:
        """
        Delete a recorded tag
        i.e. : add empty label
        """
        if scheme not in self.available():
            raise NotFoundError("Scheme doesn't exist")

        self.projects_service.add_annotation(
            dataset="delete",
            user_name=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=None,
        )
        self.projects_service.add_annotation(
            dataset=dataset,
            user_name=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=None,
        )

        # update cache
        self.cache.update(scheme, element_id, None, user)

    def push_annotation(
        self,
        element_id: str,
        label: str | None,
        scheme: str,
        user: str = "server",
        mode: str | None = "train",
        comment: str | None = "",
        selection: str | None = None,
    ) -> None:
        """
        Record a tag in the database
        mode : train, predict, test

        + update cache
        """

        if element_id == "noelement":
            raise InvalidInputError("No element id")

        if mode is None:
            mode = "undefined"

        # test if the action is possible
        a = self.available()
        if scheme not in a:
            raise NotFoundError("Scheme doesn't exist")

        # test if the labels used exist in the scheme
        if label is None:
            print("Add null label for ", element_id)

        if a[scheme].kind == "multiclass":
            if label not in a[scheme].labels and label is not None:
                raise InvalidInputError(f"Label {label} not in the scheme")

        elif a[scheme].kind == "multilabel" and label is not None:
            er = [i for i in label.split("|") if i not in a[scheme].labels]
            if len(er) > 0:
                raise InvalidInputError(f"Labels {er} not in the scheme")
        elif a[scheme].kind == "span":
            print("Span annotation, no label check for the moment")

        self.projects_service.add_annotation(
            dataset=mode,
            user_name=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=label,
            comment=comment,
            selection=selection,
        )

        # update cache
        self.cache.update(scheme, element_id, label if label is not None else "", user)

    def push_annotations_table(self, table: TableAnnotationsModel, username: str) -> list | None:
        """
        Push annotations from a table of elements
        """
        errors = []
        for annotation in table.annotations:
            if annotation.label is None or annotation.element_id is None:
                errors.append(annotation)
                continue
            try:
                self.push_annotation(
                    annotation.element_id,
                    annotation.label,
                    annotation.scheme,
                    username,
                    table.dataset,
                    annotation.comment,
                    annotation.selection or "table",
                )
            except Exception:
                errors.append(annotation)
                continue
        return errors if len(errors) > 0 else None

    def get_coding_users(self, scheme: str) -> list[str]:
        """
        Get users action for a scheme
        """
        return self.db_manager.users_service.get_coding_users(scheme, self.project_slug)

    def add_codebook(self, scheme: str, codebook: str, time: str) -> None:
        """
        Add codebook
        if mismatch between date, keep both and return error
        """
        # get lastmodified timestamp for the scheme
        r = self.projects_service.get_scheme_codebook(self.project_slug, scheme)
        # if no modification since the last time, ok
        if r["time"] == time:
            try:
                self.projects_service.update_scheme_codebook(self.project_slug, scheme, codebook)
            except DBException:
                raise
        # if scheme have been modified since the last time
        else:
            new_codebook = f"""
# [CONFLICT] -------- NEW CODEBOOK --------

{codebook}

# [CONFLICT] -------- PREVIOUS CODEBOOK --------

{r["codebook"]}"""
            try:
                self.projects_service.update_scheme_codebook(
                    self.project_slug, scheme, new_codebook
                )
            except DBException:
                raise
            raise AlreadyExistsError("Codebook in conflict, please refresh and arbitrate")

    def get_codebook(self, scheme: str) -> CodebookModel:
        """
        Get codebook
        """
        try:
            r = self.projects_service.get_scheme_codebook(self.project_slug, scheme)
            return CodebookModel(
                scheme=scheme,
                content=str(r["codebook"]),
                time=str(r["time"]),
            )
        except DBException:
            raise

    def add_file_annotations(
        self, annotationsdata: AnnotationsDataModel, file_path: Path, user: str
    ) -> None:
        """
        Add annotations from an uploaded csv file to the trainset
        Create labels if not exist
        """
        # check if the scheme exist
        if annotationsdata.scheme not in self.available():
            raise NotFoundError("Scheme doesn't exist")
        else:
            labels = self.available()[annotationsdata.scheme].labels

        # convert the data, match the external index, drop empty
        df = pd.read_csv(file_path, sep=None, engine="python")

        # test if the index is unique
        df[annotationsdata.col_id] = df[annotationsdata.col_id].astype(str)
        if len(set(df[annotationsdata.col_id])) != len(df):
            raise Exception("Duplicate IDs after slugify in the column selected as ID")

        # keep only id and label, and rename the label to avoid clashes with train columns
        # (e.g. text, dataset, id_external) when the user re-imports an exported file
        label_col = "__imported_label__"
        df = df[[annotationsdata.col_id, annotationsdata.col_label]].rename(
            columns={annotationsdata.col_label: label_col}
        )

        # match to the internal index and keep only annotated
        # (compare external ids as strings: the trainset may store numeric
        # ids as int64 while the imported column was cast to str above)
        df = df.set_index(annotationsdata.col_id)
        col = self.data.train["id_external"].astype(str).map(df[label_col]).dropna().astype(str)

        # if needed, create the labels in the scheme
        for i in col.unique():
            if i not in labels:
                self.add_label(i, annotationsdata.scheme, user)

        # create the elements to add
        elements = [
            {"element_id": element_id, "annotation": label, "comment": ""}
            for element_id, label in col.items()
        ]

        # add the new tags in batch
        self.db_manager.projects_service.add_annotations(
            dataset="train",
            user_name=user,
            project_slug=self.project_slug,
            scheme=annotationsdata.scheme,
            elements=elements,
            selection="import",
        )

        if len(col) < len(df):
            print(
                f"Some elements annoted in the dataset where not added (index mismatch) or not in the trainset. \
                    Number of elements added : {len(col)} (total annotated : {len(df)})"
            )

    def state(self) -> SchemesProjectStateModel:
        """
        Get the state of the project
        """
        return SchemesProjectStateModel(available=self.available())

    def compare(self, schemeA: str, schemeB: str, dataset: str) -> CompareSchemesModel:
        """
        Compare two schemes
        """

        labels = self.available()
        if schemeA not in labels:
            raise NotFoundError("Scheme A doesn't exist")
        if schemeB not in labels:
            raise NotFoundError("Scheme B doesn't exist")

        schemeA_labels = labels[schemeA].labels
        schemeB_labels = labels[schemeB].labels

        # proportion of similar labels (both schemes may have no labels yet)
        n_distinct_labels = len(set(schemeA_labels + schemeB_labels))
        labels_overlapping = 0.0
        if n_distinct_labels > 0:
            labels_overlapping = round(
                100 * len([i for i in schemeA_labels if i in schemeB_labels]) / n_distinct_labels,
                2,
            )

        df_A = self.get_scheme(schemeA, datasets=[dataset])
        df_B = self.get_scheme(schemeB, datasets=[dataset])

        # only keeps elements that have been annotated in both schemes
        df = pd.concat({"schemeA": df_A["labels"], "schemeB": df_B["labels"]}, axis=1).dropna()

        # compute scores
        n_overlapping_annotations = len(df)
        score_ck = None
        percentage: float = 0
        if n_overlapping_annotations > 0:
            score_ck = cohen_kappa_score(df["schemeA"], df["schemeB"])
            percentage = len(df[df["schemeA"] == df["schemeB"]]) / n_overlapping_annotations

        return CompareSchemesModel(
            datetime=datetime.datetime.now(timezone.utc),
            project_slug=self.project_slug,
            schemeA=schemeA,
            schemeB=schemeB,
            labels_overlapping=labels_overlapping,
            n_annotated=n_overlapping_annotations,
            cohen_kappa=score_ck,
            percentage=percentage,
        )
