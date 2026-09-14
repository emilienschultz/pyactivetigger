from pathlib import Path
from typing import cast

import pandas as pd
from pandas import DataFrame

from activetigger.errors import InvalidInputError


class Data:
    """
    Class to manage data

    TODO :
    - move update methods here
    - add methodes for data operations rather than using directly the DataFrame
    """

    path_project: Path
    path_data_all: Path
    path_datasets: Path
    path_train: Path
    path_valid: Path
    path_test: Path
    index: DataFrame
    train: DataFrame
    valid: DataFrame | None
    test: DataFrame | None
    formats_supported = [".csv", ".parquet", ".xlsx"]

    def __init__(
        self,
        path_project: Path,
        path_data_all: Path,
        path_features: Path,
        path_train: Path,
        path_valid: Path,
        path_test: Path,
    ):
        """
        Initialize data object
        """
        self.path_project = path_project
        self.path_datasets = self.path_project.joinpath("data")
        if not self.path_datasets.exists():
            self.path_datasets.mkdir(parents=True, exist_ok=True)
        if not path_train.exists():
            raise FileNotFoundError(f"Training data file not found: {path_train}")

        self.path_train = path_train
        self.path_valid = path_valid
        self.path_test = path_test
        self.path_data_all = path_data_all
        self.path_features = path_features
        self.train = DataFrame()
        self.index = DataFrame()
        self.valid = None
        self.test = None
        self.load_dataset("all")

    @staticmethod
    def _sanitize_dataset(df: DataFrame) -> DataFrame:
        """
        Strips any leading/trailing white spaces in columns
        """
        cols = cast(list[str], df.select_dtypes(include="str"))
        for col in cols:
            df[col] = df[col].str.strip()
        return df

    @staticmethod
    def read_dataset(file_path: Path) -> DataFrame:
        """
        Read a data file and return a DataFrame
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if file_path.suffix.lower() == ".csv":
            return Data._sanitize_dataset(pd.read_csv(file_path, sep=None, engine="python"))
        elif file_path.suffix.lower() == ".parquet":
            return Data._sanitize_dataset(pd.read_parquet(file_path))
        elif file_path.suffix.lower() == ".xlsx":
            return Data._sanitize_dataset(pd.read_excel(file_path))
        else:
            raise InvalidInputError(f"Unsupported file format: {file_path}")

    def get_path(self, filename: str) -> Path:
        """
        Get the full path of a dataset file
        """
        return self.path_datasets / filename

    def load_dataset(self, dataset: str):
        """
        Reload dataset in the memory
        """
        if dataset == "train":
            self.train = pd.read_parquet(self.path_train)
            self.train["dataset"] = "train"
        elif dataset == "valid":
            self.valid = pd.read_parquet(self.path_valid)
            self.valid["dataset"] = "valid"
        elif dataset == "test":
            self.test = pd.read_parquet(self.path_test)
            self.test["dataset"] = "test"
        elif dataset == "all":
            self.train = pd.read_parquet(self.path_train)
            self.train["dataset"] = "train"
            if self.path_valid.exists():
                self.valid = pd.read_parquet(self.path_valid)
                self.valid["dataset"] = "valid"
            if self.path_test.exists():
                self.test = pd.read_parquet(self.path_test)
                self.test["dataset"] = "test"
        else:
            raise InvalidInputError(f"Unknown dataset: {dataset}")

        # temporary fix for index (to remove when the index will be properly set in the raw file)
        if "id_external" not in self.train.columns:
            self.train["id_external"] = self.train.index.astype(str)
        if self.valid is not None and "id_external" not in self.valid.columns:
            self.valid["id_external"] = self.valid.index.astype(str)
        if self.test is not None and "id_external" not in self.test.columns:
            self.test["id_external"] = self.test.index.astype(str)

        # update the general index of annotable
        self.index = self.get_index()

    def get_index(self) -> DataFrame:
        """
        Get the list of available datasets
        """
        corpus = [self.train[["dataset", "id_external"]]]
        if self.valid is not None:
            corpus.append(self.valid[["dataset", "id_external"]])
        if self.test is not None:
            corpus.append(self.test[["dataset", "id_external"]])
        df = pd.concat(corpus)
        return df

    def check_format(self, filename: str) -> bool:
        """
        Check if the file format is supported
        """
        for ext in self.formats_supported:
            if filename.endswith(ext):
                return True
        return False

    def check_dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists
        """
        return (self.path_datasets / dataset_name).exists()

    def get_full_id(self) -> DataFrame:
        """
        Get the full list of IDs from loaded datasets
        """
        return self.index

    def get_external_id(self, element_id: str) -> str:
        """
        Get the external ID for a given internal element ID
        """
        return str(self.index.loc[element_id, "id_external"])
