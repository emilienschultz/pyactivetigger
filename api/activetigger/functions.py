import io
import json
import os
import string
import unicodedata
from getpass import getpass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import bcrypt
import numpy as np
import pandas as pd
import pandas._libs.missing  # noqa: F401
import regex
import spacy
import torch
from cryptography.fernet import Fernet
from fastapi import HTTPException
from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder
from slugify import slugify as python_slugify
from torch import Tensor
from torch.nn import Sigmoid

from activetigger.config import config
from activetigger.datamodels import GpuInformationModel, MLStatisticsModel
from activetigger.errors import InvalidInputError, NotFoundError


def slugify(text: str, way: str = "file") -> str:
    """
    Convert a string to a slug format
    """
    if way == "file":
        return python_slugify(text)
    elif way == "url":
        return quote(text, safe="")
    else:
        raise InvalidInputError("Invalid way parameter. Use 'file' or 'url'.")


def sanitize_uploaded_filename(filename: str) -> str:
    """
    Reduce an uploaded filename to the safe name used on disk.

    The upload endpoints (which write the file) and the tasks that later read
    it back must both go through this function, otherwise a filename with
    accents or special characters is stored under a different name than the
    one the task looks up.
    """

    def to_ascii(text: str) -> str:
        # transliterate accents (é -> e) instead of losing them to underscores
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return regex.sub(r"[^A-Za-z0-9._-]", "_", text)

    base = Path(filename).name
    stem = to_ascii(Path(base).stem).lstrip(".")
    suffix = to_ascii(Path(base).suffix)
    # a fully non-ascii stem (e.g. "中文.csv") transliterates to nothing
    if not stem:
        stem = "file"
    return stem + suffix


def safe_upload_path(
    directory: Path,
    filename: str | None,
    allowed_extensions: tuple[str, ...],
) -> Path:
    """
    Return a path inside `directory` that is safe to write to.

    Strips any directory components from `filename`, restricts the result to a
    conservative character set, and verifies the final path stays inside
    `directory` so a crafted filename cannot escape via traversal or absolute
    path tricks.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    safe_name = sanitize_uploaded_filename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_name.lower().endswith(allowed_extensions):
        allowed_str = ", ".join(ext.lstrip(".") for ext in allowed_extensions)
        raise HTTPException(status_code=400, detail=f"Only {allowed_str} files are allowed")

    directory_resolved = directory.resolve()
    target = (directory_resolved / safe_name).resolve()
    if not target.is_relative_to(directory_resolved):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return target


def remove_punctuation(text) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def replace_accented_chars(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def get_root_pwd() -> str:
    """
    Function to get the password in the CLI start
    """
    print("╔═════════════════════════════════╗")
    print("║    Define a Root Password       ║")
    print("╠═════════════════════════════════╣")
    print("║  Your password must be at least ║")
    print("║  6 characters long and entered  ║")
    print("║  twice to confirm.              ║")
    print("╚═════════════════════════════════╝")
    while True:
        root_password = getpass("Enter a root password : ")

        if len(root_password) < 6:
            print("The password need to have 6 character at minimum")
            continue
        confirm_password = getpass("Re-enter the root password: ")

        if root_password != confirm_password:
            print("Error: The passwords do not match. Please try again.")

        else:
            print("Password confirmed successfully.")
            print("Creating the entry in the database...")
            return root_password


def get_hash(text: str) -> bytes:
    """
    Build a hash string from text
    """
    salt = bcrypt.gensalt()
    hashed: bytes = bcrypt.hashpw(text.encode(), salt)
    return hashed


def compare_to_hash(text: str, hash: str | bytes) -> bool:
    """
    Compare string to its hash
    """

    bytes_hash: bytes
    if type(hash) is str:
        bytes_hash = hash.encode()
    else:
        bytes_hash = cast(bytes, hash)
    return bcrypt.checkpw(text.encode(), bytes_hash)


def tokenize(texts: Series, language: str = "fr", batch_size=100) -> Series:
    """
    Clean texts with tokenization to facilitate word count
    """

    models = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "ja": "ja_core_news_sm",
        "cn": "zh_core_web_sm",
        "es": "es_core_news_sm",
        "nb": "nb_core_news_sm",
    }
    if language not in models:
        raise Exception(f"Language {language} is not supported")
    nlp = spacy.load(models[language], disable=["ner", "tagger"])
    docs = nlp.pipe(texts, batch_size=batch_size)
    textes_tk = [" ".join([str(token) for token in doc]) for doc in docs]
    del nlp
    return pd.Series(textes_tk, index=texts.index)


def get_device() -> torch.device:
    """
    Centralized device selection respecting CPU_ONLY config.
    Priority: CPU_ONLY > CUDA > MPS > CPU
    """

    if config.cpu_only:
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_device_memory() -> None:
    """
    Return cached accelerator memory to the OS after a heavy torch job.
    Freed Python objects stay in the CUDA/MPS allocator cache until this
    is called; no-op on CPU-only setups.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_gpu_memory_info() -> GpuInformationModel:
    """
    Get info on GPU
    """
    if config.cpu_only or not torch.cuda.is_available():
        return GpuInformationModel(
            gpu_available=False,
            total_memory=0.0,
            available_memory=0.0,
        )
    else:
        torch.cuda.empty_cache()
        mem = torch.cuda.mem_get_info()

        return GpuInformationModel(
            gpu_available=True,
            total_memory=round(mem[1] / 1e9, 2),  # Convert to GB
            available_memory=round(mem[0] / 1e9, 2),  # Convert to GB
        )


def cat2num(df):
    """
    Transform a categorical variable to numerics
    """
    df = pd.DataFrame(df)
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df)
    encoded = pd.DataFrame(encoded, index=df.index)
    encoded.columns = ["col" + str(i) for i in encoded.columns]
    return encoded


def clean_regex(text: str) -> str:
    """
    Remove special characters from a string
    """
    if text == "\\" or text == "\\\\":
        text = ""
    if len(text) > 1 and text[-1] == "\\":
        text = text[:-1]
    return text


def sanitize_query_expression(expr: str, allowed_columns: list[str]) -> str:
    """
    Validate a pandas query expression to prevent code injection via df.eval().

    Only allows simple comparison expressions on known columns:
      - column operators: ==, !=, >, <, >=, <=
      - logical connectors: and, or, not
      - string/numeric literals
      - parentheses for grouping

    Raises ValueError if the expression contains disallowed tokens.
    """
    import re

    expr = expr.strip()
    if not expr:
        raise ValueError("Empty query expression")

    # Tokenize: identifiers, quoted strings, numbers, operators, parens
    token_pattern = re.compile(
        r"""
        "[^"]*"          |  # double-quoted string
        '[^']*'          |  # single-quoted string
        [!=<>]=?         |  # comparison operators
        [()]             |  # parentheses
        [A-Za-z_]\w*     |  # identifiers
        -?\d+\.?\d*      |  # numbers
        \S+                 # anything else (will be rejected)
        """,
        re.VERBOSE,
    )
    tokens = token_pattern.findall(expr)

    # Reconstruct to make sure we parsed the whole expression
    if "".join(tokens) != expr.replace(" ", ""):
        raise ValueError("Query expression contains invalid characters")

    allowed_keywords = {"and", "or", "not", "in", "True", "False", "None"}

    for token in tokens:
        # Skip string literals, numbers, operators, parens
        if token.startswith(("'", '"')):
            continue
        if re.fullmatch(r"-?\d+\.?\d*", token):
            continue
        if token in ("==", "!=", ">", "<", ">=", "<=", "(", ")"):
            continue
        if token in allowed_keywords:
            continue
        if token in allowed_columns:
            continue
        raise ValueError(
            f"Disallowed token in query expression: '{token}'. "
            f"Only column names ({', '.join(allowed_columns)}), "
            f"comparisons, and logical operators are allowed."
        )

    return expr


def regex_contains(
    series: pd.Series, pattern: str, case: bool = True, na: bool = False
) -> pd.Series:
    """
    Like pandas str.contains but uses the `regex` module
    to support Unicode property escapes (e.g. \\p{Ll}).
    """
    compiled = regex.compile(pattern, flags=0 if case else regex.IGNORECASE)
    return series.apply(lambda x: bool(compiled.search(str(x))) if pd.notna(x) else na)


def encrypt(text: str | None, secret_key: str | None) -> str:
    """
    Encrypt a string
    """
    if text is None or secret_key is None:
        raise Exception("Text or secret key is None")
    cipher = Fernet(secret_key)
    encrypted_token = cipher.encrypt(text.encode())
    return encrypted_token.decode()


def decrypt(text: str | None, secret_key: str | None) -> str:
    """
    Decrypt a string
    """
    if text is None or secret_key is None:
        raise Exception("Text or secret key is None")
    cipher = Fernet(secret_key)

    decrypted_token = cipher.decrypt(text.encode())
    return decrypted_token.decode()


def logits_to_probs(logits: np.ndarray, kind: str) -> np.ndarray:
    """
    Transform the logit to probabilities using the sigmoid
    """
    if kind == "multilabel":
        sigm = Sigmoid()
        return sigm(Tensor(logits)).numpy()
    elif kind == "multiclass":
        return Tensor(logits).softmax(1).numpy()
    else:
        raise Exception(
            f'logits_to_probs only accepts type = "multilabel" or "multiclass" (received:{kind})'
        )


def activate_probs(
    probs: np.ndarray,
    threshold: float = 0.5,
    strategy: str = "threshold",
    force_max_1_per_row: bool = False,
) -> np.ndarray:
    """
    If strategy = "threshold", use threshold to activate a probability matrix,
    if strategy = "max" use the maximum probability instead
    """
    if strategy == "threshold":
        label_prediction = np.zeros(probs.shape)
        label_prediction[np.where(probs >= threshold)] = 1
    elif strategy == "max":
        label_prediction = np.zeros(probs.shape)
        max_per_row = np.max(probs, axis=1).reshape(-1, 1)
        label_prediction[np.where((probs - max_per_row) >= 0)] = 1
    else:
        raise ValueError(f"Strategy ({strategy})not supported")
    if force_max_1_per_row:
        # sanitize, if equality
        for iRow, row in enumerate(label_prediction):
            if sum(row) > 1:
                argmax = np.argmax(row)
                label_prediction[iRow, :] = 0
                label_prediction[iRow, argmax] = 1
    return label_prediction


# def find_best_threshold(
#     y_true: pd.Series, y_prob_pred: pd.Series
# ) -> tuple[float, np.ndarray, np.ndarray]:
#     """
#     Find the best threshold using Precision-Recall curve and return the probabilities
#     as well as the activated matrix
#     https://www.geeksforgeeks.org/machine-learning/how-to-use-scikit-learns-tunedthresholdclassifiercv-for-threshold-optimization/
#     """
#     if y_true.shape != y_prob_pred.shape:
#         raise ValueError(
#             f"find_best_threshold: Shape missmatch {y_true.shape}!={y_prob_pred.shape}"
#         )

#     thresholds = list(set(y_prob_pred.reshape(-1)))
#     best_threshold, best_f1 = -1.0, -1.0
#     for t in thresholds:
#         y_pred = activate_probs(y_prob_pred.values(), threshold=t, strategy="threshold")
#         f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=1)
#         if f1 > best_f1:
#             best_f1 = float(f1)
#             best_threshold = float(t)
#     return best_threshold


def _clean_score(x: Any, decimals: int = 3) -> float | None:
    """
    Round a metric, converting NaN (from sklearn zero_division=np.nan) to None
    so the value is JSON-safe and clearly signals an undefined score.
    """
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except (TypeError, ValueError):
        return None
    return round(float(x), decimals)


def get_metrics_multiclass(
    Y_true: pd.Series,
    Y_pred: pd.Series,
    id2label: dict[int, str] | None = None,
    texts: pd.Series | None = None,
    decimals: int = 3,
) -> MLStatisticsModel:
    """
    Compute metrics for a prediction
    - precision, f1, recall per label
    - f1 (weighted macro, micro) and precision micro
    - confusion matrix and table
    """
    print("Calculating metrics (multiclass)")
    if id2label is None:
        labels = list(Y_true.unique())
    else:
        labels = list(id2label.values())

    # Compute scores per label --- --- --- --- --- --- --- --- --- --- --- --- -
    precision_label = precision_score(
        Y_true, Y_pred, average=None, labels=labels, zero_division=np.nan
    )
    precision_label = [_clean_score(score, decimals) for score in precision_label]

    f1_label = f1_score(Y_true, Y_pred, average=None, labels=labels, zero_division=np.nan)
    f1_label = [_clean_score(score, decimals) for score in f1_label]

    recall_label = recall_score(Y_true, Y_pred, average=None, labels=labels, zero_division=np.nan)
    recall_label = [_clean_score(score, decimals) for score in recall_label]

    # Compute score averaged (micro, macro, weighted) --- --- --- --- --- --- --
    f1_weighted = _clean_score(
        f1_score(Y_true, Y_pred, average="weighted", labels=labels, zero_division=np.nan),
        decimals,
    )

    f1_macro = _clean_score(
        f1_score(Y_true, Y_pred, average="macro", labels=labels, zero_division=np.nan),
        decimals,
    )

    f1_micro = _clean_score(
        f1_score(Y_true, Y_pred, average="micro", labels=labels, zero_division=np.nan),
        decimals,
    )

    precision_micro = _clean_score(
        precision_score(Y_true, Y_pred, average="micro", labels=labels, zero_division=np.nan),
        decimals,
    )

    accuracy = _clean_score(accuracy_score(Y_true, Y_pred), decimals)

    # Compute confiusion matrix --- --- --- --- --- --- --- --- --- --- --- --- -
    confusion = confusion_matrix(Y_true, Y_pred, labels=labels)

    table = pd.DataFrame(confusion, index=labels, columns=labels)
    table["Total"] = table.sum(axis=1)
    table = table.T
    table["Total"] = table.sum(axis=1)
    table = table.T

    # Create a table of false predictions --- --- --- --- --- --- --- --- --- --
    filter_false_prediction = Y_true != Y_pred
    if texts is not None:
        # Conca
        tab = pd.concat(
            [
                pd.Series(Y_true[filter_false_prediction]),
                pd.Series(Y_pred[filter_false_prediction]),
                pd.Series(texts),
            ],
            axis=1,
            join="inner",
        ).reset_index()
        tab.columns = pd.Index(["id", "GS-label", "prediction", "text"])
        false_prediction = tab.to_dict(orient="records")
    else:
        # TODO: explicit or refactor
        false_prediction = filter_false_prediction.loc[lambda x: x].index.tolist()

    statistics = MLStatisticsModel(
        training_kind="multiclass",
        f1_label=dict(zip(labels, f1_label)),
        precision_label=dict(zip(labels, precision_label)),
        recall_label=dict(zip(labels, recall_label)),
        confusion_matrix=confusion.tolist(),
        f1_weighted=f1_weighted,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        precision=precision_micro,
        accuracy=accuracy,
        false_predictions=false_prediction,
        table=cast(dict[str, Any], table.to_dict(orient="split")),
    )
    return statistics


def get_metrics_multilabel(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    id2label: dict[int, str],
    texts: pd.Series | None = None,
    decimals: int = 3,
) -> MLStatisticsModel:
    """
    Compute metrics for a prediction
    - precision, f1, recall per label
    - f1 (weighted macro, micro) and precision micro
    - confusion matrix and table
    """
    print("Calculating metrics (multilabel)")
    # Compute scores and confusion matrices per label --- --- --- --- --- --- --- --- --- --- --- --- -
    precision_label = {}
    f1_label = {}
    recall_label = {}
    confusion = {}
    dict_of_tables: dict[str, pd.DataFrame] = {}
    for id, label in id2label.items():
        parameters = {
            "y_true": Y_true[:, id],
            "y_pred": Y_pred[:, id],
            "average": "macro",
            "zero_division": np.nan,
        }
        precision_label[label] = _clean_score(precision_score(**parameters), decimals)
        recall_label[label] = _clean_score(recall_score(**parameters), decimals)
        f1_label[label] = _clean_score(f1_score(**parameters), decimals)

        dummy_labels = [label, f"not-{label}"]
        confusion[label] = confusion_matrix(
            y_true=[label if y == 1 else f"not-{label}" for y in Y_true[:, id]],
            y_pred=[label if y == 1 else f"not-{label}" for y in Y_pred[:, id]],
            labels=dummy_labels,
        )
        dict_of_tables[label] = pd.DataFrame(
            confusion[label], index=dummy_labels, columns=dummy_labels
        )
        dict_of_tables[label]["Total"] = dict_of_tables[label].sum(axis=1)
        dict_of_tables[label] = dict_of_tables[label].T
        dict_of_tables[label]["Total"] = dict_of_tables[label].sum(axis=1)
        dict_of_tables[label] = dict_of_tables[label].T

    # Compute score averaged (micro, macro, weighted) --- --- --- --- --- --- --
    f1_weighted = _clean_score(
        f1_score(Y_true, Y_pred, average="weighted", zero_division=np.nan), decimals
    )
    f1_macro = _clean_score(
        f1_score(Y_true, Y_pred, average="macro", zero_division=np.nan), decimals
    )
    f1_micro = _clean_score(
        f1_score(Y_true, Y_pred, average="micro", zero_division=np.nan), decimals
    )
    precision_micro = _clean_score(
        precision_score(Y_true, Y_pred, average="micro", zero_division=np.nan), decimals
    )
    accuracy = _clean_score(accuracy_score(Y_true, Y_pred), decimals)

    # Create a table of false predictions --- --- --- --- --- --- --- --- --- --
    filter_false_prediction = (Y_true != Y_pred).any(axis=1)
    if texts is not None:
        # Need to reconstruct the labels, for now they are matrices [[1,0,0], [1,1,0], ...]
        Y_true_as_series = (
            pd.Series(Y_true.tolist(), index=texts.index)
            .apply(lambda row: matrix_to_label(row, id2label))
            .apply(rejoin_annotation)
        )
        Y_pred_as_series = (
            pd.Series(Y_pred.tolist(), index=texts.index)
            .apply(lambda row: matrix_to_label(row, id2label))
            .apply(rejoin_annotation)
        )
        # Now: ["label1|label2", "label2", ...]
        # Conca
        tab = pd.concat(
            [
                pd.Series(Y_true_as_series[filter_false_prediction]),
                pd.Series(Y_pred_as_series[filter_false_prediction]),
                pd.Series(texts[filter_false_prediction]),
            ],
            axis=1,
            join="inner",
        ).reset_index()
        tab.columns = pd.Index(["id", "GS-label", "prediction", "text"])
        false_prediction = tab.to_dict(orient="records")
    else:
        # TODO: explicit or refactor
        false_prediction = filter_false_prediction.loc[lambda x: x].index.tolist()

    statistics = MLStatisticsModel(
        training_kind="multilabel",
        f1_label=f1_label,
        precision_label=precision_label,
        recall_label=recall_label,
        confusion_matrix=[[]],  # Unused later, table is used instead
        f1_weighted=f1_weighted,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        precision=precision_micro,
        accuracy=accuracy,
        false_predictions=false_prediction,
        table=cast(
            dict[str, Any],
            {key: table.to_dict(orient="split") for key, table in dict_of_tables.items()},
        ),
    )
    return statistics


def get_dir_size(path: str = ".") -> float:
    """
    Get size of a directory in MB. Tolerates entries that disappear mid-scan
    (e.g. a project being deleted concurrently).
    """
    total: float = 0
    stack: list[str] = [str(path)]
    while stack:
        current_path = stack.pop()
        try:
            with os.scandir(current_path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size / (
                                1024**2
                            )  # Convert to MB
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                    except (FileNotFoundError, PermissionError, OSError):
                        continue
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return total


def process_payload_csv(csv_str: str, cols: list[str]) -> pd.DataFrame:
    """
    Process payload from a CSV file in str to get a DataFrame with specific columns
    """
    csv_buffer = io.StringIO(csv_str)
    df = pd.read_csv(
        csv_buffer,
    )
    return df[cols]


def concat_text_columns(df: pd.DataFrame, cols_text: list[str]) -> pd.Series:
    """
    Concatenate several text columns into a single text column,
    joining non-null values with a double newline separator.
    """
    return df[cols_text].apply(lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1)


def get_model_metrics(path_model: Path) -> dict | None:
    """
    Get the scores of the model for a dataset
    - training metrics
    - last computed metrics
    """
    if not path_model.exists():
        raise NotFoundError(f"The folder {path_model} does not exist")

    # training metrics
    if not path_model.joinpath("metrics_training.json").exists():
        raise NotFoundError(f"The file metrics_training.json does not exist in {path_model}")
    with open(
        path_model.joinpath("metrics_training.json"),
        "r",
    ) as f:
        scores = json.load(f)

    # computed metrics and concatenate
    files = sorted(
        [
            f.name
            for f in path_model.iterdir()
            if f.is_file() and f.name.startswith("metrics_predict_")
        ],
    )
    if len(files) > 0:
        last_stat_file = files[-1]
        with open(path_model.joinpath(last_stat_file), "r") as f:
            stats = json.load(f)
        scores = {**scores, **stats}

    return scores


def split_annotation(annotation: str) -> list[str] | pd._libs.missing.NAType:
    """
    Generalise the annoation splitting for multilabel
    annotation : label = multiclass  -> return ["label"]
    annotation : label1|label2 = multilabel -> return ["label1","label2"]

    if annotation is not a string, return NaN
    """
    if isinstance(annotation, str):
        return annotation.split("|")
    else:
        return pd.NA


def rejoin_annotation(list_of_annotations: list[str]) -> str | pd._libs.missing.NAType:
    """
    the opposite of split_annotation
    if list of annotations is not a list of strings, return np.NaN
    """
    if isinstance(list_of_annotations, list):
        if len(list_of_annotations) > 0 and all(
            [isinstance(annotation, str) for annotation in list_of_annotations]
        ):
            return "|".join(list_of_annotations)
    return pd.NA


def matrix_to_label(row: list[int], id2label: dict[int, str]) -> list[str]:
    """
    For a row of labels: [1,0,1] => ["label1", "label3"]
    return the labels associated to the columns with a 1
    """
    if np.isscalar(row):
        return [id2label[int(row)]]  # ty: ignore[invalid-argument-type]

    return [id2label[i] for i, value in enumerate(row) if value == 1]


def get_number_occurrences_per_label(annotations: pd.Series, labels: list[str]) -> dict[str, int]:
    """
    For all labels in annotations ("labelX" if multiclass, "labelX|labelY" if
    multilabel) count the number of occurences.
    """
    n_occurrences = {}
    for label in labels:
        n_occurrences[label] = int(
            annotations.apply(split_annotation)
            .apply(lambda list_annotation: label in list_annotation)
            .sum()
        )
    return n_occurrences


def remove_labels_without_enough_annotations(
    df: pd.DataFrame, col_label: str, label_counts: dict[str, int], class_min_freq: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    For each row, remove annotations containing classes with not enough labels
    and remove rows that do not contain annotations
    """
    annotations = df[col_label].copy()
    scheme_labels = []
    for label in label_counts:
        if label_counts[label] < class_min_freq:
            # Each iteration, split the annotations into list of annotations and
            # rejoin the annotations without a given label
            annotations = annotations.apply(split_annotation).apply(
                lambda LoA: rejoin_annotation([A for A in LoA if A != label])
            )
        else:
            scheme_labels += [label]

    df[col_label] = annotations
    return df, scheme_labels


def dichotomize(
    df: pd.DataFrame, label_col: str, label_for_dichotomization: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    dichotomize labels accordint to the provided label, return the updated label
    scheme as well as the dichotomized dataframe
    """

    annotations = df[label_col].copy()

    def binarize(
        list_of_annotations: list[str] | pd._libs.missing.NAType,
    ) -> bool | pd._libs.missing.NAType:
        if isinstance(list_of_annotations, list):
            return label_for_dichotomization in list_of_annotations
        else:
            return pd.NA

    annotations = (
        annotations.apply(split_annotation)
        .apply(binarize)
        .replace({True: label_for_dichotomization, False: f"not-{label_for_dichotomization}"})
    )
    df[label_col] = annotations
    new_scheme_labels = [label_for_dichotomization, f"not-{label_for_dichotomization}"]
    return df, new_scheme_labels


def annotations_to_matrix(annotations: pd.Series, labels: list[str]) -> np.ndarray:
    """
    Convert a series of pipe-separated annotations to a binary matrix.
    """
    rows = []
    for annotation in annotations:
        parts = annotation.split("|") if isinstance(annotation, str) else []
        rows.append([int(label in parts) for label in labels])
    return np.array(rows)
