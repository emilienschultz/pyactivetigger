import gc
import json
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import shutil
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import datasets
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from activetigger.config import config
from activetigger.datamodels import EventsModel, LMParametersModel, MLStatisticsModel
from activetigger.functions import (
    activate_probs,
    get_device,
    get_metrics_multiclass,
    get_metrics_multilabel,
    logits_to_probs,
    matrix_to_label,
    release_device_memory,
    split_annotation,
)
from activetigger.monitoring import TaskTimer
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.predict_bert import annotations_to_matrix
from activetigger.tasks.utils import length_after_tokenizing, retrieve_model_max_length

pd.set_option("future.no_silent_downcasting", True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLoggingCallback(TrainerCallback):
    event: Optional[multiprocessing.synchronize.Event]
    current_path: Path
    logger: Logger

    def __init__(self, event, logger, current_path):
        super().__init__()
        self.event = event
        self.current_path = current_path
        self.logger = logger
        # Set from trainer.model_accepts_loss_kwargs after the Trainer is built
        # (see __load_trainer). When the model's forward accepts **kwargs, the
        # Trainer assumes the loss is normalized via num_items_in_batch and
        # skips its own division by gradient_accumulation_steps — but our loss
        # paths (model-internal encoder heads, CustomTrainer) all ignore
        # num_items_in_batch, so the logged train loss comes out inflated by
        # gradacc and must be corrected here. When the forward does NOT accept
        # **kwargs (transformers <= 4.x encoder models), the Trainer already
        # normalizes and dividing again would deflate the train curve (#1109).
        self.needs_gradacc_correction = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.logger.info(f"Step {state.global_step}")
        progress_percentage = (state.global_step / state.max_steps) * 100
        with open(self.current_path.joinpath("progress_train"), "w") as f:
            f.write(str(progress_percentage))
        gradacc = args.gradient_accumulation_steps if self.needs_gradacc_correction else 1
        adjusted_history = []
        for entry in state.log_history:
            if "loss" in entry and "eval_loss" not in entry and gradacc > 1:
                entry = dict(entry)
                entry["loss"] = entry["loss"] / gradacc
            adjusted_history.append(entry)
        with open(self.current_path.joinpath("log_history.txt"), "w") as f:
            json.dump(adjusted_history, f)
        # end if event set
        if self.event is not None:
            if self.event.is_set():
                self.logger.info("Event set, stopping training.")
                control.should_training_stop = True
                raise Exception("Process interrupted by user")


# Function for the weighted loss computation


# Rescaling the weights
def compute_class_weights(dataset, label_key="labels"):
    # Labels are stored as one-hot vectors; convert to class indices
    labels = [example[label_key].argmax().item() for example in dataset]
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    # Inverse frequency weight, ordered by label index
    weights = [total / (num_classes * label_counts[k]) for k in sorted(label_counts.keys())]
    return torch.tensor(weights, dtype=torch.float)


# CustomTrainer is a subclass of Trainer that allows for custom loss computation.
# https://stackoverflow.com/questions/70979844/using-weights-with-transformers-huggingface
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weights = kwargs.pop("class_weights", None)
        self.training_kind = kwargs.pop("training_kind", "multiclass")
        super().__init__(*args, **kwargs)
        self._loss_fct = None  # avoid device mismatch
        print("CustomTrainer initialized with class weights:", self.class_weights)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self._loss_fct is None:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None  # fmt:skip
            if self.training_kind == "multiclass":  # AO:Left multiclass by default
                self._loss_fct = nn.CrossEntropyLoss(weight=weights)
            elif self.training_kind == "multilabel":
                self._loss_fct = nn.BCEWithLogitsLoss(weight=weights)
            else:
                raise ValueError(f"Training kind {self.training_kind} not recognized.")
        if self.training_kind == "multiclass":
            label_indices = labels.argmax(dim=-1)
            loss = self._loss_fct(
                logits.view(-1, self.model.config.num_labels),  # ty: ignore[unresolved-attribute]
                label_indices.view(-1),
            )
        else:
            loss = self._loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


class TrainBert(BaseTask):
    """
    Class to train a bert model

    Parameters:
    ----------
    path (Path): path to save the files
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    base_model (str): model to use
    params (dict) : training parameters
    test_size (dict): train/test distribution
    event : possibility to interrupt
    unique_id : unique id for the current task
    loss : loss function to use (cross_entropy, weighted_cross_entropy)

    TODO : test more weighted loss entropy
    """

    kind = "train_bert"

    def __init__(
        self,
        path: Path,
        project_slug: str,
        model_name: str,
        df: DataFrame | datasets.Dataset,
        training_kind: str,
        scheme_labels: list[str],
        use_dichotomization: bool,
        col_text: str,
        col_label: str,
        base_model: str,
        params: LMParametersModel,
        test_size: float,
        label_for_dichotomization: str | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        loss: Optional[str] = "cross_entropy",
        max_length: int = 512,
        auto_max_length: bool = False,
        class_balance: bool = False,
        class_min_freq: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.project_slug = project_slug
        self.name = model_name
        df.index.name = "id"  # ty: ignore[unresolved-attribute]
        self.df = df
        if training_kind not in ["multiclass", "multilabel"]:
            raise ValueError(
                (
                    f"TrainBERT only works for multiclass and "
                    f"multilabel but you set training_kind = {training_kind}"
                )
            )
        self.training_kind = training_kind
        if len(scheme_labels) != len(set(scheme_labels)):
            raise ValueError(
                (f"Labels in your scheme are not unique.\nLabels provided : {scheme_labels}")
            )
        if use_dichotomization:
            raise ValueError("Dichotomization not supported in multilabel.")
        self.use_dichotomization = use_dichotomization
        self.label_for_dichotomization = label_for_dichotomization
        self.scheme_labels = scheme_labels
        self.col_text = col_text
        self.col_label = col_label
        self.base_model = base_model
        self.params = params
        self.test_size = test_size
        self.event = event
        self.unique_id = unique_id
        if loss == "weighted_cross_entropy" and training_kind == "multilabel":
            raise ValueError(
                "weighted_cross_entropy loss is not supported for multilabel classification."
            )
        self.loss = loss
        self.max_length = max_length
        self.auto_max_length = auto_max_length
        self.class_balance = class_balance
        self.class_min_freq = class_min_freq

    def __init_paths(self) -> Tuple[Path, Path]:
        """Initiate the current path (directory for the model) and for the logger"""
        #  create repertory for the specific model
        current_path = self.path.joinpath(self.name)
        if not current_path.exists():
            os.makedirs(current_path)
        # logging the process
        log_path = current_path.joinpath("status.log")
        return current_path, log_path

    def __init_logger(self, log_path) -> Logger:
        """Load the logger and set it up"""
        logger = logging.getLogger("train_bert_model")
        # without an explicit level the logger inherits WARNING and every
        # logger.info() is dropped — status.log stays empty
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # the logger is a per-process singleton and loky reuses workers:
        # drop handlers from previous trainings or logs leak across models
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start {self.base_model}")
        return logger

    def __check_data(self, df: pd.DataFrame, col_label: str, col_text: str) -> pd.DataFrame:
        """Remove rows missing labels or text"""
        df = df.copy()
        # test labels missing values and remove them
        if df[col_label].isnull().sum() > 0:
            df = df[df[col_label].notnull()]
            self.logger.info(f"Missing labels - reducing training data to {len(df)}")
            print(f"Missing labels - reducing training data to {len(df)}")

        # test empty texts and remove them
        if df[col_text].isnull().sum() > 0:
            df = df[df[col_text].notnull()]
            self.logger.info(f"Missing texts - reducing training data to {len(df)}")
            print(f"Missing texts - reducing training data to {len(df)}")

        # Test that all labels in the label column appear in the scheme labels
        scheme_set = set(self.scheme_labels)

        def _check_labels(annotation: object) -> bool:
            if not isinstance(annotation, str):
                return False
            parts = split_annotation(annotation)
            if not isinstance(parts, list):
                return False
            return all(part in scheme_set for part in parts)

        condition = df[col_label].apply(_check_labels)
        if (~condition).sum() > 0:
            df = df[condition]
            self.logger.info(f"Labels unrecognised - reducing training data to {len(df)}")
            print(f"Labels not recognised - reducing training data to {len(df)}")

        return df

    def __retrieve_labels(self, scheme_labels):
        if len(scheme_labels) < 2:
            raise ValueError(
                "Not enough classes. Either you excluded classes or "
                "there are not enough annotations."
            )

        label2id = {j: i for i, j in enumerate(scheme_labels)}
        id2label = {i: j for i, j in enumerate(scheme_labels)}
        return scheme_labels, label2id, id2label

    def __transform_to_dataset(
        self,
        training_kind: str,
        df: pd.DataFrame,
        col_label: str,
        col_text: str,
        label2id: dict[str, int],
    ) -> datasets.Dataset:
        """Transform the dataframe into a dataset with the right format for
        training"""
        ids = df.reset_index()["id"].copy().to_list()
        texts = df[col_text].copy().to_list()
        one_hot = "weight" in self.loss.lower() if self.loss is not None else False
        print(f"One hot encoding: {one_hot}", flush=True)
        if training_kind == "multiclass":
            print("Preprocess multiclass")
            labels_as_list = df[col_label].copy().replace(label2id).tolist()  # maybe map useful
            if one_hot:
                labels = torch.tensor(
                    [[int(i == j) for j in range(len(label2id))] for i in labels_as_list],
                    dtype=torch.float32,
                )
            else:
                labels = torch.tensor(labels_as_list, dtype=torch.long)
        elif training_kind == "multilabel":
            print("Preprocess multilabel")
            labels = torch.tensor(
                annotations_to_matrix(df[col_label], list(label2id.keys())).tolist(),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Training kind {training_kind} not recognized.")

        return datasets.Dataset.from_dict(
            {
                "id": ids,
                "text": texts,
                "labels": labels,
            }
        ).with_format("torch")

    def __load_tokenizer(self, base_model: str):
        """Load the tokenize"""
        return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def __cap_tokenizer_max_length(
        self,
        texts: pd.Series,
        tokenizer,
        auto_max_length: bool,
        original_max_length: int,
        base_model_max_length: int,
        adapt: bool,
    ) -> Tuple[Any, int, int]:
        """Cap the tokenizer max length and create a tokenizing function"""

        # if auto_max_length set max_length to the maximum length of tokenized sentences
        # Tokenize the text column
        def get_n_tokens(txt):
            return length_after_tokenizing(txt, tokenizer)

        if auto_max_length:
            max_length = int(texts.apply(get_n_tokens).dropna().max())
        else:
            max_length = original_max_length

        # cap max_length to the model's supported maximum
        max_length = min(max_length, base_model_max_length)
        # evaluate the proportion of elements truncated
        percentage_truncated = int(100 * (texts.apply(get_n_tokens).dropna() > max_length).mean())

        if adapt:

            def tokenizing_function(e):
                return tokenizer(
                    e["text"],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=int(max_length),
                )
        else:

            def tokenizing_function(e):
                return tokenizer(
                    e["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=max_length,
                )

        return tokenizing_function, percentage_truncated, max_length

    def __load_trainer(
        self,
        current_path: Path,
        ds: datasets.DatasetDict,
        bert_model,
        params: LMParametersModel,
        loss: str,
    ) -> Trainer:
        """Load the training arguments and update the configuration"""

        # Calculate the number of steps (total, warmup and eval)
        has_test = "test" in ds

        total_steps = (float(params.epochs) * len(ds["train"])) // (
            int(params.batchsize) * float(params.gradacc)
        )
        warmup_steps = int((total_steps) // 10)
        eval_steps = (total_steps - warmup_steps) // params.eval
        eval_steps = max(eval_steps, 1)

        # Load the training arguments
        seed = int(config.random_seed)
        training_args = TrainingArguments(
            # Directories
            output_dir=str(current_path.joinpath("train")),
            #logging_dir=str(current_path.joinpath("logs")),
            # Hyperparameters
            learning_rate=float(params.lrate),
            weight_decay=float(params.wdecay),
            num_train_epochs=float(params.epochs),
            warmup_steps=int(warmup_steps),
            # Batch sizes
            gradient_accumulation_steps=int(params.gradacc),
            per_device_train_batch_size=int(params.batchsize),
            per_device_eval_batch_size=int(params.batchsize),
            # Logging and saving parameters
            eval_strategy="steps" if has_test else "no",
            eval_steps=eval_steps if has_test else None,
            # save_strategy must be "steps", not "best": on transformers 5.x
            # SaveStrategy.BEST never sets state.best_model_checkpoint (only
            # the STEPS/EPOCH branches track best_global_step), so
            # load_best_model_at_end silently reloads nothing and the saved /
            # exported model is the last-step one instead of the best (#1116).
            save_strategy="steps" if has_test else "epoch",
            metric_for_best_model="eval_loss" if has_test else None,
            save_steps=float(eval_steps) if has_test else 500,
            # Checkpoints are written every eval_steps, which collapses to a
            # few steps on small labelled sets. Optimizer state (~2/3 of the
            # checkpoint size) is only needed to resume training, which we
            # never do — load_best_model_at_end only reloads weights. Without
            # these caps a camembert-large run writes 3.4GB per save and can
            # fill the disk (and on slow volumes the saves dominate wall-clock).
            save_only_model=True,
            save_total_limit=2,
            logging_steps=int(eval_steps),
            do_eval=has_test,
            greater_is_better=False if has_test else None,
            load_best_model_at_end=params.best if has_test else False,
            use_cpu=config.cpu_only or not bool(params.gpu),  # deactivate gpu
            # Reproducibility: seed Trainer's model init / DataLoader shuffling
            # and dataset shuffling. config.random_seed defaults to 42.
            seed=seed,
            data_seed=seed,
            # No auto-attached reporting integrations: with codecarbon installed,
            # transformers would otherwise add its own CodeCarbonCallback, whose
            # NVML power query crashes Trainer init on GPUs where energy counters
            # are restricted (e.g. containers). Emissions are tracked by our
            # failure-safe EmissionsMonitor instead.
            report_to=[],
        )

        callback = CustomLoggingCallback(self.event, current_path=current_path, logger=self.logger)
        eval_dataset = ds["test"] if has_test else None
        if loss == "cross_entropy":
            trainer = Trainer(
                model=bert_model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=eval_dataset,
                callbacks=[callback],
            )
        elif loss == "weighted_cross_entropy":
            print("Using weighted cross entropy loss - EXPERIMENTAL")
            trainer = CustomTrainer(
                model=bert_model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=eval_dataset,
                callbacks=[callback],
                class_weights=compute_class_weights(ds["train"], label_key="labels"),
            )
        else:
            raise ValueError(f"Loss function {loss} not recognized.")

        # On very old transformers (<4.46) the attribute doesn't exist and the
        # Trainer always normalizes the logged loss itself: no correction.
        callback.needs_gradacc_correction = getattr(trainer, "model_accepts_loss_kwargs", False)

        return trainer

    def __create_save_files(
        self,
        current_path: Path,
        log_path: Path,
        df_train_results: pd.DataFrame,
        df_test_results: pd.DataFrame | None,
        training_data: pd.DataFrame,
        bert_model,
        tokenizer,
        params_to_save: dict[str, Any],
        metrics_train: MLStatisticsModel,
        metrics_test: MLStatisticsModel | None,
    ) -> None:
        """Save the model and parameters
        Save the following objects:
        - predictions of the train set (csv)
        - predictions of the test set  (csv)
        - data used during the training (parquet)
        - the trained model and its tokenizer
        - the parameters used during the training (json)
        - metrics (json)

        Also delete intermediate files
        """

        # Save results for the train and test set
        (
            df_train_results[
                [c for c in df_train_results.columns if c not in ["input_ids", "attention_mask"]]
            ].to_csv(current_path.joinpath("train_dataset_eval.csv"))
        )
        if df_test_results is not None:
            (
                df_test_results[
                    [c for c in df_test_results.columns if c not in ["input_ids", "attention_mask"]]
                ].to_csv(current_path.joinpath("test_dataset_eval.csv"))
            )
        training_data.to_parquet(current_path.joinpath("training_data.parquet"))

        # save the trained bert model with its tokenizer so the exported
        # archive can be loaded without fetching the base model from the hub
        bert_model.save_pretrained(current_path)
        tokenizer.save_pretrained(current_path)

        # Save parameters
        with open(current_path.joinpath("parameters.json"), "w") as f:
            json.dump(params_to_save, f)

        # remove intermediate steps and logs if succeed
        shutil.rmtree(current_path.joinpath("train"))
        os.rename(log_path, current_path.joinpath("finished"))

        # make archive (create dir if needed)
        path_static = f"{config.data_path}/projects/static/{self.project_slug}"
        os.makedirs(path_static, exist_ok=True)
        shutil.make_archive(
            f"{path_static}/{self.name}",
            "gztar",
            str(self.path.joinpath(self.name)),
        )

        metrics_data: dict[str, Any] = {
            "train": metrics_train.model_dump(mode="json"),
        }
        if metrics_test is not None:
            metrics_data["trainvalid"] = metrics_test.model_dump(mode="json")
        with open(str(current_path.joinpath("metrics_training.json")), "w") as f:
            json.dump(metrics_data, f)

    def __call__(self) -> EventsModel:
        """
        Main process to the task
        """
        task_timer = TaskTimer(compulsory_steps=["setup", "train", "evaluate", "save_files"])
        task_timer.start("setup")

        # Seed everything (python random, numpy, torch, torch.cuda) so that
        # successive runs of the same training produce the same model.
        # HF Trainer also gets seed via TrainingArguments below; both layers
        # are needed because operations that run before Trainer.__init__
        # (datasets shuffle, train_test_split) don't use HF's seed.
        seed = int(config.random_seed)
        set_seed(seed)

        current_path, log_path = self.__init_paths()
        self.logger = self.__init_logger(log_path)
        device = get_device()

        self.df = self.__check_data(
            self.df,  # ty: ignore[invalid-argument-type]
            self.col_label,
            self.col_text,
        )
        labels, label2id, id2label = self.__retrieve_labels(self.scheme_labels)
        self.ds = self.__transform_to_dataset(
            self.training_kind, self.df, self.col_label, self.col_text, label2id
        )

        tokenizer = self.__load_tokenizer(self.base_model)
        tokenizing_function, percentage_truncated, effective_max_length = (
            self.__cap_tokenizer_max_length(
                texts=self.df[self.col_text],
                tokenizer=tokenizer,
                auto_max_length=self.auto_max_length,
                original_max_length=self.max_length,
                base_model_max_length=retrieve_model_max_length(self.base_model),
                adapt=self.params.adapt,
            )
        )
        self.max_length = effective_max_length
        self.ds = self.ds.map(tokenizing_function, batched=True)

        # Build train/test dataset for dev eval
        if self.test_size > 0:
            self.ds = self.ds.train_test_split(test_size=self.test_size, seed=seed)
        else:
            self.ds = datasets.DatasetDict({"train": self.ds})
        self.logger.info("Train/test dataset created")

        # Model
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            # Some checkpoints (e.g. deberta-v3) store fp16 weights and
            # transformers>=5 loads them in the checkpoint dtype by default.
            # fp16 on CPU falls back to extremely slow kernels (and pure-fp16
            # training is numerically fragile anyway): always fine-tune in fp32.
            dtype=torch.float32,
            problem_type="multi_label_classification"
            if self.training_kind == "multilabel"
            else "single_label_classification",
        ).to(device=device)
        bert_model.config.use_cache = False
        self.logger.info(f"Model loaded on {bert_model.device}")
        print(f"Model loaded on {bert_model.device}")

        try:
            trainer = self.__load_trainer(
                current_path, self.ds, bert_model, self.params, self.loss or "cross_entropy"
            )
            task_timer.stop("setup")

            task_timer.start("train")
            trainer.train()
            self.logger.info(f"Model trained {current_path}")
            task_timer.stop("train")

            # predict on the data (separation validation set and training set)
            task_timer.start("evaluate")
            # Hoisted so it stays defined for the multiclass branch + final
            # params save; only meaningful when training_kind == "multilabel".
            threshold: float = 0.5
            train_ds = cast(datasets.Dataset, self.ds["train"])
            predictions_train = trainer.predict(cast(TorchDataset, train_ds))
            train_label_ids = cast(np.ndarray, predictions_train.label_ids)
            train_logits = cast(np.ndarray, predictions_train.predictions)

            # Compute the metrics
            df_train_results = cast(pd.DataFrame, train_ds.to_pandas()).set_index("id")

            df_train_results["true_label-matrix"] = train_label_ids.tolist()
            df_train_results["true_label"] = [
                "|".join(matrix_to_label(row, id2label)) for row in train_label_ids
            ]

            y_prob_pred = logits_to_probs(train_logits, self.training_kind)

            if self.training_kind == "multiclass":
                labels_predicted = activate_probs(
                    probs=y_prob_pred, strategy="max", force_max_1_per_row=True
                )
            elif self.training_kind == "multilabel":
                # threshold = find_best_threshold(
                #     y_true = predictions_train.label_ids,
                #     y_prob_pred = y_prob_pred,
                # )
                labels_predicted = activate_probs(
                    probs=y_prob_pred,
                    strategy="threshold",
                    threshold=threshold,
                    force_max_1_per_row=False,
                )

            df_train_results["predicted_label-matrix"] = y_prob_pred.tolist()
            df_train_results["predicted_label"] = [
                "|".join(matrix_to_label(row, id2label))
                for row in labels_predicted  # ty: ignore[possibly-unresolved-reference]
            ]

            if self.training_kind == "multiclass":
                metrics_train = get_metrics_multiclass(
                    Y_true=df_train_results["true_label"],
                    Y_pred=df_train_results["predicted_label"],
                    texts=df_train_results["text"],
                    id2label=id2label,
                )
            elif self.training_kind == "multilabel":
                metrics_train = get_metrics_multilabel(
                    Y_true=train_label_ids,
                    Y_pred=labels_predicted,  # ty: ignore[possibly-unresolved-reference]
                    texts=df_train_results["text"],
                    id2label=id2label,
                )

            if "test" in self.ds:
                test_ds = cast(datasets.Dataset, self.ds["test"])
                predictions_test = trainer.predict(cast(TorchDataset, test_ds))
                test_label_ids = cast(np.ndarray, predictions_test.label_ids)
                test_logits = cast(np.ndarray, predictions_test.predictions)

                df_test_results = cast(pd.DataFrame, test_ds.to_pandas()).set_index("id")

                df_test_results["true_label-matrix"] = test_label_ids.tolist()
                df_test_results["true_label"] = [
                    "|".join(matrix_to_label(row, id2label)) for row in test_label_ids
                ]

                y_prob_pred = logits_to_probs(test_logits, kind=self.training_kind)
                if self.training_kind == "multiclass":
                    y_label_pred = activate_probs(
                        y_prob_pred, strategy="max", force_max_1_per_row=True
                    )
                else:
                    y_label_pred = activate_probs(y_prob_pred, threshold, strategy="threshold")
                df_test_results["predicted_label-matrix"] = y_prob_pred.tolist()
                df_test_results["predicted_label"] = [
                    "|".join(matrix_to_label(row, id2label)) for row in y_label_pred
                ]

                if self.training_kind == "multiclass":
                    metrics_test = get_metrics_multiclass(
                        Y_true=df_test_results["true_label"],
                        Y_pred=df_test_results["predicted_label"],
                        texts=df_test_results["text"],
                        id2label=id2label,
                    )
                elif self.training_kind == "multilabel":
                    metrics_test = get_metrics_multilabel(
                        Y_true=test_label_ids,
                        Y_pred=y_label_pred,
                        texts=df_test_results["text"],
                        id2label=id2label,
                    )

            else:
                df_test_results = None
                metrics_test = None
            task_timer.stop("evaluate")

            task_timer.start("save_files")
            params_to_save = self.params.model_dump()
            params_to_save.update(
                {
                    "training_kind": self.training_kind,
                    "test_size": self.test_size,
                    "use_dichotomization": self.use_dichotomization,
                    "label_for_dichotomization": self.label_for_dichotomization,
                    "base_model": self.base_model,
                    "n_train": len(self.ds["train"]),
                    "max_length": self.max_length,
                    "device": str(device),
                    "Proportion of elements truncated (%)": percentage_truncated,
                    "loss": self.loss,
                    "auto context length": self.auto_max_length,
                    "balance classes": self.class_balance,
                    "class_min_freq": self.class_min_freq,
                }
            )
            if self.training_kind == "multilabel":
                params_to_save["threshold"] = threshold
            self.__create_save_files(
                current_path=current_path,
                log_path=log_path,
                df_train_results=df_train_results,
                df_test_results=df_test_results,
                training_data=self.df[[self.col_text, self.col_label]],
                bert_model=bert_model,
                tokenizer=tokenizer,
                params_to_save=params_to_save,
                metrics_train=metrics_train,  # ty: ignore[possibly-unresolved-reference]
                metrics_test=metrics_test,  # ty: ignore[possibly-unresolved-reference]
            )
            task_timer.stop("save_files")

        except Exception as e:
            print("Error in training", e)
            shutil.rmtree(current_path)
            # PyTorch can fail with a cryptic "NVML_SUCCESS == r INTERNAL ASSERT
            # FAILED at CUDACachingAllocator.cpp" while formatting an OOM error
            # (see pytorch#157535): both cases are GPU out-of-memory.
            if isinstance(e, torch.cuda.OutOfMemoryError) or "NVML_SUCCESS" in str(e):
                raise Exception(
                    "GPU ran out of memory during training. "
                    "Reduce the batch size or increase gradient accumulation."
                ) from e
            raise e
        finally:
            print("Cleaning memory")
            try:
                del (
                    trainer,
                    bert_model,
                    self.df,
                    self.ds,
                    device,
                    self.event,
                )
                release_device_memory()
                gc.collect()

            except Exception as e:
                print("Error in cleaning memory", e)

        return EventsModel(events=task_timer.get_events())
