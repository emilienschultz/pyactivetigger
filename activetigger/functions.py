import pandas as pd
import multiprocessing
from pandas import DataFrame, Series
import fasttext
from fasttext.util import download_model
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.manifold import TSNE
import umap
import bcrypt
import numpy as np
import torch
import os
import logging
import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
)
from transformers import Trainer, TrainingArguments, TrainerCallback
import json
import shutil


def get_hash(text: str):
    """
    Hash string
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(text.encode(), salt)
    return hashed


def compare_to_hash(text: str, hash: str | bytes):
    """
    Compare string to hash
    """
    if type(text) is str:
        text = text.encode()
    if type(hash) is str:
        hash = hash.encode()
    r = bcrypt.checkpw(text, hash)
    return r


def to_dtm(
    texts: Series,
    tfidf: bool = False,
    ngrams: int = 1,
    min_term_freq: int = 5,
    max_term_freq: int | float = 1.0,
    log: bool = False,
    norm=None,
    **kwargs,
):
    """
    Compute DFM embedding

    Norm :  None, l1, l2
    sublinear_tf : log
    Pas pris en compte : DFM : Min Docfreq
    https://quanteda.io/reference/dfm_tfidf.html
    + stop_words
    """
    if tfidf:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, ngrams),
            min_df=min_term_freq,
            sublinear_tf=log,
            norm=norm,
            max_df=max_term_freq,
        )
    else:
        vectorizer = CountVectorizer(
            ngram_range=(1, ngrams), min_df=min_term_freq, max_df=max_term_freq
        )

    dtm = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    dtm = pd.DataFrame(dtm.toarray(), columns=names, index=texts.index)
    return {"success": dtm}


def tokenize(texts: Series, model: str = "fr_core_news_sm") -> Series:
    """
    Clean texts with tokenization to facilitate word count
    TODO : faster tokenization ?
    """

    nlp = spacy.load(model)

    def tokenize(t, nlp):
        return " ".join([str(token) for token in nlp.tokenizer(t)])

    textes_tk = texts.apply(lambda x: tokenize(x, nlp))
    return textes_tk


def to_fasttext(texts: Series, language: str, path_models: Path, **kwargs) -> DataFrame:
    """
    Compute fasttext embedding
    Download the model if needed
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    if not path_models.exists():
        return {"error": f"path {str(path_models)} does not exist"}
    os.chdir(path_models)
    print(
        "If the model doesn't exist, it will be downloaded first. It could talke some time."
    )
    model_name = download_model(language, if_exists="ignore")
    print("Model loaded")
    texts_tk = tokenize(texts)
    ft = fasttext.load_model(model_name)
    emb = [ft.get_sentence_vector(t.replace("\n", " ")) for t in texts_tk]
    df = pd.DataFrame(emb, index=texts.index)
    df.columns = ["ft%03d" % (x + 1) for x in range(len(df.columns))]
    return {"success": df}


def to_sbert(
    texts: Series, model: str = "distiluse-base-multilingual-cased-v1", **kwargs
) -> DataFrame:
    """
    Compute sbert embedding
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    sbert = SentenceTransformer(model)
    sbert.max_seq_length = 512
    emb = sbert.encode(list(texts))
    emb = pd.DataFrame(emb, index=texts.index)
    emb.columns = ["sb%03d" % (x + 1) for x in range(len(emb.columns))]
    return {"success": emb}


def compute_umap(features: DataFrame, params: dict, **kwargs):
    """
    Compute UMAP
    """
    try:
        scaled_features = StandardScaler().fit_transform(features)
        reducer = umap.UMAP(**params)
        reduced_features = reducer.fit_transform(scaled_features)
        df = pd.DataFrame(reduced_features, index=features.index)
    except:
        df = DataFrame()
    return df


def compute_tsne(features: DataFrame, params: dict, **kwargs):
    """
    Compute TSNE
    """
    scaled_features = StandardScaler().fit_transform(features)
    reduced_features = TSNE(**params).fit_transform(scaled_features)
    df = pd.DataFrame(reduced_features, index=features.index)
    return df


def fit_model(model, X, Y, labels, **kwargs):
    """
    Fit simplemodel and calculate statistics
    """

    # drop NA values
    f = Y.notnull()
    Xf = X[f]
    Yf = Y[f]

    # fit model
    model.fit(Xf, Yf)

    # compute probabilities
    proba = model.predict_proba(X)
    proba = pd.DataFrame(proba, columns=model.classes_, index=X.index)
    proba["entropy"] = -1 * (proba * np.log(proba)).sum(axis=1)
    proba["prediction"] = proba.drop(columns="entropy").idxmax(axis=1)

    # compute statistics
    Y_pred = model.predict(Xf)
    f1 = f1_score(Yf, Y_pred, average=None)
    weighted_f1 = f1_score(Yf, Y_pred, average="weighted")
    accuracy = accuracy_score(Yf, Y_pred)
    precision = precision_score(
        list(Yf),
        list(Y_pred),
        average="micro",
    )
    macro_f1 = f1_score(Yf, Y_pred, average="macro")
    statistics = {
        "f1": list(f1),
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "precision": precision,
    }

    # compute 10-crossvalidation
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    Y_pred = cross_val_predict(model, Xf, Yf, cv=kf)
    weighted_f1 = f1_score(Yf, Y_pred, average="weighted")
    accuracy = accuracy_score(Yf, Y_pred)
    macro_f1 = f1_score(Yf, Y_pred, average="macro")
    cv10 = {
        "weighted_f1": round(weighted_f1, 3),
        "macro_f1": round(macro_f1, 3),
        "accuracy": round(accuracy, 3),
    }

    r = {"model": model, "proba": proba, "statistics": statistics, "cv10": cv10}
    return r


def train_bert(
    path: Path,
    name: str,
    df: DataFrame,
    col_text: str,
    col_label: str,
    base_model: str,
    params: dict,
    test_size: float,
    event: multiprocessing.Event,
) -> bool:
    """
    Train a bert model and write it

    Parameters:
    ----------
    path (Path): path to save the files
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    model (str): model to use
    params (dict) : training parameters
    test_size (dict): train/test distribution

    # pour le moment fichier status.log existe tant que l'entrainement est en cours
    """

    # check if GPU is available
    gpu = False
    if torch.cuda.is_available():
        print("GPU is available")
        gpu = True

    #  create repertory for the specific model
    current_path = path / name
    if not current_path.exists():
        os.makedirs(current_path)

    # logging the process
    log_path = current_path / "status.log"
    logger = logging.getLogger("train_bert_model")
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Start {base_model}")

    # test labels missing values
    if df[col_label].isnull().sum() > 0:
        df = df[df[col_label].notnull()]
        logger.info(f"Missing labels - reducing training data to {len(df)}")

    # test empty texts
    if df[col_text].isnull().sum() > 0:
        df = df[df[col_text].notnull()]
        logger.info(f"Missing texts - reducing training data to {len(df)}")

    # formatting data
    labels = sorted(list(df[col_label].dropna().unique()))  # alphabetical order
    label2id = {j: i for i, j in enumerate(labels)}
    id2label = {i: j for i, j in enumerate(labels)}
    training_data = df[[col_text, col_label]]
    df["labels"] = df[col_label].copy().replace(label2id)
    df["text"] = df[col_text]
    df = datasets.Dataset.from_pandas(df[["text", "labels"]])

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("tokenize")

    # Tokenize
    if params["adapt"]:
        df = df.map(
            lambda e: tokenizer(
                e["text"], truncation=True, padding=True, max_length=512
            ),
            batched=True,
        )
    else:
        df = df.map(
            lambda e: tokenizer(
                e["text"], truncation=True, padding="max_length", max_length=512
            ),
            batched=True,
        )

    # Build test dataset
    df = df.train_test_split(test_size=test_size)  # stratify_by_column="label"
    logger.info(f"Train/test dataset created")

    # Model
    bert = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    logger.info(f"Model loaded")

    if gpu:
        bert.cuda()

    total_steps = (params["epochs"] * len(df["train"])) // (
        params["batchsize"] * params["gradacc"]
    )
    warmup_steps = (total_steps) // 10
    eval_steps = total_steps // params["eval"]

    training_args = TrainingArguments(
        output_dir=current_path / "train",
        logging_dir=current_path / "logs",
        learning_rate=params["lrate"],
        weight_decay=params["wdecay"],
        num_train_epochs=params["epochs"],
        gradient_accumulation_steps=params["gradacc"],
        per_device_train_batch_size=params["batchsize"],
        per_device_eval_batch_size=32,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=eval_steps,
        logging_steps=eval_steps,
        do_eval=True,
        greater_is_better=False,
        load_best_model_at_end=params["best"],
        metric_for_best_model="eval_loss",
    )

    logger.info(f"Start training")

    class CustomLoggingCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            logger.info(f"Step {state.global_step}")
            # end if event set
            if event.is_set():
                logger.info("Event set, stopping training.")
                control.should_training_stop = True

    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=df["train"],
        eval_dataset=df["test"],
        callbacks=[CustomLoggingCallback()],
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        shutil.rmtree(current_path)
        return False

    # save model
    bert.save_pretrained(current_path)
    logger.info(f"Model trained {current_path}")

    # save training data
    training_data.to_parquet(current_path / "training_data.parquet")

    # save parameters
    params["test_size"] = test_size
    params["base_model"] = base_model
    with open(current_path / "parameters.json", "w") as f:
        json.dump(params, f)

    # remove intermediate steps and logs if succeed
    shutil.rmtree(current_path / "train")
    os.rename(log_path, current_path / "finished")

    # save log history of the training for statistics
    with open(current_path / "log_history.txt", "w") as f:
        json.dump(trainer.state.log_history, f)

    # remove temporary logs
    os.remove(log_path)

    return True


def predict_bert(
    model,
    tokenizer,
    path: Path,
    df: DataFrame,
    col_text: str,
    event: multiprocessing.Event,
    col_labels: str | None = None,
    batch: int = 128,
    file_name: str = "predict.parquet",
) -> DataFrame | bool:
    """
    Predict from a model
    + probabilities
    + entropy
    """
    # check if GPU available
    gpu = False
    if torch.cuda.is_available():
        print("GPU is available")
        gpu = True

    # logging the process
    log_path = path / "status.log"
    logger = logging.getLogger("predict_bert_model")
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print("function prediction : start")
    if torch.cuda.is_available():
        model.cuda()

    # Start prediction with batches
    predictions = []
    # logging the process
    for chunk in [df[col_text][i : i + batch] for i in range(0, df.shape[0], batch)]:
        # user interrupt
        if event.is_set():
            logger.info("Event set, stopping training.")
            return False

        print("Next chunck prediction")
        chunk = tokenizer(
            list(chunk),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if gpu:
            chunk = chunk.to("cuda")
        with torch.no_grad():
            outputs = model(**chunk)
        res = outputs[0]
        if gpu:
            res = res.cpu()
        res = res.softmax(1).detach().numpy()
        predictions.append(res)

    # to dataframe
    pred = pd.DataFrame(
        np.concatenate(predictions),
        columns=sorted(list(model.config.label2id.keys())),
        index=df.index,
    )

    # calculate entropy
    entropy = -1 * (pred * np.log(pred)).sum(axis=1)
    pred["entropy"] = entropy

    # calculate label
    pred["prediction"] = pred.drop(columns="entropy").idxmax(axis=1)

    # if asked, add the label column for latter statistics
    if col_labels:
        pred[col_labels] = df[col_labels]

    # write the content in a parquet file
    pred.to_parquet(path / file_name)
    # delete the logs
    os.remove(log_path)
    print("function prediction : finished")
    return pred


def truncate_text(text: str, max_tokens: int = 512):
    """
    Limit a text to a specific number of tokens
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    if num_tokens > max_tokens:
        print(num_tokens)
        truncated_tokens = tokens[:max_tokens]
        text_t = tokenizer.convert_tokens_to_string(truncated_tokens)
    else:
        text_t = text
    return text_t


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
