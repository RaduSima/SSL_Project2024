import gc
import os
import pickle as pkl
from time import time as tm

import numpy
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score)
from transformers import (BigBirdForSequenceClassification, BigBirdTokenizer, BigBirdModel,
                          T5ForSequenceClassification, T5Tokenizer)

from architectures import (EmbeddingBigBirdModel, OrdinalRegressionClassifier,
                           OurDifficultyClassifierModel, OurTagClassifierModel,
                           TagClassifier)

transformers_repo = [
    "google/bigbird-roberta-base",
    "google/bigbird-roberta-large",
    "albert-base-v2",
    "albert-large-v2",
    "allenai/longformer-base-4096",
    "t5-small",
    "t5-base",
    "t5-large",
    "facebook/bart-base",
    "facebook/bart-large",
    "microsoft/deberta-base",
    "microsoft/deberta-large",
]

transformers_classes = {
    "google/bigbird-roberta-base": {
        "model_class": BigBirdModel,
        "tokenizer_class": BigBirdTokenizer,
        "embedding_size": 768,
    },
    "google/bigbird-roberta-large": {
        "model_class": BigBirdModel,
        "tokenizer_class": BigBirdTokenizer,
        "embedding_size": 1024,
    },
    "t5-small": {
        "model_class": T5ForSequenceClassification,
        "tokenizer_class": T5Tokenizer,
        "embedding_size": 512,
    },
    "t5-base": {
        "model_class": T5ForSequenceClassification,
        "tokenizer_class": T5Tokenizer,
        "embedding_size": 512,
    },
    "t5-large": {
        "model_class": T5ForSequenceClassification,
        "tokenizer_class": T5Tokenizer,
        "embedding_size": 512,
    },
}

MAX_DIFFICULTY_RATING = 3500
MIN_DIFFICULTY_RATING = 800


def remove_percentage(df, percent):
    """
    Useful method for development purposes. It removes a percentage of the rows from the dataframe.

    Parameters
    ----------
    df : dataframe
        The dataframe to remove rows from.
    percent : float
        The percentage of rows to remove.

    Returns
    -------
    dataframe
        The dataframe with the rows removed.
    """
    if percent == 0:
        return df
    numpy.random.seed(42)
    num_rows_to_remove = int(len(df) * percent)
    df_removed = df.sample(frac=1).iloc[num_rows_to_remove:]
    return df_removed


def convert_tags_to_one_hot_encodings(tags: list[list[str]], tag2id: dict):
    """
    Convert the tags to a class representation. The class representation is a one-hot encoding of the tags.

    Parameters
    ----------
    tags : list[list[str]]
        The list of tags to convert.
    tag2id : dict
        The dictionary that maps the tags to their ids.
    num_tags : int
        The number of tags in the classification problem.

    Returns
    -------
    tensor
        The tensor of the one-hot encoding of the tags.
    """
    tags_tensor = numpy.zeros((len(tags), len(tag2id)))
    for i, tag_list in enumerate(tags):
        for tag in tag_list:
            tags_tensor[i][tag2id[tag]] = 1
    return torch.tensor(tags_tensor)


def convert_difficulty_difficulty_rating_to_one_hot_encodings(difficulty_ratings: list[float], num_classes,
                                                              max_difficulty_rating=MAX_DIFFICULTY_RATING,
                                                              min_difficulty_rating=MIN_DIFFICULTY_RATING):
    """
    Convert the difficulty_ratings to a class representation. The class representation is a one-hot encoding of the difficulty_ratings.
    This preparation is for the ordinal regression problem.

    If the one-hot encoding of the difficulty_rating 3 for multi-class classification is [0, 0, 1, 0, 0],
      then the one-hot encoding of difficulty_rating 3 for ordinal regression is [1, 1, 0, 0].

    To convert from multi-class to ordinal regression, we can do the following trick:
        - compute the one-hot encoding of the difficulty_ratings for multi-class classification
        - replace all the 0s with 1s until the first 1 is found (for multi-class classification a 1 is found at the index of the class)
        - remove the first column of the one-hot encoding (the first 1)

    Parameters
    ----------
    difficulty_ratings : list[float]
        The list of difficulty_ratings to convert.
    num_classes : int
        The number of classes in the classification problem.
    max_difficulty_rating : int, optional
        The maximum difficulty_rating in the dataset. This is used to normalize the difficulty_ratings. The default is 2800.

    Returns
    -------
    tensor
        The tensor of the one-hot encoding of the difficulty_ratings.
    """
    # reminder: in ordinal regression, class = sum(output > 0.5) + 1

    # normalize the difficulty_ratings
    difficulty_ratings = numpy.array(
        difficulty_ratings) / (max_difficulty_rating - min_difficulty_rating)
    class_difficulty_ratings = numpy.zeros(
        (len(difficulty_ratings), num_classes - 1))
    for i, difficulty_rating in enumerate(difficulty_ratings):
        class_difficulty_ratings[i] = numpy.array(
            [1 if j / num_classes <= difficulty_rating else 0 for j in range(1, num_classes)])
    return torch.tensor(class_difficulty_ratings)


def create_metrics_tag_function(threshold):
    """
    Create a metrics function that includes the threshold parameter.

    Parameters
    ----------
    threshold : float
        The threshold value to be used in the metrics computation.

    Returns
    -------
    function
        A function that computes metrics using the specified threshold.
    """

    def wrapper(pred):
        return compute_metrics_tag_classifier(pred, threshold)

    return wrapper


def create_metrics_difficulty_function(threshold):
    """
    Create a metrics function that includes the threshold parameter.

    Parameters
    ----------
    threshold : float
        The threshold value to be used in the metrics computation.

    Returns
    -------
    function
        A function that computes metrics using the specified threshold.
    """

    def wrapper(pred):
        return compute_metrics_difficulty_classifier(pred, threshold)

    return wrapper


def compute_metrics_difficulty_classifier(pred, threshold):
    """
    Compute the metrics for the model. The metrics are accuracy, recall, precision, f1, and neighborhood accuracy.

    Parameters
    ----------
    pred : object
        The predictions of the model, wrapped in an object by Hugging Face trainer.

    Returns
    -------
    dict
        The dictionary of the metrics.
    """
    difficulty_ratings = pred.label_ids
    preds = pred.predictions

    return _compute_metrics_difficulty_classifier(preds, difficulty_ratings, threshold)


def _compute_metrics_difficulty_classifier(preds, difficulty_ratings, threshold):
    target_class = numpy.sum(difficulty_ratings > threshold, axis=-1) + 1
    output_class = numpy.sum(preds > threshold, axis=-1) + 1

    # compute accuracy, recall, precision, f1 for threshold 0.5
    accuracy = accuracy_score(target_class, output_class)
    recall = recall_score(target_class, output_class,
                          average='macro', zero_division=0)
    precision = precision_score(
        target_class, output_class, average='macro', zero_division=0)
    f1 = f1_score(target_class, output_class, average='macro', zero_division=0)

    # compute neighborhood accuracy -- consider accurate all predictions that are off by 1
    neighborhood_accuracy = numpy.sum(
        numpy.abs(target_class - output_class) <= 1).item() / (len(difficulty_ratings) * 1.0)

    unique_classes = numpy.unique(target_class)
    ap_scores = []
    for cls in unique_classes:
        binarized_target = (target_class == cls).astype(int)
        binarized_output = (output_class == cls).astype(int)
        ap = average_precision_score(binarized_target, binarized_output)
        ap_scores.append(ap)
    mean_average_precision = numpy.mean(ap_scores)

    neighborhood_accuracy3 = numpy.sum(
        numpy.abs(target_class - output_class) <= 3).item() / (len(difficulty_ratings) * 1.0)
    
    neighborhood_accuracy5 = numpy.sum(
        numpy.abs(target_class - output_class) <= 5).item() / (len(difficulty_ratings) * 1.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "neighborhood_accuracy": neighborhood_accuracy,
        "neighborhood_accuracy3": neighborhood_accuracy3,
        "neighborhood_accuracy5": neighborhood_accuracy5,
        "map": mean_average_precision
    }


def _compute_metrics_tag_classifier(preds, tags, threshold):
    # compute accuracy, recall, precision, f1
    print("Hey!")
    pred_classes = numpy.array(preds > threshold, dtype=int)
    accuracy = accuracy_score(tags, pred_classes)
    recall = recall_score(tags, pred_classes, average='macro', zero_division=0)
    precision = precision_score(
        tags, pred_classes, average='macro', zero_division=0)
    f1 = f1_score(tags, pred_classes, average='macro', zero_division=0)

    num_classes = tags.shape[1]
    ap_scores = []
    for i in range(num_classes):
        ap = average_precision_score(tags[:, i], preds[:, i])
        ap_scores.append(ap)
    mean_average_precision = numpy.mean(ap_scores)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": mean_average_precision
    }


def compute_metrics_tag_classifier(pred, threshold):
    """
    Compute the metrics for the model. The metrics are accuracy, recall, precision, f1, and neighborhood accuracy.

    Parameters
    ----------
    pred : object
        The predictions of the model, wrapped in an object by Hugging Face trainer.

    Returns
    -------
    dict
        The dictionary of the metrics.
    """
    print(pred)
    tags = pred.label_ids
    preds = pred.predictions

    return _compute_metrics_tag_classifier(preds, tags, threshold)


def get_embedding(model, encoding):
    """
    Get the embedding from the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the embeddings from.
    encoding : dict
        The encoding of the text.

    Returns
    -------
    tensor
        The tensor of the embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = torch.tensor(encoding["input_ids"]).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"]).to(device)

    batch_size = 4
    model = model.to(device)
    model.eval()
    embeddings = []

    start_time = tm()

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]
            embedding = model(
                batch_input_ids, attention_mask=batch_attention_mask).to("cpu")
            embeddings.append(embedding)
            gc.collect()
            torch.cuda.empty_cache()
            elapsed_time = int(tm() - start_time)
            time_per_iter = round(elapsed_time / (i + batch_size), 2)
            remaining_time = int((len(input_ids) - i) * time_per_iter)
            print(
                f"Status {i + batch_size}/{len(input_ids)}  Elapsed: {elapsed_time}s ({time_per_iter}s/it)  Remaining: {remaining_time}s",
                end="\r")
    print("Embeddings done.")
    del input_ids
    del attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings)
    return embeddings


def save_embedding(embeddings, filename):
    """
    Save the embeddings to a file.

    Parameters
    ----------
    embeddings : tensor
        The tensor of the embeddings.
    filename : str
        The filename to save the embeddings to.
    """
    with open(filename, 'wb') as f:
        pkl.dump(embeddings, f)


def load_embedding(filename):
    """
    Load the embeddings from a file.

    Parameters
    ----------
    filename : str
        The filename to load the embeddings from.

    Returns
    -------
    tensor
        The tensor of the embeddings.
    """
    with open(filename, 'rb') as f:
        embeddings = pkl.load(f)
    return embeddings


def prepare_dataset_tag_classifier(embeddings, tags, tag2id):
    tags_tensor = convert_tags_to_one_hot_encodings(tags, tag2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags_tensor = tags_tensor.to(device)
    embeddings = embeddings.to(device)
    dataset = Dataset.from_dict(
        {
            "embeddings": embeddings,
            "labels": tags_tensor
        })
    return dataset


def prepare_dataset_difficulty_classifier(embeddings, difficulty_ratings, num_classes=5,
                                          max_difficulty_rating=MAX_DIFFICULTY_RATING,
                                          min_difficulty_rating=MIN_DIFFICULTY_RATING):
    difficulty_ratings_tensor = convert_difficulty_difficulty_rating_to_one_hot_encodings(
        difficulty_ratings, num_classes, max_difficulty_rating=max_difficulty_rating, min_difficulty_rating=min_difficulty_rating)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    difficulty_ratings_tensor = difficulty_ratings_tensor.to(device)
    embeddings = embeddings.to(device)
    dataset = Dataset.from_dict(
        {
            "embeddings": embeddings,
            "labels": difficulty_ratings_tensor
        })
    return dataset


def prepare_finetune_dataset_tag_classifier(transformer_name, texts, tags, tag2id):
    tags_tensor = convert_tags_to_one_hot_encodings(tags, tag2id)

    tokenizer = transformers_classes[transformer_name]["tokenizer_class"].from_pretrained(
        transformer_name)
    embeddings = tokenizer(texts, truncation=True, padding=True)

    dataset = Dataset.from_dict(
        {
            "input_ids": embeddings["input_ids"],
            "attention_mask": embeddings["attention_mask"],
            "labels": tags_tensor
        })
    return dataset


def prepare_finetune_dataset_difficulty_classifier(transformer_name, texts, difficulty_ratings, num_classes=5,
                                                   max_difficulty_rating=MAX_DIFFICULTY_RATING,
                                                   min_difficulty_rating=MIN_DIFFICULTY_RATING):
    difficulty_ratings_tensor = convert_difficulty_difficulty_rating_to_one_hot_encodings(
        difficulty_ratings, num_classes, max_difficulty_rating=max_difficulty_rating, min_difficulty_rating=min_difficulty_rating)

    tokenizer = transformers_classes[transformer_name]["tokenizer_class"].from_pretrained(
        transformer_name)
    embeddings = tokenizer(texts, truncation=True, padding=True)

    dataset = Dataset.from_dict(
        {
            "input_ids": embeddings["input_ids"],
            "attention_mask": embeddings["attention_mask"],
            "labels": difficulty_ratings_tensor
        })
    return dataset


def _compute_full_embeddings(transformer_name, texts, set_name, base_path):
    tokenizer = transformers_classes[transformer_name]["tokenizer_class"].from_pretrained(
        transformer_name)
    model = transformers_classes[transformer_name]["model_class"].from_pretrained(
        transformer_name)

    train_encodings = tokenizer(texts, truncation=True, padding=True)

    embedding_model = EmbeddingBigBirdModel(model)
    embedding_model.eval()

    train_embeddings = get_embedding(embedding_model, train_encodings)

    transformer_name_path = transformer_name.replace("/", "_")
    save_embedding(train_embeddings.cpu().numpy(
    ), f"{base_path}/{transformer_name_path}_{set_name}_embeddings.pkl")
    return train_embeddings


def maybe_load_embeddings(transformer_name, texts, set_name, base_path):
    transformer_name_path = transformer_name.replace("/", "_")
    embeddings_filename = f"{base_path}/{transformer_name_path}_{set_name}_embeddings.pkl"

    if os.path.exists(embeddings_filename):
        embeddings = load_embedding(embeddings_filename)
        embeddings = torch.tensor(embeddings)
    else:
        embeddings = _compute_full_embeddings(
            transformer_name, texts, set_name, base_path)
    return embeddings


def get_model_class_from_name(name):
    if name == "OurClassifierModel":
        return OurDifficultyClassifierModel
    elif name == "OrdinalRegressionClassifier":
        return OrdinalRegressionClassifier
    elif name == "OurTagClassifierModel":
        return OurTagClassifierModel
    elif name == "TagClassifier":
        return TagClassifier
    else:
        return None


def load_torch_model(model_name):
    dct_model_params = torch.load(f"./models/{model_name}.pth")
    hyperparameters = dct_model_params.pop("hyperparameters", None)

    if hyperparameters is None:
        raise ValueError("The model does not have hyperparameters saved.")

    model_class = get_model_class_from_name(hyperparameters["model_class"])
    model = model_class(**hyperparameters)
    model.load_state_dict(dct_model_params)
    return model, hyperparameters


def save_torch_model(model, hyperparameters, model_name):
    dct_to_save = model.state_dict()
    dct_to_save["hyperparameters"] = hyperparameters

    torch.save(dct_to_save, f"./models/{model_name}.pth")


def load_tag_classifier_model(model_name, use_pretrained_transformer=False):
    loaded_model, hyperparameters = load_torch_model(model_name)

    transformer_name = hyperparameters["transformer_name"]

    tokenizer = transformers_classes[transformer_name]["tokenizer_class"].from_pretrained(
        transformer_name)
    transformer = transformers_classes[transformer_name]["model_class"].from_pretrained(
        transformer_name)
    model = OurTagClassifierModel(
        transformer=transformer,
        embedding_size=transformers_classes[transformer_name]["embedding_size"],
        num_classes=len(hyperparameters["tag2id"]),
        intermediate_layers=hyperparameters["intermediate_layers"],
    )

    if use_pretrained_transformer:
        model.classifier.load_state_dict(loaded_model.head.state_dict())
    else:
        model.load_state_dict(loaded_model.state_dict())

    model.eval()
    return model, tokenizer, hyperparameters


def load_difficulty_classifier_model(model_name, use_pretrained_transformer=False):
    loaded_model, hyperparameters = load_torch_model(model_name)

    transformer_name = hyperparameters["transformer_name"]

    tokenizer = transformers_classes[transformer_name]["tokenizer_class"].from_pretrained(
        transformer_name)
    transformer = transformers_classes[transformer_name]["model_class"].from_pretrained(
        transformer_name)
    model = OurDifficultyClassifierModel(
        transformer=transformer,
        embedding_size=transformers_classes[transformer_name]["embedding_size"],
        num_classes=hyperparameters["num_classes"],
        intermediate_layers=hyperparameters["intermediate_layers"],
    )

    if use_pretrained_transformer:
        model.classifier.load_state_dict(loaded_model.head.state_dict())
    else:
        model.load_state_dict(loaded_model.state_dict())

    model.eval()
    return model, tokenizer, hyperparameters


def predict_difficulty(input, model, tokenizer, hyperparameters):
    encoding = tokenizer(input, truncation=True, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = torch.tensor(encoding["input_ids"]).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"]).to(device)

    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask=attention_mask).detach().cpu().numpy()
    difficulty_class = numpy.sum(predictions > 0.41, axis=-1) + 1

    num_classes = hyperparameters["num_classes"]

    min_difficulty_rating = (difficulty_class - 1) * \
                            (MAX_DIFFICULTY_RATING - MIN_DIFFICULTY_RATING) / num_classes + MIN_DIFFICULTY_RATING
    max_difficulty_rating = difficulty_class * (MAX_DIFFICULTY_RATING - MIN_DIFFICULTY_RATING) / num_classes + MIN_DIFFICULTY_RATING

    difficulty_range = [(m, M) for m, M in
                        zip(numpy.uint32(min_difficulty_rating), numpy.uint32(max_difficulty_rating))]

    if len(difficulty_range) == 1:
        difficulty_range = difficulty_range[0]

    return difficulty_range


def predict_tags(input, model, tokenizer, hyperparameters):
    encoding = tokenizer(input, truncation=True, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = torch.tensor(encoding["input_ids"]).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"]).to(device)

    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask=attention_mask).detach().cpu().numpy()
    tags = numpy.array(predictions > 0.4, dtype=int)

    tags = [i for i, tag in enumerate(tags[0]) if tag == 1]

    tag2id = hyperparameters["tag2id"]
    id2tag = {v: k for k, v in tag2id.items()}
    tags = [id2tag[tag] for tag in tags]
    return tags


