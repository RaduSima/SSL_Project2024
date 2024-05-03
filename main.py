import numpy
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer, Trainer, TrainingArguments

from architectures.big_bird import OurBigBirdModel


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
    num_rows_to_remove = int(len(df) * percent)
    df_removed = df.sample(frac=1).iloc[num_rows_to_remove:]
    return df_removed


def convert_label_to_one_hot_encodings(labels: list[float], num_classes, max_label=2800):
    """
    Convert the labels to a class representation. The class representation is a one-hot encoding of the labels.
    This preparation is for the ordinal regression problem.
    
    If the one-hot encoding of the label 3 for multi-class classification is [0, 0, 1, 0, 0],
      then the one-hot encoding of label 3 for ordinal regression is [1, 1, 0, 0].

    To convert from multi-class to ordinal regression, we can do the following trick:
        - compute the one-hot encoding of the labels for multi-class classification
        - replace all the 0s with 1s until the first 1 is found (for multi-class classification a 1 is found at the index of the class)
        - remove the first column of the one-hot encoding (the first 1)

    Parameters
    ----------
    labels : list[float]
        The list of labels to convert.
    num_classes : int
        The number of classes in the classification problem.
    max_label : int, optional
        The maximum label in the dataset. This is used to normalize the labels. The default is 2800.

    Returns
    -------
    tensor
        The tensor of the one-hot encoding of the labels.
    """
    # reminder: in ordinal regression, class = sum(output > 0.5) + 1

    # normalize the labels
    labels = numpy.array(labels) / max_label
    class_labels = numpy.zeros((len(labels), num_classes - 1))
    for i, label in enumerate(labels):
        class_labels[i] = numpy.array(
            [1 if j / num_classes <= label else 0 for j in range(1, num_classes)])
    return torch.tensor(class_labels)


def compute_metrics(pred):
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
    threshold = 0.5

    labels = pred.label_ids
    preds = pred.predictions

    target_class = numpy.sum(labels > threshold, axis=-1) + 1
    output_class = numpy.sum(preds > threshold, axis=-1) + 1

    # compute accuracy, recall, precision, f1 for threshold 0.5
    accuracy = accuracy_score(target_class, output_class)
    recall = recall_score(target_class, output_class, average='macro')
    precision = precision_score(target_class, output_class, average='macro')
    f1 = f1_score(target_class, output_class, average='macro')

    # compute neighborhood accuracy -- consider accurate all predictions that are off by 1
    neighborhood_accuracy = numpy.sum(
        numpy.abs(target_class - output_class) <= 1).item() / (len(labels) * 1.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "neighborhood_accuracy": neighborhood_accuracy
    }


if __name__ == '__main__':
    labels = list(range(10))
    onehot = convert_label_to_one_hot_encodings(labels, 5, 9)

    print(labels)
    print(onehot)

if __name__ == '2__main__':
    num_epochs = 3
    num_classes = 5

    percentage_to_remove = 0.998
    train_data = pd.read_csv('./data/AMT10/AMT10_train.csv')
    val_data = pd.read_csv('./data/AMT10/AMT10_validation.csv')
    test_data = pd.read_csv('./data/AMT10/AMT10_test.csv')

    train_data = remove_percentage(train_data, percentage_to_remove)
    val_data = remove_percentage(val_data, percentage_to_remove)
    test_data = remove_percentage(test_data, percentage_to_remove)

    train_texts, train_labels = train_data['description'].tolist(), train_data['rating'].tolist()
    val_texts, val_labels = val_data['description'].tolist(), val_data['rating'].tolist()
    test_texts, test_labels = test_data['description'].tolist(), test_data['rating'].tolist()

    model_name = "google/bigbird-roberta-base"
    tokenizer = BigBirdTokenizer.from_pretrained(model_name)
    model = BigBirdForSequenceClassification.from_pretrained(model_name)

    model = OurBigBirdModel(model.bert, num_classes=num_classes)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_labels_tensor = convert_label_to_one_hot_encodings(train_labels, num_classes=num_classes)
    val_labels_tensor = convert_label_to_one_hot_encodings(val_labels, num_classes=num_classes)
    test_labels_tensor = convert_label_to_one_hot_encodings(test_labels, num_classes=num_classes)

    train_dataset = Dataset.from_dict(
        {
        "input_ids": train_encodings["input_ids"], 
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels_tensor
        })

    val_dataset = Dataset.from_dict(
        {
        "input_ids": val_encodings["input_ids"], 
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_labels_tensor
        })

    test_dataset = Dataset.from_dict(
        {
        "input_ids": test_encodings["input_ids"], 
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels_tensor
        })

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate the model on the testing set
    eval_results = trainer.evaluate(test_dataset)
    print(eval_results)
