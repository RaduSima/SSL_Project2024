import gc
import numpy
import pandas as pd
import pickle as pkl
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer, Trainer, TrainingArguments

from architectures.big_bird import EmbeddingBigBirdModel, OurBigBirdModel
from architectures.ordinal_regression_head import OrdinalRegressionClassifier


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


def convert_label_to_one_hot_encodings(labels: list[float], num_classes, max_label=3500):
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

    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(len(input_ids)):
            embedding = model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0)).squeeze(0).cpu()
            embeddings.append(embedding)
            gc.collect()
            torch.cuda.empty_cache()
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

if __name__ == '__main__':
    num_classes = 5
    percentage_to_remove = 0.99

    load_embeddings = True
    
    test_data = pd.read_csv('./data/AMT10/AMT10_test.csv')

    test_data = remove_percentage(test_data, percentage_to_remove)

    test_texts, test_labels = test_data['description'].tolist(), test_data['rating'].tolist()
    if load_embeddings:
        test_embeddings = load_embedding('./data/AMT10/test_embeddings.pkl')
    else:

        model_name = "google/bigbird-roberta-base"
        tokenizer = BigBirdTokenizer.from_pretrained(model_name)
        model = BigBirdForSequenceClassification.from_pretrained(model_name)

        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        embedding_model = EmbeddingBigBirdModel(model.bert)
        embedding_model.eval()

        test_embeddings = get_embedding(embedding_model, test_encodings)
        
        save_embedding(test_embeddings, './data/AMT10/test_embeddings.pkl')
        # endif
    test_labels_tensor = convert_label_to_one_hot_encodings(test_labels, num_classes=num_classes)

    classifier_model_weights = torch.load("./models/ordinal_regression_model.pth")
    classifier_model = OrdinalRegressionClassifier(embeddings_size=768, num_classes=num_classes, intermediate_layers=[512, 128, 32])
    classifier_model.load_state_dict(classifier_model_weights)
    
    # model = OurBigBirdModel(BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base"), num_classes=num_classes, intermediate_layers=[512, 128, 32])
    # model.classifier = classifier_model.head

    # classifier = model.classifier
    classifier = classifier_model.head
    outputs = []
    
    classifier.eval()
    with torch.no_grad():
      for i in range(len(test_embeddings)):
          logit, output = classifier(test_embeddings[i])
          outputs.append(output.unsqueeze(0))

    outputs = torch.cat(outputs).cpu().numpy()
    output_class = numpy.sum(outputs > 0.5, axis=-1) + 1
    labels = test_labels_tensor.cpu().numpy()
    target_class = numpy.sum(labels > 0.5, axis=-1) + 1

    for text, output, target in zip(test_texts, output_class, target_class):
      output_score = int(3500/num_classes * (output - 1)), int(3500/num_classes * output)
      target_score = int(3500/num_classes * (target - 1)), int(3500/num_classes * target)

      print(f"Predicted {output_score}, Target {target_score}\nFor text:\n{text}\n\n")
      
