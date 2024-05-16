import gc
import numpy
import pandas as pd
import pickle as pkl
import torch
from datasets import Dataset
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer, Trainer, TrainingArguments

from architectures.big_bird import EmbeddingBigBirdModel
from architectures.ordinal_regression_head import OrdinalRegressionClassifier
from utils import compute_metrics, convert_label_to_one_hot_encodings, get_embedding, load_embedding, remove_percentage, save_embedding


if __name__ == '__main__':
    num_epochs = 200
    num_classes = 5
    percentage_to_remove = 0.99

    load_embeddings = False
    
    train_data = pd.read_csv('./data/AMT10/AMT10_train.csv')
    val_data = pd.read_csv('./data/AMT10/AMT10_validation.csv')
    test_data = pd.read_csv('./data/AMT10/AMT10_test.csv')

    train_data = remove_percentage(train_data, percentage_to_remove)
    val_data = remove_percentage(val_data, percentage_to_remove)
    test_data = remove_percentage(test_data, percentage_to_remove)

    train_texts, train_labels = train_data['description'].tolist(), train_data['rating'].tolist()
    val_texts, val_labels = val_data['description'].tolist(), val_data['rating'].tolist()
    test_texts, test_labels = test_data['description'].tolist(), test_data['rating'].tolist()
    if load_embeddings:
        train_embeddings = load_embedding('./data/AMT10/train_embeddings.pkl')
        train_embeddings = torch.tensor(train_embeddings)
        val_embeddings = load_embedding('./data/AMT10/val_embeddings.pkl')
        val_embeddings = torch.tensor(val_embeddings)
        test_embeddings = load_embedding('./data/AMT10/test_embeddings.pkl')
        test_embeddings = torch.tensor(test_embeddings)
    else:

        model_name = "google/bigbird-roberta-base"
        tokenizer = BigBirdTokenizer.from_pretrained(model_name)
        model = BigBirdForSequenceClassification.from_pretrained(model_name)

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        embedding_model = EmbeddingBigBirdModel(model.bert)
        embedding_model.eval()

        train_embeddings = get_embedding(embedding_model, train_encodings)
        val_embeddings = get_embedding(embedding_model, val_encodings)
        test_embeddings = get_embedding(embedding_model, test_encodings)
        
        save_embedding(train_embeddings.cpu().numpy(), './data/AMT10/train_embeddings.pkl')
        save_embedding(val_embeddings.cpu().numpy(), './data/AMT10/val_embeddings.pkl')
        save_embedding(test_embeddings.cpu().numpy(), './data/AMT10/test_embeddings.pkl')
        # endif
    to_train_model = OrdinalRegressionClassifier(embeddings_size=768, num_classes=num_classes, intermediate_layers=[512, 128, 32])
    train_labels_tensor = convert_label_to_one_hot_encodings(train_labels, num_classes=num_classes)
    val_labels_tensor = convert_label_to_one_hot_encodings(val_labels, num_classes=num_classes)
    test_labels_tensor = convert_label_to_one_hot_encodings(test_labels, num_classes=num_classes)

    train_dataset = Dataset.from_dict(
        {
        "embeddings": train_embeddings, 
        "labels": train_labels_tensor
        })

    val_dataset = Dataset.from_dict(
        {
        "embeddings": val_embeddings, 
        "labels": val_labels_tensor

        })

    test_dataset = Dataset.from_dict(
        {
        "embeddings": test_embeddings, 
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
        model=to_train_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate the model on the testing set
    eval_results = trainer.evaluate(test_dataset)
    print(eval_results)
    
    torch.save(trainer.model.state_dict(), "./models/ordinal_regression_model.pth")
