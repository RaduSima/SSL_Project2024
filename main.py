import pandas as pd
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

def remove_percentage(df, percent):
    num_rows_to_remove = int(len(df) * percent)
    df_removed = df.sample(frac=1).iloc[num_rows_to_remove:]
    return df_removed

if __name__ == '__main__':
    percentage_to_remove = 0.999
    num_epochs = 1

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

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_labels_tensor = torch.tensor(train_labels)
    val_labels_tensor = torch.tensor(val_labels)
    test_labels_tensor = torch.tensor(test_labels)

    train_dataset = Dataset.from_dict(
        {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"],
         "labels": train_labels_tensor})
    val_dataset = Dataset.from_dict(
        {"input_ids": val_encodings["input_ids"], "attention_mask": val_encodings["attention_mask"],
         "labels": val_labels_tensor})
    test_dataset = Dataset.from_dict(
        {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"],
         "labels": test_labels_tensor})

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
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
    )

    trainer.train()

    # Evaluate the model on the testing set
    eval_results = trainer.evaluate(test_dataset)
    print(eval_results)