import numpy
import pandas as pd
import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer

from architectures.big_bird import EmbeddingBigBirdModel, OurBigBirdModel
from architectures.ordinal_regression_head import OrdinalRegressionClassifier
from utils import _compute_metrics, convert_label_to_one_hot_encodings, get_embedding, load_embedding, remove_percentage, save_embedding


if __name__ == '__main__':
    num_classes = 5
    percentage_to_remove = 0.99

    load_embeddings = True
    
    test_data = pd.read_csv('./data/AMT10/AMT10_test.csv')

    test_data = remove_percentage(test_data, percentage_to_remove)

    test_texts, test_labels = test_data['description'].tolist(), test_data['rating'].tolist()
    if load_embeddings:
        test_embeddings = load_embedding('./data/AMT10/test_embeddings.pkl')
        test_embeddings = torch.tensor(test_embeddings)
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

    for text, label, output, target in zip(test_texts, test_labels, output_class, target_class):
      output_score = int(3500/num_classes * (output - 1)), int(3500/num_classes * output)
      target_score = int(3500/num_classes * (target - 1)), int(3500/num_classes * target)

      print(f"Predicted {output_score}, Target {target_score}, Real label {int(label)}\nFor text:\n{text}\n\n")
      
    print(_compute_metrics(outputs, labels))
