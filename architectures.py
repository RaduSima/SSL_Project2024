import torch


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_features, num_classes, intermediate_layers=None):
        super(ClassificationHead, self).__init__()

        if intermediate_layers is None:
            intermediate_layers = []

        input_size = in_features
        layers = []
        for layer_size in intermediate_layers:
            layers.append(torch.nn.Linear(input_size, layer_size))
            layers.append(torch.nn.ReLU())
            input_size = layer_size
        layers.append(torch.nn.Linear(input_size, num_classes))

        self.fc = torch.nn.Sequential(*layers)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        return y, self.activation(y)


class EmbeddingBigBirdModel(torch.nn.Module):
    def __init__(self, bert):
        super(EmbeddingBigBirdModel, self).__init__()
        self.bert = bert
        # The output of the bert model is 768, as it is the output of the last hidden state.
        return

    def forward(self, input_ids, attention_mask):
        """
        The forward method for OurBigBirdModel class.

        Parameters
        ----------
        input_ids : tensor
            The input tensor, used for bert.
        attention_mask : tensor
            The attention mask tensor, used for bert.
        difficulty_ratings : tensor
            The target difficulty_ratings for the classification problem.

        Returns
        -------
        tuple(tensor, tensor)
            The loss and the output of the model.
            The loss is useful in the training phase, as we are using Trainer from HuggingFace.
        """
        return self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]


class OrdinalRegressionHead(torch.nn.Module):
    """
    Ordinal regression head for classification problems. The way it works is by having a single layer that outputs a single value, which is then added to a learnable bias. The output is then passed through a sigmoid function.
    The bias is a learnable parameter that is used to shift the output of the single layer to the desired range. It will be learned to have descending order.

    class = sum(b_i > 0.5) + 1

    class 1: 0, 0, 0, 0
    class 2: 1, 0, 0, 0
    class 3: 1, 1, 0, 0
    class 4: 1, 1, 1, 0
    class 5: 1, 1, 1, 1
    and so on ...

    TODO: should have more than one fc layer.
    """

    def __init__(self, in_features, num_classes, intermediate_layers=None):
        """
        The constructor for OrdinalRegressionHead class.

        Parameters
        ----------
        in_features : int
            The number of input features. This is the number of features of the output layer of the big model.
        num_classes : int
            The number of classes in the classification problem.
        """
        super(OrdinalRegressionHead, self).__init__()

        if intermediate_layers is None:
            intermediate_layers = []

        input_size = in_features
        layers = []
        for layer_size in intermediate_layers:
            layers.append(torch.nn.Linear(input_size, layer_size))
            layers.append(torch.nn.ReLU())
            input_size = layer_size
        layers.append(torch.nn.Linear(input_size, 1, bias=False))

        self.fc = torch.nn.Sequential(*layers)

        self.b = torch.nn.Parameter(torch.zeros(num_classes - 1))
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        """
        The forward method for OrdinalRegressionHead class.

        Parameters
        ----------
        x : tensor
            the input tensor.

        Returns
        -------
        tuple(tensor, tensor)
            The logits and the output of the model.
            The logits are useful in the training phase, as BCEWithLogitsLoss is used.
        """
        x = self.fc(x)
        y = x + self.b
        return y, self.activation(y)


class OrdinalRegressionClassifier(torch.nn.Module):
    def __init__(self, embedding_size, num_classes, intermediate_layers=None, **kwargs) -> None:
        super(OrdinalRegressionClassifier, self).__init__()
        self.head = OrdinalRegressionHead(
            embedding_size, num_classes, intermediate_layers)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, embeddings, labels=None):
        logits, output = self.head(embeddings)
        if labels is None:
            return output
        return self.loss(logits, labels), output


class TagClassifier(torch.nn.Module):
    def __init__(self, embedding_size, num_classes, intermediate_layers=None, **kwargs):
        super(TagClassifier, self).__init__()

        self.head = ClassificationHead(
            embedding_size, num_classes, intermediate_layers=intermediate_layers)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, embeddings, labels=None):
        logits, output = self.head(embeddings)
        if labels is None:
            return output
        return self.loss(logits, labels), output


class OurDifficultyClassifierModel(torch.nn.Module):
    """
    OurClassifierModel class is a class for the model that uses the BigBird model and the OrdinalRegressionHead.
    It uses the pretrained classifier of a BigBird model and adds an ordinal regression head on top of it.

    """

    def __init__(self, transformer, embedding_size=768, num_classes=5, intermediate_layers=None, **kwargs):
        super(OurDifficultyClassifierModel, self).__init__()

        self.transformer = transformer
        # The output of the transformer model is 768, as it is the output of the last hidden state.
        self.classifier = OrdinalRegressionHead(
            embedding_size, num_classes, intermediate_layers=intermediate_layers)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        The forward method for OurClassifierModel class.

        Parameters
        ----------
        input_ids : tensor
            The input tensor, used for transformer.
        attention_mask : tensor
            The attention mask tensor, used for transformer.
        labels : tensor
            The target labels for the classification problem.

        Returns
        -------
        tuple(tensor, tensor)
            The loss and the output of the model.
            The loss is useful in the training phase, as we are using Trainer from HuggingFace.
        """
        x = self.transformer(
            input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits, output = self.classifier(x)
        if labels is None:
            return output
        return self.loss(logits, labels), output


class OurTagClassifierModel(torch.nn.Module):
    def __init__(self, transformer, embedding_size=768, num_classes=5, intermediate_layers=None, **kwargs):
        super(OurTagClassifierModel, self).__init__()

        self.transformer = transformer
        # The output of the transformer model is 768, as it is the output of the last hidden state.
        self.classifier = ClassificationHead(
            embedding_size, num_classes, intermediate_layers=intermediate_layers)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.transformer(
            input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits, output = self.classifier(x)
        if labels is None:
            return output
        return self.loss(logits, labels), output
