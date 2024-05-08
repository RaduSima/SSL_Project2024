import torch


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
    def __init__(self, in_features, num_classes):
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
        self.in_features = in_features
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1, bias=False),
        )

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
    def __init__(self, embeddings_size, num_classes) -> None:
        super(OrdinalRegressionClassifier, self).__init__()
        self.head = OrdinalRegressionHead(embeddings_size, num_classes)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, embeddings, labels):
        logits, output = self.head(embeddings)
        return self.loss(logits, labels), output