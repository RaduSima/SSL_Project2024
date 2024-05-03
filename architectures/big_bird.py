import torch
from architectures.ordinal_regression_head import OrdinalRegressionHead


class OurBigBirdModel(torch.nn.Module):
    """
    OurBigBirdModel class is a class for the model that uses the BigBird model and the OrdinalRegressionHead.
    It uses the pretrained classifier of a BigBird model and adds an ordinal regression head on top of it.

    """
    def __init__(self, bert, num_classes=5):
        super(OurBigBirdModel, self).__init__()

        self.bert = bert
        # The output of the bert model is 768, as it is the output of the last hidden state.
        self.classifier = OrdinalRegressionHead(768, num_classes)
        
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels):
        """
        The forward method for OurBigBirdModel class.

        Parameters
        ----------
        input_ids : tensor
            The input tensor, used for bert.
        attention_mask : tensor
            The attention mask tensor, used for bert.
        labels : tensor
            The target labels for the classification problem.

        Returns
        -------
        tuple(tensor, tensor)
            The loss and the output of the model.
            The loss is useful in the training phase, as we are using Trainer from HuggingFace.
        """
        x = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits, output = self.classifier(x)
        return self.loss(logits, labels), output
