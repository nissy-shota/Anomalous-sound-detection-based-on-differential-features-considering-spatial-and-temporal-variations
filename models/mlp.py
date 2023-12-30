import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):

    """
    This is the simplest MLP baseline. BatchNorm and Dropout are not implemented.
    The reason for this is to perform ablation studies with complex-valued-MLP.
    """

    def __init__(self, num_features: int, num_classes: int, num_hidden: int):

        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.classifier = nn.Linear(num_hidden, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)

        return x

    def get_loss(self, inputs, labels):

        output = self.forward(inputs)
        loss = self.criterion(output, labels)

        return loss
