import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, num_hidden: int, cfg: DictConfig):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=cfg.preprocessing.conv1_in_channels,
            out_channels=cfg.models.conv1_out_channels,
            kernel_size=cfg.models.conv1_kernel_size,
        )
        self.conv2 = nn.Conv2d(
            in_channels=cfg.models.conv1_out_channels,
            out_channels=cfg.models.conv2_out_channels,
            kernel_size=cfg.models.conv2_kernel_size,
        )
        self.conv3 = nn.Conv2d(
            in_channels=cfg.models.conv2_out_channels,
            out_channels=cfg.models.conv3_out_channels,
            kernel_size=cfg.models.conv3_kernel_size,
        )

        self.fc1 = nn.Linear(cfg.preprocessing.linear_in, cfg.models.num_hidden)
        self.fc2 = nn.Linear(cfg.models.num_hidden, cfg.models.num_hidden)
        self.classifier = nn.Linear(cfg.models.num_hidden, num_classes)

        self.criterion = nn.NLLLoss()
        self.activation = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(cfg.models.conv1_out_channels)
        self.bn2 = nn.BatchNorm2d(cfg.models.conv2_out_channels)
        self.bn3 = nn.BatchNorm2d(cfg.models.conv3_out_channels)
        self.bn4 = nn.BatchNorm1d(cfg.models.num_hidden)
        self.bn5 = nn.BatchNorm1d(cfg.models.num_hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.pool1(self.activation(self.bn1(self.conv1(inputs))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        h = self.classifier(x)
        x = F.log_softmax(h, dim=1)

        return x

    def get_loss(self, inputs, labels):

        output = self.forward(inputs)
        loss = self.criterion(output, labels)

        return loss
