import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features, 64)
        self.ReLu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.ReLu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)

    def loss(self, output, target):
        return self.bce_loss(output, target)
