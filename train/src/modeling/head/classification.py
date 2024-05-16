import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features, dropout, num_model_names: int):
        super().__init__()
        self._num_classes = num_model_names

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, 64)
        self.ReLu = nn.ReLU()
        self.ai_generated_linear = nn.Linear(64, 1)
        self.name_model_linear = nn.Linear(64, num_model_names)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.ReLu(x)
        output = dict(
            ai_output=torch.sigmoid(self.ai_generated_linear(x)).squeeze(1),
            model_name_output=self.name_model_linear(x)
        )
        return output

    def loss(self, outputs, targets):
        model_name_target = targets[0]
        ai_target = targets[1]
        return dict(
            ai_loss=self.bce_loss(outputs['ai_output'], ai_target),
            model_name_loss=self.ce_loss(outputs['model_name_output'], model_name_target),
        )

    def export_onnx(self, outputs):
        return torch.cat([outputs['ai_output'], outputs['model_name_output']], dim=1)
