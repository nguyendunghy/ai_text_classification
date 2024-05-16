from typing import Any

import torch
import torchmetrics
import lightning as pl
from torch import optim, nn

from src.module.builder import ModuleBuilder


# define the LightningModule
class BaselineClassificator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.build()

    def build(self):
        self.backbone, self.tokenizer, self.head = ModuleBuilder.build_model(self.hparams.backbone_cfg,
                                                                             self.hparams.tokenizer_cfg,
                                                                             self.hparams.head_cfg)

        self.metrics = torch.nn.ModuleList()

        self.acc_binary = torchmetrics.Accuracy(task='binary')
        self.acc_multiclass = torchmetrics.Accuracy(task='multiclass', num_classes=self.head._num_classes)

        self.metrics.append(self.acc_binary)
        self.metrics.append(self.acc_multiclass)
        # for cfg in self.hparams.metrics:
        #     self.metrics.append(ModuleBuilder.build_metric(cfg))

    def forward(self, x: torch.Tensor) -> Any:
        x = self.tokenizer.tokenize(x)
        x = x.to(self.device)
        x = self.backbone(x)
        x = self.head(x)
        return x

    def forward_postprocess(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            outputs = self.head(x)
            return outputs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        texts, labels = batch
        outputs = self.forward(texts)
        losses = self.head.loss(outputs, labels)

        # Logging to TensorBoard (if installed) by default
        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=True)
        return sum(losses.values())

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self.forward(texts)
        losses = self.head.loss(outputs, labels)
        # Logging to TensorBoard (if installed) by default
        for name, value in losses.items():
            self.log(f"val/{name}", value, prog_bar=True)

        self.acc_multiclass.update(outputs['model_name_output'], labels[0])
        self.acc_binary.update(outputs['ai_output'], labels[1])
        return sum(losses.values())

    def on_validation_epoch_end(self):
        for metric in self.metrics:
            value = metric.compute()
            metric.reset()
            self.log(metric.__class__.__name__, value, sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, texts, batch_idx) -> Any:
        return self.forward(texts)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer
