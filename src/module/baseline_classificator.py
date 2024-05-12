from typing import Any

import torch
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
        for cfg in self.hparams.metrics:
            self.metrics.append(ModuleBuilder.build_metric(cfg))

    def forward(self, x: torch.Tensor) -> Any:
        x = self.tokenizer.tokenize(x)
        x = x.to(self.device)
        x = self.backbone(x)
        x = self.head(x)
        x = x.squeeze(1)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        texts, labels = batch
        preds = self.forward(texts)
        loss = self.head.loss(preds, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        preds = self.forward(texts)
        loss = self.head.loss(preds, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("val/loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric.update(preds, labels)
        return loss

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
        # print(batch)
        # texts, labels = batch
        preds = self.forward(texts)
        return preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
