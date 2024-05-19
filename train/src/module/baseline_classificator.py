from typing import Any

import lightning as pl
import torch

from src.module.builder import ModuleBuilder


# define the LightningModule
class BaselineClassificator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.build()

    def build(self):
        self.backbone, self.head = ModuleBuilder.build_model(
            self.hparams.backbone_cfg,
            self.hparams.head_cfg,
        )

        self.optimizer, self.scheduler = ModuleBuilder.build_optimizers(
            self.parameters(),
            self.hparams.optimizer_cfg,
            self.hparams.scheduler_cfg)

        self.metrics = torch.nn.ModuleList()
        for cfg in self.hparams.metrics:
            self.metrics.append(ModuleBuilder.build_metric(cfg))

    def forward(self, x: torch.Tensor) -> Any:
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
        x, labels = batch
        outputs = self.forward(x)
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

        for metric in self.metrics:
            metric.update(outputs['ai_output'], labels[1])
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
        return [self.optimizer], [self.scheduler]
