from typing import Dict, Any

import torchmetrics as PLMetrics
import lightning.pytorch.callbacks as Callbacks
import torch.utils.data as TorchDatasets
from torch import optim as Optimizers
from transformers import get_linear_schedule_with_warmup

import src.data as DataModules
import src.data.datasets as Datasets
import src.data.trasnforms as Trasnforms
import src.modeling.backbone as Backbones
import src.modeling.tokenizer as Tokenizers
import src.modeling.head as Heads
import src.module as MainModules


def _base_build(config, modules_to_find):
    if config is None:
        return None

    assert isinstance(config, dict) and 'type' in config, f'Check config type validity: {config}'

    args = config.copy()
    obj_type_name = args.pop('type')

    real_type = None
    for module in modules_to_find:
        if not hasattr(module, obj_type_name):
            continue
        real_type = getattr(module, obj_type_name)
        if real_type:
            break

    assert real_type is not None, f'{obj_type_name} is not registered type in any modules: {modules_to_find}'
    return real_type(**args)


def build_backbone(cfg: Dict[str, Any]):
    return _base_build(cfg, [Backbones])


def build_head(cfg: Dict[str, Any]):
    return _base_build(cfg, [Heads])


def build_tokenizer(cfg: Dict[str, Any]):
    return _base_build(cfg, [Tokenizers])


def build_module(cfg: Dict[str, Any]):
    return _base_build(cfg, [MainModules, DataModules])


def build_metric(cfg: Dict[str, Any]):
    return _base_build(cfg, [PLMetrics])


def build_dataset(tokenizer, cfg: Dict[str, Any]):
    if cfg.get('type') == 'ConcatDataset':
        datasets = []
        for dataset_cfg in cfg.get('datasets'):
            dataset_cfg['tokenizer'] = tokenizer
            datasets.append(build_dataset(tokenizer, dataset_cfg))
        cfg['datasets'] = datasets
        return _base_build(cfg, [TorchDatasets])
    cfg['tokenizer'] = tokenizer
    return _base_build(cfg, [Datasets])


def build_transform(cfg: Dict[str, Any]):
    return _base_build(cfg, [Trasnforms])


def build_optimizer(params, cfg: Dict[str, Any]):
    cfg['params'] = params
    return _base_build(cfg, [Optimizers])


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    return get_linear_schedule_with_warmup(optimizer=optimizer, **cfg)


def build_callbacks(cfg: Dict[str, Any]):
    return _base_build(cfg, [Callbacks])
