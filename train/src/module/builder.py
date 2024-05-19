from typing import Dict, Any


class ModuleBuilder:
    @staticmethod
    def build_model(backbone_cfg: Dict[str, Any], head_cfg: Dict[str, Any]):
        from src.utils.builders import build_backbone, build_head

        backbone = build_backbone(backbone_cfg)
        head = build_head(head_cfg)
        return backbone, head

    @staticmethod
    def build_optimizers(params, optimizer_cfg: Dict[str, Any], scheduler_cfg: Dict[str, Any]):
        from src.utils.builders import build_optimizer, build_scheduler

        optimizer = build_optimizer(params, optimizer_cfg)
        scheduler = build_scheduler(optimizer, scheduler_cfg)
        return optimizer, scheduler

    @staticmethod
    def build_metric(metric_cfg: Dict[str, Any]):
        from src.utils.builders import build_metric

        metric = build_metric(metric_cfg)
        return metric
