from typing import Dict, Any


class ModuleBuilder:
    @staticmethod
    def build_model(backbone_cfg: Dict[str, Any], tokenizer_cfg: Dict[str, Any], head_cfg: Dict[str, Any]):
        from src.utils.builders import build_backbone, build_head, build_tokenizer

        backbone = build_backbone(backbone_cfg)
        tokenizer = build_tokenizer(tokenizer_cfg)
        head = build_head(head_cfg)
        return backbone, tokenizer, head

    @staticmethod
    def build_metric(metric_cfg: Dict[str, Any]):
        from src.utils.builders import build_metric
        metric = build_metric(metric_cfg)
        return metric
