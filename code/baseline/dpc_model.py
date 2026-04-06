from transformers import T5ForConditionalGeneration


def get_dpc_model(cfg):
    return T5ForConditionalGeneration.from_pretrained(cfg.model.backbone_path)
