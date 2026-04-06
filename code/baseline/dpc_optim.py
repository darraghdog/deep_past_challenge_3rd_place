import bitsandbytes as bnb
from torch import optim


def get_optimizer_grouped_parameters(cfg, model, print_fn=print):
    param_dict = {name: param for name, param in model.named_parameters()}
    param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

    # param shape based groupings --
    params_dict_encoder = {name: param for name, param in param_dict.items() if "encoder." in name}
    params_dict_decoder = {name: param for name, param in param_dict.items() if "decoder." in name}
    params_dict_remaining = {name: param for name, param in param_dict.items() if not any(x in name for x in ["encoder", "decoder"])}

    # decay / no-decay groupings --
    params_dict_encoder_no_decay = {name: param for name, param in params_dict_encoder.items() if len(param.shape) == 1}
    params_dict_encoder_decay = {name: param for name, param in params_dict_encoder.items() if len(param.shape) != 1}
    params_dict_decoder_no_decay = {name: param for name, param in params_dict_decoder.items() if len(param.shape) == 1}
    params_dict_decoder_decay = {name: param for name, param in params_dict_decoder.items() if len(param.shape) != 1}
    params_dict_remaining_no_decay = {name: param for name, param in params_dict_remaining.items() if len(param.shape) == 1}
    params_dict_remaining_decay = {name: param for name, param in params_dict_remaining.items() if len(param.shape) != 1}

    # info ---
    def print_param_group_info(group, group_name):
        n_params = round(sum(p.numel() for p in group.values()) / 1e6, 2)
        print_fn(f"{group_name}: # params: {n_params}M | Sample keys: {list(group.keys())[:2]}")

    # print info for each parameter group
    print_param_group_info(params_dict_encoder_no_decay, "Optimizer (encoder_no_decay)")
    print_param_group_info(params_dict_encoder_decay, "Optimizer (encoder_decay)")
    print_param_group_info(params_dict_decoder_no_decay, "Optimizer (decoder_no_decay)")
    print_param_group_info(params_dict_decoder_decay, "Optimizer (decoder_decay)")
    print_param_group_info(params_dict_remaining_no_decay, "Optimizer (remaining_no_decay)")
    print_param_group_info(params_dict_remaining_decay, "Optimizer (remaining_decay)")

    # create optimizer groups ---
    wd = cfg.optimizer.weight_decay
    optim_groups = [
        {"params": list(params_dict_encoder_no_decay.values()), "lr": cfg.optimizer.lr, "weight_decay": 0.0},
        {"params": list(params_dict_encoder_decay.values()), "lr": cfg.optimizer.lr_encoder, "weight_decay": wd},
        {"params": list(params_dict_decoder_no_decay.values()), "lr": cfg.optimizer.lr, "weight_decay": 0.0},
        {"params": list(params_dict_decoder_decay.values()), "lr": cfg.optimizer.lr_decoder, "weight_decay": wd},
        {"params": list(params_dict_remaining_no_decay.values()), "lr": cfg.optimizer.lr, "weight_decay": 0.0},
        {"params": list(params_dict_remaining_decay.values()), "lr": cfg.optimizer.lr, "weight_decay": 0.0},
    ]

    # filter out groups with no params
    optim_groups = [group for group in optim_groups if len(group["params"]) > 0]
    return optim_groups


def get_optimizer(cfg, model, print_fn=print):
    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "AdamW8bit": bnb.optim.Adam8bit,
        # "Adafactor": optim.Adafactor,
    }
    assert cfg.optimizer.name in _optimizers, f"Optimizer {cfg.optimizer.name} not supported"

    optim_groups = get_optimizer_grouped_parameters(cfg, model, print_fn)

    optimizer = _optimizers[cfg.optimizer.name](
        optim_groups,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.adam_beta_1, cfg.optimizer.adam_beta_2),
        eps=cfg.optimizer.adam_epsilon,
    )

    return optimizer
