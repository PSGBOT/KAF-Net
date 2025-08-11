import torch
import torch.nn as nn
from collections import OrderedDict


def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)  # 2 for reg and wh
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)  # gather the data on the gt_detected point
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feature(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feature(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = (
            tmp[:, e[1], ...].copy(),
            tmp[:, e[0], ...].copy(),
        )
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2, tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = (
            tmp[:, e[1], ...].copy(),
            tmp[:, e[0], ...].copy(),
        )
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def load_model_for_inference(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        start_epoch = checkpoint.get("epoch", 0)
    else:
        state_dict = checkpoint
        start_epoch = 0

    # Handle DataParallel/DistributedDataParallel wrapper
    if any(k.startswith("module.") for k in state_dict.keys()):
        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Load the state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print(f"Loaded pretrained weights from {checkpoint_path}!")
    return model, start_epoch


def load_model(model, pretrain_dir):
    ckpt = torch.load(pretrain_dir, map_location="cuda:0")
    state_dict_ = ckpt["model_state_dict"]
    print("Loaded pretrained weights from %s!" % pretrain_dir)

    new_state_dict = OrderedDict()
    model_state_dict = model.state_dict()

    # 检测当前模型是否使用 DataParallel（即参数是否带 module.）
    model_is_parallel = list(model_state_dict.keys())[0].startswith("module")
    ckpt_is_parallel = list(state_dict_.keys())[0].startswith("module")

    for k, v in state_dict_.items():
        # 去掉/添加 module. 前缀
        if ckpt_is_parallel and not model_is_parallel:
            k = k[len("module.") :]
        elif not ckpt_is_parallel and model_is_parallel:
            k = "module." + k
        new_state_dict[k] = v

    # 逐参数检查是否 shape 匹配
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            if new_state_dict[k].shape != model_state_dict[k].shape:
                print(
                    f"⚠️ Skip loading parameter {k}: "
                    f"required shape {model_state_dict[k].shape}, "
                    f"loaded shape {new_state_dict[k].shape}."
                )
                del new_state_dict[k]
        else:
            print(f"🗑️ Drop parameter {k}.")

    for k in model_state_dict:
        if k not in new_state_dict:
            print(f"❌ No param {k} in checkpoint.")

    model.load_state_dict(new_state_dict, strict=False)

    return model, ckpt.get("epoch", 0) + 1


def count_parameters(model):
    num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if "aux" not in k]
    print("Total num of param = %f M" % sum(num_paras))


def count_flops(model, input_size=384):
    flops = []
    handles = []

    def conv_hook(self, input, output):
        flops.append(
            output.shape[2] ** 2
            * self.kernel_size[0] ** 2
            * self.in_channels
            * self.out_channels
            / self.groups
            / 1e6
        )

    def fc_hook(self, input, output):
        flops.append(self.in_features * self.out_features / 1e6)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(fc_hook))

    with torch.no_grad():
        _ = model(torch.randn(1, 3, input_size, input_size))
    print("Total FLOPs = %f M" % sum(flops))

    for h in handles:
        h.remove()
