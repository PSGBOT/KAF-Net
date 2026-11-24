"""Route DSFlash before the legacy entry points import KAF-specific dependencies."""

import argparse
import os
from pathlib import Path
import sys


_DSFLASH_CONFIGS = {
    "dsflash_swin": "dsflash_swin_psr.json",
    "dsflash_swin_functions": "dsflash_swin_psr_functions.json",
}


def dsflash_command(operation, arguments):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--arch")
    args, remaining = parser.parse_known_args(arguments)
    if args.arch is None or not args.arch.startswith("dsflash"):
        return None
    if args.arch not in _DSFLASH_CONFIGS:
        parser.error(f"unknown DSFlash architecture {args.arch!r}; choose {', '.join(_DSFLASH_CONFIGS)}")
    if args.arch == "dsflash_swin_functions":
        remaining.append("--require-functions")
    if operation == "train" and not any(arg == "--config" or arg.startswith("--config=") for arg in remaining):
        config = Path(__file__).resolve().parent.parent / "experiments" / _DSFLASH_CONFIGS[args.arch]
        remaining.extend(("--config", str(config)))
    return ["-m", "experiments.dsflash_psr", operation, *remaining]


def route_model(operation):
    command = dsflash_command(operation, sys.argv[1:])
    if command is not None:
        # Keep torchrun's rank environment and import DSFlash even from outside the checkout.
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        os.execve(sys.executable, [sys.executable, *command], env)


def kaf_architecture(arch, *, image_size=512, pretrained=True):
    from datasets.psr_labels import PSR_FUNC_CAT, PSR_KR_CAT

    functions, relations = len(PSR_FUNC_CAT), len(PSR_KR_CAT)
    levels = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    if "swin" in arch:
        from nets.kaf.kaf_swint import get_kaf_swint

        model = get_kaf_swint(num_classes=functions, num_rel=relations, image_size=image_size, pretrained=pretrained)
        return model, levels, 6
    if arch.startswith(("resdcn_", "reskaf_")):
        from nets.kaf.kaf_resdcn import get_kaf_resdcn

        model = get_kaf_resdcn(int(arch.split("_")[-1]), num_classes=functions, num_rel=relations, pretrained=pretrained)
        return model, levels, 6
    if "hrnet" in arch:
        from nets.kaf.HRnet import get_kaf_hrnet

        return get_kaf_hrnet(num_classes=functions, num_relations=relations), {"p2": 4}, 4
    if "resnet" in arch:
        from nets.kaf.resnet_pretrain import get_resnet50_fpn

        return get_resnet50_fpn(pretrained=pretrained), {"p2": 4}, 4
    aliases = {
        "large_hourglass": "hourglass_large", "large_hg": "hourglass_large",
        "small_hourglass": "hourglass_small", "small_hg": "hourglass_small",
    }
    if "hourglass" in arch or arch in aliases:
        from nets.kaf.hourglass import get_kaf_hourglass

        return get_kaf_hourglass[aliases.get(arch, arch)], {"p2": 4}, 4
    raise ValueError(f"Unsupported KAF architecture: {arch!r}")
