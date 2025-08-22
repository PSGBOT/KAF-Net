import argparse
import json
from multiprocessing import process
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from utils.utils import load_model_for_inference

from nets.kaf import kaf_resdcn as reskaf
from nets.kaf.kaf_swint import get_kaf_swint
from nets.kaf import hourglass
from datasets.utils.augimg import process_image_and_masks, process_image_and_masks_mcm
from get_scene import get_scene_graph
from datasets.psr import PSR_FUNC_CAT, PSR_KR_CAT


def parse_args():
    parser = argparse.ArgumentParser(description="KAF-Net Inference")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to the pre-trained model weights (.pth file).",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default="output.json",
        help="Path to save the output JSON file.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resdcn_50",
        help="Model architecture. Currently only resdcn_50 is supported.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=13, help="Number of object classes."
    )
    parser.add_argument(
        "--num_relations", type=int, default=14, help="Number of relationship types."
    )
    parser.add_argument(
        "--head_conv",
        type=int,
        default=64,
        help="Number of channels for the head convolution.",
    )
    parser.add_argument(
        "--visualize_output",
        action="store_true",
        help="If set, visualize and save heatmap and RAF outputs.",
    )
    parser.add_argument(
        "--visualization_dir",
        type=str,
        default="visualization_output",
        help="Directory to save visualization outputs.",
    )
    args = parser.parse_args()
    return args


# Main inference logic
def main():
    args = parse_args()

    # Load the model
    if args.arch == "reskaf_50":
        model = reskaf.get_kaf_resdcn(
            num_layers=50,
        )
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    elif args.arch == "reskaf_18":
        model = reskaf.get_kaf_resdcn(
            num_layers=18,
        )
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    elif args.arch == "reskaf_101":
        model = reskaf.get_kaf_resdcn(
            num_layers=101,
        )
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    elif "swin" in args.arch:
        model = get_kaf_swint(
            head_conv=64,
        )
        down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    elif args.arch == "small_hg":
        model = hourglass.get_kaf_hourglass["hourglass_small"]
        down_ratio = {"p2": 4}
    elif args.arch == "large_hg":
        model = hourglass.get_kaf_hourglass["hourglass_large"]
        down_ratio = {"p2": 4}
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    # Load weights
    model, _ = load_model_for_inference(model, args.model_weights)
    model.eval()  # Set model to evaluation mode

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load and preprocess the image
    inp_image, bbox = process_image_and_masks_mcm(args.image_path)
    inp_image = torch.from_numpy(inp_image).unsqueeze(0).to(device)
    print(inp_image.shape)
    image_name = os.path.basename(args.image_path)
    output_dir = os.path.join(
        args.visualization_dir, f"{os.path.splitext(image_name)[0]}"
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(
        args.image_path, os.path.join(output_dir, image_name), dirs_exist_ok=True
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(inp_image)
        # The model output is a list containing one element, which is another list of tensors.
        # The inner list contains [hmap, regs, w_h_, raf]
        hmaps, regs, w_h_s, rafs = outputs

        for fpn_level in range(len(down_ratio)):
            hmap = hmaps[fpn_level]
            reg = regs[fpn_level]
            w_h_ = w_h_s[fpn_level]
            raf = rafs[fpn_level]

            # Visualize outputs if requested
            if args.visualize_output:
                visualize_heatmap(hmap, output_dir, image_name, fpn_level)
                visualize_raf(
                    raf,
                    output_dir,
                    image_name,
                    args.num_relations,
                    fpn_level,
                )

        # Post-processing parameters
        max_objects = 100  # Maximum number of objects to detect
        topk_relations = 50  # Top K relations to consider
        get_scene_graph(
            hmaps,
            regs,
            w_h_s,
            rafs,
            bbox,
            topk_relations,
            inp_image=inp_image,
        )

    # print(f"Inference complete. Results saved to {args.output_json_path}")


def visualize_heatmap(hmap, output_dir, image_name, fpn_level=0):
    """
    Visualizes and saves the heatmap.
    hmap: torch.Tensor of shape [1, num_classes, H, W]
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert heatmap to numpy and remove batch dimension
    hmap = torch.sigmoid(hmap)
    hmap_np = hmap.squeeze(0).cpu().numpy()  # Shape: [num_classes, H, W]

    num_classes = hmap_np.shape[0]

    max_heat = 0
    min_heat = 0
    for i in range(num_classes):
        max_heat = max(max_heat, hmap_np[i].max())

    for i in range(num_classes):
        heatmap = hmap_np[i]
        # Normalize to 0-255 for visualization
        print(
            f"level {fpn_level} class {PSR_FUNC_CAT[i]} {heatmap.max()} {heatmap.min()}"
        )
        heatmap = (heatmap - min_heat) / (max_heat - min_heat + 1e-8) * 255
        heatmap = heatmap.astype(np.uint8)

        # Apply a colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Save the heatmap
        output_path = os.path.join(
            output_dir,
            f"level_{fpn_level}_heatmap_class_{PSR_FUNC_CAT[i]}.png",
        )
        cv2.imwrite(output_path, heatmap_colored)
        # print(f"Saved heatmap for class {PSR_FUNC_CAT[i]} to {output_path}")


def visualize_raf(raf, output_dir, image_name, num_relations, fpn_level=0):
    """
    Visualizes and saves the Relation Affinity Field (RAF).
    raf: torch.Tensor of shape [1, num_relations * 2, H, W]
    """
    os.makedirs(output_dir, exist_ok=True)

    raf_np = raf.squeeze(0).cpu().numpy()  # Shape: [num_relations * 2, H, W]

    # RAF has 2 channels per relation (dx, dy)
    # We can visualize the magnitude or direction, or individual channels
    # For simplicity, let's visualize the magnitude of the vector field for each relation type

    # get max magnitude
    max_mag = 0
    for i in range(num_relations):
        dx = raf_np[i * 2]
        dy = raf_np[i * 2 + 1]
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude.max() > max_mag:
            max_mag = magnitude.max()
    # print(f"max mag: {max_mag}")

    for i in range(num_relations):
        # Extract dx and dy channels for the current relation
        dx = raf_np[i * 2]
        dy = raf_np[i * 2 + 1]

        # Calculate angle (direction) from dx and dy
        # atan2 returns values in radians from -pi to pi.
        # We need to convert it to degrees (0 to 360) and then map to 0-179 for OpenCV's HSV H channel.
        angle = np.arctan2(dy, dx)  # Radians from -pi to pi
        angle_degrees = (np.degrees(angle) + 180) % 360  # Convert to 0-360 degrees

        # Normalize angle to 0-179 for OpenCV's H channel (Hue: 0-179, Saturation: 0-255, Value: 0-255)
        hue = (angle_degrees / 2).astype(np.uint8)

        # Create HSV image: Hue for direction, Saturation and Value for intensity/magnitude
        # Create HSV image for base color: Hue for direction, full Saturation, full Value
        saturation = np.full_like(hue, 255)  # Full saturation
        base_value = np.full_like(hue, 255)  # Full brightness for the base color

        hsv_base_color = cv2.merge([hue, saturation, base_value])
        bgr_base_color = cv2.cvtColor(hsv_base_color, cv2.COLOR_HSV2BGR)

        # Calculate magnitude and normalize for Alpha channel
        magnitude = np.sqrt(dx**2 + dy**2)
        # Normalize magnitude to 0-255 for the alpha channel
        alpha = (magnitude - 0) / (max_mag - 0 + 1e-8) * 255
        alpha = alpha.astype(np.uint8)

        # Merge BGR channels with the alpha channel
        bgra_image = cv2.merge(
            [
                bgr_base_color[:, :, 0],
                bgr_base_color[:, :, 1],
                bgr_base_color[:, :, 2],
                alpha,
            ]
        )

        # Save the direction visualization
        output_path = os.path.join(
            output_dir,
            f"level_{fpn_level}_raf_{i}_{PSR_KR_CAT[i]}_direction.png",
        )
        cv2.imwrite(output_path, bgra_image)
        # print(f"Saved RAF direction for relation {PSR_KR_CAT[i]} to {output_path}")


if __name__ == "__main__":
    main()
