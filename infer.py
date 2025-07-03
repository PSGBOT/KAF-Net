import argparse
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from nets.kaf import resdcn
from utils.image import get_affine_transform, affine_transform, transform_preds


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
        "--num_classes", type=int, default=80, help="Number of object classes."
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
    args = parser.parse_args()
    return args


# Main inference logic
def main():
    args = parse_args()

    # Load the model
    if args.arch == "resdcn_50":
        model = resdcn.get_kaf_resdcn(
            num_layers=50,
            head_conv=args.head_conv,
            num_classes=args.num_classes,
            num_rel=args.num_relations,
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    # Load weights
    model.load_state_dict(torch.load(args.model_weights, map_location="cpu"))
    model.eval()  # Set model to evaluation mode

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Image preprocessing parameters (assuming a fixed input size for the model)
    input_res = 512  # Example resolution, adjust if your model expects a different size
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

    # Load and preprocess the image
    image = Image.open(args.image_path).convert("RGB")
    width, height = image.size
    new_width, new_height = input_res, input_res
    center = np.array([width // 2, height // 2], dtype=np.float32)
    scale = max(height, width) * 1.0

    trans_input = get_affine_transform(center, scale, 0, [new_width, new_height])
    inp_image = image.transform(
        (new_width, new_height),
        Image.AFFINE,
        (
            trans_input[0, 0],
            trans_input[0, 1],
            trans_input[0, 2],
            trans_input[1, 0],
            trans_input[1, 1],
            trans_input[1, 2],
        ),
        Image.BILINEAR,
    )
    inp_image = (np.array(inp_image) / 255.0 - mean) / std
    inp_image = inp_image.transpose(2, 0, 1).astype(np.float32)
    inp_image = torch.from_numpy(inp_image).unsqueeze(0).to(device)

    # Store original image dimensions for post-processing
    meta = {
        "height": height,
        "width": width,
        "center": center,
        "scale": scale,
        "trans_input": trans_input,
    }

    # Perform inference
    with torch.no_grad():
        outputs = model(inp_image)
        # The model output is a list containing one element, which is another list of tensors.
        # The inner list contains [hmap, regs, w_h_, raf]
        hmap, regs, w_h_, raf = outputs[0]

    # Post-processing parameters
    down_ratio_hmap = 32
    down_ratio_reg = 16
    down_ratio_wh = 8
    down_ratio_raf = 4
    max_objects = 100  # Maximum number of objects to detect
    topk_relations = 50  # Top K relations to consider

    # print(f"Inference complete. Results saved to {args.output_json_path}")


if __name__ == "__main__":
    main()
