import cv2
import os
import numpy as np
from PIL import Image
from utils.image import color_aug


def maskname_to_index(maskname="mask0"):
    return int(maskname.split("mask")[1])


def process_image_and_masks(sample_path, img_size={"w": 512, "h": 512}):
    src_img_path = os.path.join(sample_path, "src_img.png")
    src_img = Image.open(src_img_path).convert("RGB")
    src_img = src_img.resize((img_size["w"], img_size["h"]), Image.BILINEAR)
    src_img = np.asarray(src_img)
    height, width = src_img.shape[0], src_img.shape[1]
    # center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    # scale = max(height, width) * 1.0
    # Apply flipping to src_img
    # Initialize an empty mask channel
    combined_masks_channel = np.zeros((img_size["h"], img_size["w"]), dtype=np.uint8)

    # Load and process individual masks
    masks_dir = sample_path
    masks_bbox = {}
    if os.path.exists(masks_dir):
        all_contours = []
        for mask_filename in os.listdir(masks_dir):
            if mask_filename.startswith(
                "mask"
            ):  # Assuming masks: "mask0.png" "mask1.png"...
                mask_path = os.path.join(masks_dir, mask_filename)
                mask = Image.open(mask_path).convert("L")  # Load as grayscale
                # Resize mask to original image size before any transformations
                mask = mask.resize((width, height), Image.BILINEAR)
                mask = np.asarray(mask)
                # Now, process the mask to set border to 128 and inside to 255
                # Ensure mask is binary (0 or 255) for findContours
                mask_binary = (mask > 0).astype(np.uint8) * 255

                # find bbox scale of the mask
                bbox = cv2.boundingRect(mask_binary)
                # get the height and width
                mask_height = bbox[3]
                mask_width = bbox[2]
                # get the center
                mask_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                masks_bbox[maskname_to_index(os.path.splitext(mask_filename)[0])] = {
                    "center": mask_center,
                    "scale": [mask_width, mask_height],
                }

                kernel = np.ones((3, 3), np.uint8)
                erode_mask = cv2.erode(mask, kernel, iterations=1)
                contours, _ = cv2.findContours(
                    erode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                all_contours.extend(contours)

                # Combine with overall combined_masks_channel
                combined_masks_channel = np.maximum(combined_masks_channel, mask_binary)
            cv2.drawContours(combined_masks_channel, all_contours, -1, (128,), 3)

    # Convert src_img and combined_masks_channel to float and normalize to 0-1
    src_img = src_img.astype(np.float32) / 255.0
    combined_masks_channel = combined_masks_channel.astype(np.float32) / 255.0

    # Concatenate along the channel dimension to create a 4-channel masked_img
    masked_img = np.concatenate(
        (src_img, combined_masks_channel[:, :, np.newaxis]), axis=2
    )  # [H, W, C+1]
    masked_img = masked_img.transpose(2, 0, 1)  # from [H, W, C+1] to [C+1, H, W]
    return masked_img, masks_bbox
