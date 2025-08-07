import cv2
import os
import numpy as np
from PIL import Image
from utils.image import color_aug


def maskname_to_index(maskname="mask0"):
    return int(maskname.split("mask")[1])


def generate_mask_colors(num_masks, max_masks=128):
    """
    Generate distinct RGB colors for different masks.
    Uses HSV color space to ensure good color separation.
    """
    import colorsys

    colors = []

    # Special case for background (index 0) - use black
    colors.append([0.0, 0.0, 0.0])

    # Generate colors for masks using HSV space for better distribution
    for i in range(1, max_masks + 1):
        if i <= num_masks:
            # Use golden ratio to distribute hues evenly
            hue = (i * 0.618033988749895) % 1.0
            saturation = 0.8 + 0.2 * ((i % 3) / 2)  # Vary saturation slightly
            value = 0.8 + 0.2 * (i % 2)  # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        else:
            # For unused mask indices, use black
            rgb = (0.0, 0.0, 0.0)
        colors.append(list(rgb))

    return colors


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


def process_image_and_masks_mcm(sample_path, img_size={"w": 512, "h": 512}):
    src_img_path = os.path.join(sample_path, "src_img.png")
    src_img = Image.open(src_img_path).convert("RGB")
    src_img = src_img.resize((img_size["w"], img_size["h"]), Image.BILINEAR)
    src_img = np.asarray(src_img)
    height, width = src_img.shape[0], src_img.shape[1]

    # Initialize 3 RGB channels for masks
    combined_masks_channels = np.zeros(
        (img_size["h"], img_size["w"], 3), dtype=np.float32
    )

    # Load and process individual masks
    masks_dir = sample_path
    masks_bbox = {}

    # Count total masks first to generate appropriate colors
    mask_files = [f for f in os.listdir(masks_dir) if f.startswith("mask")]
    num_masks = len(mask_files)
    mask_colors = generate_mask_colors(num_masks, 128)

    if os.path.exists(masks_dir):
        # Store individual masks for overlap resolution
        individual_masks = {}

        # First pass: load all masks without combining
        for mask_filename in mask_files:
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

            mask_index = maskname_to_index(os.path.splitext(mask_filename)[0])
            masks_bbox[mask_index] = {
                "center": mask_center,
                "scale": [mask_width, mask_height],
            }

            # Store the processed mask for overlap resolution
            individual_masks[mask_index] = (mask_binary > 0).astype(np.uint8)

        # Second pass: resolve overlaps using boolean subtraction
        # Lower index masks have priority over higher index masks
        resolved_masks = {}
        sorted_mask_indices = sorted(individual_masks.keys())

        for i, mask_idx in enumerate(sorted_mask_indices):
            current_mask = individual_masks[mask_idx].copy()

            # Subtract all previously processed masks (lower indices have priority)
            for j in range(i):
                prev_mask_idx = sorted_mask_indices[j]
                if prev_mask_idx in resolved_masks:
                    # Boolean subtraction: current_mask = current_mask AND NOT prev_mask
                    current_mask = cv2.bitwise_and(
                        current_mask, cv2.bitwise_not(resolved_masks[prev_mask_idx])
                    )

            resolved_masks[mask_idx] = current_mask

            # Update bbox information based on resolved mask
            if np.any(current_mask):
                bbox = cv2.boundingRect(current_mask)
                mask_height = bbox[3]
                mask_width = bbox[2]
                mask_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                masks_bbox[mask_idx] = {
                    "center": mask_center,
                    "scale": [mask_width, mask_height],
                }
            else:
                # If mask is completely removed by subtraction, set empty bbox
                masks_bbox[mask_idx] = {
                    "center": (0, 0),
                    "scale": [0, 0],
                }

        # Third pass: create colored mask channels using resolved masks
        for mask_idx in resolved_masks:
            # Get the color for this mask (ensure mask_index + 1 for proper color mapping)
            color = mask_colors[min(mask_idx + 1, len(mask_colors) - 1)]

            # Create colored mask
            mask_normalized = resolved_masks[mask_idx].astype(np.float32)
            for c in range(3):  # RGB channels
                channel_mask = mask_normalized * color[c]
                # Use maximum to overlay masks (brightest color wins)
                combined_masks_channels[:, :, c] = np.maximum(
                    combined_masks_channels[:, :, c], channel_mask
                )

    # Convert src_img and combined_masks_channel to float and normalize to 0-1
    src_img = src_img.astype(np.float32) / 255.0

    # Concatenate along the channel dimension to create a 6-channel masked_img (3 RGB + 3 mask RGB)
    masked_img = np.concatenate((src_img, combined_masks_channels), axis=2)  # [H, W, 6]

    masked_img = masked_img.transpose(2, 0, 1)  # from [H, W, 6] to [6, H, W]
    return masked_img, masks_bbox
