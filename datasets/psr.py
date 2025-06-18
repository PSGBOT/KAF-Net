from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
import cv2
from utils.image import get_border, get_affine_transform, color_aug


PSR_FUNC_CAT = [
    "handle",
    "housing",
    "support",
    "frame",
    "button",
    "wheel",
    "display",
    "cover",
    "plug",
    "port",
    "door",
    "container",
    "other",
]
PSR_FUNC_CAT_IDX = {v: i for i, v in enumerate(PSR_FUNC_CAT)}

PSR_KR_CAT = [
    "fixed",
    "revolute-free",
    "revolute-controlled",
    "revolute-static",
    "prismatic-free",
    "prismatic-controlled",
    "prismatic-static",
    "spherical-free",
    "spherical-controlled",
    "spherical-static",
    "supported",
    "flexible",
    "unrelated",
    "unknown",
]

PSR_KR_CAT_IDX = {v: i for i, v in enumerate(PSR_KR_CAT)}

PSR_MEAN = [0.40789654, 0.44719302, 0.47026115]
PSR_STD = [0.28863828, 0.27408164, 0.27809835]
PSR_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
PSR_EIGEN_VECTORS = [
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938],
]


class PSRDataset(Dataset):
    def __init__(self, root_dir, split, split_ratio=1.0, img_size=512):
        self.root_dir = root_dir
        self.samples = []

        self.func_cat = PSR_FUNC_CAT
        self.func_cat_ids = PSR_FUNC_CAT_IDX
        self.kr_cat = PSR_KR_CAT
        self.kr_cat_ids = PSR_KR_CAT_IDX

        self.num_func_cat = len(self.func_cat)
        self.num_kr_cat = len(self.kr_cat)

        self.data_rng = np.random.RandomState(123)
        self.mean = np.array(PSR_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(PSR_STD, dtype=np.float32)[None, None, :]
        self.eig_val = np.array(PSR_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(PSR_EIGEN_VECTORS, dtype=np.float32)

        self.split = split
        self.split_ratio = split_ratio

        self.padding = 127  # 31 for resnet/resdcn
        self.img_size = {"h": img_size, "w": img_size}
        self.rand_scales = np.arange(1, 1.4, 0.1)

        for sample_name in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_name)
            if os.path.isdir(sample_path):
                self.samples.append(sample_path)

        if 0 < split_ratio < 1:
            split_size = int(
                np.clip(split_ratio * len(self.images), 1, len(self.images))
            )
            self.images = self.images[:split_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        # Load config.json
        config_path = os.path.join(sample_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load src_img.png and masks, set the border of the masks to be 128(gray), combine these masks together into the 4th channel of the source image => `masked_img`
        src_img_path = os.path.join(sample_path, "src_img.png")
        src_img = Image.open(src_img_path).convert("RGB")
        src_img = np.asarray(src_img)
        height, width = src_img.shape[0], src_img.shape[1]
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0

        # Data augmentation parameters (determined once for both image and masks)
        flipped = False
        if self.split == "train":
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True

        # Affine transform matrix (determined once for both image and masks)
        trans_img = get_affine_transform(
            center, scale, 0, [self.img_size["w"], self.img_size["h"]]
        )

        # Apply flipping to src_img
        if flipped:
            src_img = src_img[:, ::-1, :]

        # Apply affine transform to src_img
        src_img = cv2.warpAffine(
            src_img,
            trans_img,
            (self.img_size["w"], self.img_size["h"]),
            flags=cv2.INTER_LINEAR,
        )

        # Initialize an empty mask channel
        combined_masks_channel = np.zeros(
            (self.img_size["h"], self.img_size["w"]), dtype=np.uint8
        )

        # Load and process individual masks
        masks_dir = sample_path
        origin_mask_width = 640
        origin_mask_height = 480
        if os.path.exists(masks_dir):
            all_contours = []
            for mask_filename in os.listdir(masks_dir):
                if mask_filename.startswith("mask"):  # Assuming masks are PNGs
                    mask_path = os.path.join(masks_dir, mask_filename)
                    mask = Image.open(mask_path).convert("L")  # Load as grayscale
                    origin_mask_height = np.asarray(mask).shape[0]
                    origin_mask_width = np.asarray(mask).shape[1]
                    # Resize mask to original image size before any transformations
                    mask = mask.resize((width, height), Image.BILINEAR)
                    mask = np.asarray(mask)

                    # Apply the same flipping as src_img
                    if flipped:
                        mask = mask[:, ::-1]

                    # Apply the same affine transform as src_img
                    mask = cv2.warpAffine(
                        mask,
                        trans_img,
                        (self.img_size["w"], self.img_size["h"]),
                        flags=cv2.INTER_LINEAR,
                    )

                    # Now, process the mask to set border to 128 and inside to 255
                    # Ensure mask is binary (0 or 255) for findContours
                    mask_binary = (mask > 0).astype(np.uint8) * 255

                    erode_mask = cv2.erode(mask, None, iterations=2)
                    contours, _ = cv2.findContours(
                        erode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    all_contours.extend(contours)

                    # Combine with overall combined_masks_channel
                    combined_masks_channel = np.maximum(
                        combined_masks_channel, mask_binary
                    )
                    # DEBUG: show the combined_masks here
                    cv2.imwrite(
                        "debug_outputs/combined_masks_channel.png",
                        combined_masks_channel * 255,
                    )
                cv2.drawContours(combined_masks_channel, all_contours, -1, (128,), 5)

        # Convert src_img and combined_masks_channel to float and normalize to 0-1
        src_img = src_img.astype(np.float32) / 255.0
        combined_masks_channel = combined_masks_channel.astype(np.float32) / 255.0

        # Concatenate along the channel dimension to create a 4-channel masked_img
        masked_img = np.concatenate(
            (src_img, combined_masks_channel[:, :, np.newaxis]), axis=2
        )  # [H, W, C+1]

        # Color augmentation (applied only to the first 3 RGB channels of masked_img)
        if self.split == "train":
            color_aug(self.data_rng, masked_img[:, :, :3], self.eig_val, self.eig_vec)

        # Normalize and transpose (applied to all channels)
        # Extend mean and std for the 4th channel (mask channel has 0 mean and 1 std)
        extended_mean = np.concatenate(
            (self.mean, np.array([0.0], dtype=np.float32)[None, None, :]), axis=2
        )
        extended_std = np.concatenate(
            (self.std, np.array([1.0], dtype=np.float32)[None, None, :]), axis=2
        )

        masked_img -= extended_mean
        masked_img /= extended_std
        masked_img = masked_img.transpose(2, 0, 1)  # from [H, W, C+1] to [C+1, H, W]

        # Transform part_center coordinates
        part_centers = {}
        for key in config["part center"]:
            part_center = np.array(config["part center"][key], dtype=np.float32)
            part_center[0] *= width / origin_mask_width
            part_center[1] *= height / origin_mask_height
            if flipped:
                part_center[0] = width - part_center[0] - 1
            part_center = cv2.transform(part_center[None, None, :], trans_img)[0, 0, :]
            part_centers[key] = part_center

        return {
            "masked_img": masked_img,
            "part_center": part_centers,
            "kinematic_relation": config["kinematic relation"],
        }
