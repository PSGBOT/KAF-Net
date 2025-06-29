from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import json
import os
from PIL import Image
import numpy as np
import cv2
import math
from utils.image import get_affine_transform, color_aug
from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius

PSR_FUNC_CAT = [
    "other",
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
]
PSR_FUNC_CAT_IDX = {v: i for i, v in enumerate(PSR_FUNC_CAT)}

PSR_KR_CAT = [
    "unknown",
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
        print("==> Initializing PSR Dataset")
        self.root_dir = root_dir
        self.samples = []

        self.func_cat = PSR_FUNC_CAT
        self.func_cat_ids = PSR_FUNC_CAT_IDX
        self.kr_cat = PSR_KR_CAT
        self.kr_cat_ids = PSR_KR_CAT_IDX

        self.num_func_cat = len(self.func_cat)
        self.num_kr_cat = len(self.kr_cat)
        self.max_objs = 128

        self.data_rng = np.random.RandomState(123)
        self.mean = np.array(PSR_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(PSR_STD, dtype=np.float32)[None, None, :]
        self.eig_val = np.array(PSR_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(PSR_EIGEN_VECTORS, dtype=np.float32)

        self.split = split
        self.split_ratio = split_ratio

        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {"h": img_size, "w": img_size}
        self.fmap_size = {
            "h": img_size // self.down_ratio,
            "w": img_size // self.down_ratio,
        }
        self.gaussian_iou = 0.7
        self.rand_scales = np.arange(1, 1.4, 0.1)
        for sample_name in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_name)
            if os.path.isdir(sample_path):
                self.samples.append(sample_path)

        if 0 < split_ratio < 1:
            split_size = int(
                np.clip(split_ratio * len(self.samples), 1, len(self.samples))
            )
            self.samples = self.samples[:split_size]

        if split == "train":
            print(f"load {len(self.samples)} samples for training")
        elif split == "test":
            print(f"load {len(self.samples)} samples for testing")

    def __len__(self):
        return len(self.samples)

    def _parse_relations(self, relations):
        # relations is expected to be a dictionary, e.g., {'type': 'revolute-free', 'root_index': 0}
        joint_type = relations.get("joint_type", "unknown")
        controllable = relations.get("controllable")

        if joint_type in ["fixed", "unrelated", "supported", "flexible"]:
            relation_type = joint_type
        elif controllable:
            relation_type = f"{joint_type}-{controllable}"
        else:
            relation_type = joint_type  # Fallback, though should be covered by above

        relation_idx = self.kr_cat_ids.get(relation_type, 0)
        root = int(relations.get("root", 0))  # Convert root to integer
        return relation_idx, root

    def _process_image_and_masks(self, sample_path, src_img, flipped, trans_img):
        height, width = src_img.shape[0], src_img.shape[1]
        # center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        # scale = max(height, width) * 1.0
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

                    # find bbox scale of the mask
                    bbox = cv2.boundingRect(mask_binary)
                    # get the height and width
                    mask_height = bbox[3]
                    mask_width = bbox[2]
                    # get the center
                    mask_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                    masks_bbox[os.path.splitext(mask_filename)[0]] = {
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
                    combined_masks_channel = np.maximum(
                        combined_masks_channel, mask_binary
                    )
                cv2.drawContours(combined_masks_channel, all_contours, -1, (128,), 3)

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
        return masked_img, masks_bbox

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
            # w_border = get_border(128, width)
            # h_border = get_border(128, height)
            # center[0] = np.random.randint(low=w_border, high=width - w_border)
            # center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True

        # Affine transform matrix (determined once for both image and masks)
        trans_img = get_affine_transform(
            center, scale, 0, [self.img_size["w"], self.img_size["h"]]
        )

        masked_img, masks_bbox = self._process_image_and_masks(
            sample_path,
            src_img.copy(),
            flipped,
            trans_img,
        )

        masks_cat = {}
        kr = []

        # Initialize all the masks_cat
        for mask_idx in range(len(config["part center"])):
            masks_cat[f"mask{mask_idx}"] = {}
            for cat_idx in range(self.num_func_cat):
                masks_cat[f"mask{mask_idx}"][cat_idx] = 0

        # assign category index to each mask
        for pair_relation in config["kinematic relation"]:
            part0_cat_str_list = pair_relation[2]["part0_function"]
            for cat_str in part0_cat_str_list:
                part0_cat_idx = self.func_cat_ids.get(cat_str, 0)
                if part0_cat_idx not in masks_cat[pair_relation[0]]:
                    masks_cat[pair_relation[0]][part0_cat_idx] = 1
                else:
                    masks_cat[pair_relation[0]][part0_cat_idx] += 1

            part1_cat_str_list = pair_relation[2]["part1_function"]
            for cat_str in part1_cat_str_list:
                part1_cat_idx = self.func_cat_ids.get(cat_str, 0)
                if part1_cat_idx not in masks_cat[pair_relation[1]]:
                    masks_cat[pair_relation[1]][part1_cat_idx] = 1
                else:
                    masks_cat[pair_relation[1]][part1_cat_idx] += 1

            kjs = pair_relation[2]["kinematic_joints"]
            for kj in kjs:
                relation_idx, root = self._parse_relations(kj)
                if root == 0:
                    kr.append([pair_relation[1], pair_relation[0], relation_idx])
                else:
                    kr.append([pair_relation[0], pair_relation[1], relation_idx])

        # get gt for training
        img = masked_img[:3, :, :]  # Extract RGB channels only
        new_center = np.array([0, 0], dtype=np.float32)

        hmap = np.zeros(
            (self.num_func_cat, self.fmap_size["h"], self.fmap_size["w"]),
            dtype=np.float32,
        )  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        obj_idx = 0
        for mask_name, cat_dict in masks_cat.items():
            # Get bbox info for this mask
            bbox_info = masks_bbox[mask_name]
            center = bbox_info["center"]  # [x, y] in image space
            # print(center)
            scale = bbox_info["scale"]  # [w, h] in image space

            # Transform center to feature map space
            center_fmap = [
                center[0] / self.down_ratio,
                float(center[1] / self.down_ratio),
            ]
            center_fmap = np.array(center_fmap, dtype=np.float32)
            center_int = center_fmap.astype(np.int32)
            # print(center_int)

            w, h = scale

            for cat_idx in cat_dict:  # cat_idx is the category index
                if cat_dict[cat_idx] == 0:
                    continue
                # For each category this mask belongs to, draw a heatmap
                radius = max(
                    0,
                    int(
                        gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)
                    ),
                )
                # print(radius)
                draw_umich_gaussian(hmap[cat_idx], center_int, radius, 1)

            if obj_idx < self.max_objs:
                w_h_[obj_idx] = [w, h]
                regs[obj_idx] = center_fmap - center_int
                # print(regs[obj_idx])
                inds[obj_idx] = center_int[1] * self.fmap_size["w"] + center_int[0]
                ind_masks[obj_idx] = 1
                obj_idx += 1
        # TODO: raf:
        raf_field = torch.rand(
            self.num_kr_cat * 2, self.fmap_size["h"], self.fmap_size["w"]
        )
        raf_weights = torch.rand(
            self.num_kr_cat * 2, self.fmap_size["h"], self.fmap_size["w"]
        )

        # concatenate all the masks for batch loading
        for mask_idx in range(len(config["part center"]), self.max_objs):
            masks_cat[f"mask{mask_idx}"] = {}
            for cat_idx in range(self.num_func_cat):
                masks_cat[f"mask{mask_idx}"][cat_idx] = 0

        # concatenate all the masks_bbox for batch loading
        for mask_idx in range(len(config["part center"]), self.max_objs):
            masks_bbox[f"mask{mask_idx}"] = {}
            masks_bbox[f"mask{mask_idx}"]["center"] = [0, 0]
            masks_bbox[f"mask{mask_idx}"]["scale"] = [0, 0]

        return {
            "masked_img": masked_img,
            "masks_cat": masks_cat,
            "masks_bbox": masks_bbox,
            # "kinematic_relation": kr,  # need align
            "hmap": hmap,
            "w_h_": w_h_,
            "regs": regs,
            "inds": inds,
            "ind_masks": ind_masks,
            "raf": {"gt_relations": raf_field, "gt_relations_weights": raf_weights},
        }


"""
masked_img: [4, 512, 512]
three RGB channel and one mask channel

masks_cat structure:
{
  "mask0": {
    cat1: number of shot,
    cat2: number of shot,
    ...
  },
  "mask1": {
    ...
  },
  ...
}

masks_bbox structure:
{
  "mask0": {
    "center": [x, y],
    "scale": [w, h]
  },
  ...
}

kr structure: [
  [part1, part2, relation_idx],
  [part2, part3, relation_idx],
  ...
]
"""


class PSR_eval(PSRDataset):
    def __init__(
        self, root_dir, split, test_scales=(1,), test_flip=False, fix_size=False
    ):
        super(PSR_eval, self).__init__(root_dir, split)
        self.test_flip = test_flip
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        # Load config.json
        config_path = os.path.join(sample_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load src_img.png
        src_img_path = os.path.join(sample_path, "src_img.png")
        image = cv2.imread(src_img_path)
        height, width = image.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            if self.fix_size:
                img_height, img_width = self.img_size["h"], self.img_size["w"]
                center = np.array([new_width / 2.0, new_height / 2.0], dtype=np.float32)
                scaled_size = max(height, width) * 1.0
                scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | self.padding) + 1
                img_width = (new_width | self.padding) + 1
                center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                scaled_size = np.array([img_width, img_height], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            trans_img = get_affine_transform(
                center, scaled_size, 0, [img_width, img_height]
            )
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            # Process masks for evaluation
            combined_masks_channel = np.zeros((img_height, img_width), dtype=np.uint8)
            masks_dir = sample_path
            if os.path.exists(masks_dir):
                for mask_filename in os.listdir(masks_dir):
                    if mask_filename.startswith("mask"):
                        mask_path = os.path.join(masks_dir, mask_filename)
                        mask = Image.open(mask_path).convert("L")
                        mask = mask.resize((width, height), Image.BILINEAR)
                        mask = np.asarray(mask)

                        mask = cv2.resize(mask, (new_width, new_height))
                        mask = cv2.warpAffine(
                            mask,
                            trans_img,
                            (img_width, img_height),
                            flags=cv2.INTER_LINEAR,
                        )
                        mask_binary = (mask > 0).astype(np.uint8) * 255
                        combined_masks_channel = np.maximum(
                            combined_masks_channel, mask_binary
                        )

            img = img.astype(np.float32) / 255.0
            combined_masks_channel = combined_masks_channel.astype(np.float32) / 255.0

            masked_img = np.concatenate(
                (img, combined_masks_channel[:, :, np.newaxis]), axis=2
            )

            # Normalize and transpose
            extended_mean = np.concatenate(
                (self.mean, np.array([0.0], dtype=np.float32)[None, None, :]), axis=2
            )
            extended_std = np.concatenate(
                (self.std, np.array([1.0], dtype=np.float32)[None, None, :]), axis=2
            )

            masked_img -= extended_mean
            masked_img /= extended_std
            masked_img = masked_img.transpose(2, 0, 1)[None, :, :, :]

            if self.test_flip:
                masked_img = np.concatenate(
                    (masked_img, masked_img[:, :, :, ::-1].copy()), axis=0
                )

            out[scale] = {
                "image": masked_img,
                "center": center,
                "scale": scaled_size,
                "fmap_h": img_height // self.down_ratio,
                "fmap_w": img_width // self.down_ratio,
            }

        # Return sample_path as img_id for PSR dataset
        return sample_path, out

    def convert_eval_format(self, all_bboxes):
        pass

    def run_eval(self, results, save_dir=None):
        pass

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            out.append(
                (
                    img_id,
                    {
                        s: {
                            k: torch.from_numpy(sample[s][k]).float()
                            if k == "image"
                            else np.array(sample[s][k])
                            for k in sample[s]
                        }
                        for s in sample
                    },
                )
            )
        return out
