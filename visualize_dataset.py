from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from datasets.psr import PSRDataset, PSR_MEAN, PSR_STD, PSR_KR_CAT_IDX


def visualize_dataset(root_dir, img_size=512):
    # down_ratio = {"hmap": 32, "wh": 8, "reg": 16, "kaf": 4}
    down_ratio = {"hmap": 32, "wh": 4, "reg": 4, "kaf": 4}
    dataset = PSRDataset(
        root_dir=root_dir, split="train", down_ratio=down_ratio, img_size=img_size
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        masked_img = data["masked_img"][0].numpy()  # Get the first image from the batch

        # Define extended mean and std for 4 channels (RGB + Mask)
        extended_mean = np.array(PSR_MEAN + [0.0]).reshape(4, 1, 1)
        extended_std = np.array(PSR_STD + [1.0]).reshape(4, 1, 1)

        # Unnormalize the masked_img
        unnormalized_masked_img = masked_img * extended_std + extended_mean
        unnormalized_masked_img = np.clip(
            unnormalized_masked_img, 0, 1
        )  # Clip values to [0, 1]

        # Convert from [C, H, W] to [H, W, C]
        unnormalized_masked_img = masked_img.transpose(1, 2, 0)

        # Visualize each channel
        # Changed from 1,6 to 2,4 to accommodate new plots
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        hmap = data["hmap"][0].numpy()
        regs = data["regs"][0].numpy()
        reg_inds = data["reg_inds"][0].numpy()
        ind_masks = data["ind_masks"][0].numpy()
        part_centers = data["masks_bbox_center"][data["reg_inds"] > 0].cpu().numpy()
        part_scales = data["masks_bbox_wh"][data["wh_inds"] > 0].cpu().numpy()
        print(part_centers)

        # Changed from 2,3 to 2,4 to accommodate new plots
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()

        titles = [
            "Red Channel",
            "Green Channel",
            "Blue Channel",
            "Mask Channel",
            "Hmap (Housing Category)",
            "Width/Height & Regs",
            "RAF Field Magnitude (Fixed)",  # New title
            "RAF Weights (Fixed)",  # New title
        ]

        # Display RGB and Mask channels
        for channel_idx in range(4):
            ax = axes[channel_idx]
            ax.imshow(unnormalized_masked_img[:, :, channel_idx], cmap="gray")
            ax.set_title(titles[channel_idx])
            ax.axis("off")

            if channel_idx == 3:  # Mask channel
                for idx in range(len(part_centers)):  # Loop over part:
                    ax.scatter(
                        part_centers[idx][0],
                        part_centers[idx][1],
                        color="red",
                        marker="x",
                        s=100,
                        linewidths=2,
                        label=idx,
                    )
                    width_bbox = part_scales[idx][0]
                    height_bbox = part_scales[idx][1]
                    x_min = part_centers[idx][0] - width_bbox / 2
                    y_min = part_centers[idx][1] - height_bbox / 2

                    rect = plt.Rectangle(
                        (x_min, y_min),
                        width_bbox,
                        height_bbox,
                        linewidth=1,
                        edgecolor="cyan",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                ax.legend()

        # Display Hmap for a specific category (e.g., 'door' which is index 11)
        ax = axes[4]
        # Assuming 'door' is at index 11 in PSR_FUNC_CAT
        # You might want to make this dynamic or configurable
        door_hmap_idx = dataset.func_cat_ids.get("housing", -1)
        if door_hmap_idx != -1 and hmap.shape[0] > door_hmap_idx:
            ax.imshow(hmap[door_hmap_idx], cmap="hot", alpha=1)
            ax.set_title(titles[4])
            ax.axis("off")

            # Overlay center points from inds and regs on the heatmap
            for obj_idx in range(len(reg_inds)):
                if ind_masks[obj_idx] == 1:
                    fmap_w = dataset.offset_size["w"]
                    center_fmap_x = (
                        (reg_inds[obj_idx] % fmap_w + regs[obj_idx, 0])
                        * dataset.hmap_size["w"]
                        / dataset.offset_size["w"]
                    )
                    center_fmap_y = (
                        (reg_inds[obj_idx] // fmap_w + regs[obj_idx, 1])
                        * dataset.hmap_size["w"]
                        / dataset.offset_size["w"]
                    )
                    ax.scatter(
                        center_fmap_x,
                        center_fmap_y,
                        color="blue",
                        marker="x",
                        s=50,
                        edgecolors="white",
                        linewidths=1,
                    )
        else:
            ax.set_title(f"{titles[4]} (Not Available)")
            ax.axis("off")

        # Display w_h_ and regs
        ax = axes[5]
        ax.set_title(titles[5])
        ax.axis("off")

        # Filter out zero entries (where ind_masks is 0)
        valid_indices = np.where(ind_masks == 1)[0]
        valid_regs = regs[valid_indices]

        if len(valid_indices) > 0:
            table_data = []
            for j in range(len(valid_indices)):
                table_data.append(
                    [
                        f"{part_centers[j, 0]:.2f}",
                        f"{part_centers[j, 1]:.2f}",
                        f"{part_scales[j, 0]:.2f}",
                        f"{part_scales[j, 1]:.2f}",
                        f"{valid_regs[j, 0]:.2f}",
                        f"{valid_regs[j, 1]:.2f}",
                    ]
                )
            col_labels = ["X", "Y", "Width", "Height", "Reg_X", "Reg_Y"]
            ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            ax.set_title(titles[5])
        else:
            ax.text(
                0.5,
                0.5,
                "No valid objects",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        # Display RAF Field Magnitude (Fixed)
        ax = axes[6]
        raf_field = data["gt_relations"][0].cpu().numpy()
        fixed_rel_idx = PSR_KR_CAT_IDX.get("revolute-controlled", -1)
        if fixed_rel_idx != -1 and fixed_rel_idx < raf_field.shape[0]:
            raf_field_fixed = raf_field[fixed_rel_idx]
            # Calculate magnitude of the 2D vectors
            raf_magnitude = np.linalg.norm(raf_field_fixed, axis=0)
            ax.imshow(raf_magnitude, cmap="viridis")
            ax.set_title(titles[6])
            ax.axis("off")
        else:
            ax.set_title(f"{titles[6]} (Not Available)")
            ax.axis("off")

        # Display RAF Weights (Fixed)
        ax = axes[7]
        raf_weights = data["gt_relations_weights"][0].cpu().numpy()
        if fixed_rel_idx != -1 and fixed_rel_idx < raf_weights.shape[0]:
            raf_weights_fixed = raf_weights[fixed_rel_idx]
            # Sum the weights across the 2 channels (x and y components)
            raf_weights_sum = np.sum(raf_weights_fixed, axis=0)
            ax.imshow(raf_weights_sum, cmap="hot")
            ax.set_title(titles[7])
            ax.axis("off")
        else:
            ax.set_title(f"{titles[7]} (Not Available)")
            ax.axis("off")

        plt.suptitle(f"Dataset Visualization Sample {i + 1}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        if i == 5:  # Visualize 5 images for demonstration
            break


if __name__ == "__main__":
    root_directory = "/home/cyl/Reconst/Data/PSR/deep_furniture_part1/"
    visualize_dataset(root_directory)
