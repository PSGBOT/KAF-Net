from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from datasets.psr import PSRDataset, PSR_MEAN, PSR_STD


def visualize_dataset(root_dir, img_size=512):
    dataset = PSRDataset(root_dir=root_dir, split="train", img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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
        fig, axes = plt.subplots(1, 6, figsize=(16, 4))
        hmap = data["hmap"][0].numpy()
        w_h_ = data["w_h_"][0].numpy()
        regs = data["regs"][0].numpy()
        inds = data["inds"][0].numpy()
        ind_masks = data["ind_masks"][0].numpy()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        titles = [
            "Red Channel",
            "Green Channel",
            "Blue Channel",
            "Mask Channel",
            "Hmap (Door Category)",
            "Width/Height & Regs",
        ]

        # Display RGB and Mask channels
        for channel_idx in range(4):
            ax = axes[channel_idx]
            ax.imshow(unnormalized_masked_img[:, :, channel_idx], cmap="gray")
            ax.set_title(titles[channel_idx])
            ax.axis("off")

            if channel_idx == 3:  # Mask channel
                part_centers = data["masks_bbox"]
                for part_name, bbox in part_centers.items():
                    ax.scatter(
                        bbox["center"][0],
                        bbox["center"][1],
                        color="red",
                        marker="x",
                        s=100,
                        linewidths=2,
                        label=part_name,
                    )
                    width_bbox = bbox["scale"][0]
                    height_bbox = bbox["scale"][1]
                    x_min = bbox["center"][0] - width_bbox / 2
                    y_min = bbox["center"][1] - height_bbox / 2

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
        door_hmap_idx = dataset.func_cat_ids.get("door", -1)
        if door_hmap_idx != -1 and hmap.shape[0] > door_hmap_idx:
            ax.imshow(hmap[door_hmap_idx], cmap="hot", alpha=1)
            ax.set_title(titles[4])
            ax.axis("off")

            # Overlay center points from inds and regs on the heatmap
            for obj_idx in range(len(inds)):
                if ind_masks[obj_idx] == 1:
                    fmap_w = dataset.fmap_size["w"]
                    center_fmap_x = inds[obj_idx] % fmap_w + regs[obj_idx, 0]
                    center_fmap_y = inds[obj_idx] // fmap_w + regs[obj_idx, 1]
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
        valid_w_h_ = w_h_[valid_indices]
        valid_regs = regs[valid_indices]

        if len(valid_indices) > 0:
            table_data = []
            for j in range(len(valid_indices)):
                table_data.append(
                    [
                        f"{valid_w_h_[j, 0]:.2f}",
                        f"{valid_w_h_[j, 1]:.2f}",
                        f"{valid_regs[j, 0]:.2f}",
                        f"{valid_regs[j, 1]:.2f}",
                    ]
                )
            col_labels = ["Width", "Height", "Reg_X", "Reg_Y"]
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

        plt.suptitle(f"Dataset Visualization Sample {i + 1}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        if i == 5:  # Visualize 5 images for demonstration
            break


if __name__ == "__main__":
    root_directory = "/home/cyl/Reconst/Data/PSR/deep_furniture_part1/"
    visualize_dataset(root_directory)
