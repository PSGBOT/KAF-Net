from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import _tranpose_and_gather_feature, load_model

from datasets.psr import PSRDataset, PSR_MEAN, PSR_STD, PSR_KR_CAT_IDX


def visualize_dataset(root_dir, img_size=512):
    # down_ratio = {"hmap": 32, "wh": 8, "reg": 16, "kaf": 4}
    down_ratio = {"p5": 32, "p4": 16, "p3": 8, "p2": 4}
    dataset = PSRDataset(
        root_dir=root_dir, split="train", down_ratio=down_ratio, img_size=img_size
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        masked_img = data["masked_img"][0].numpy()  # Get the first image from the batch

        # Define extended mean and std for 6 channels (3 RGB + 3 mask RGB)
        extended_mean = np.array(PSR_MEAN + [0.0, 0.0, 0.0]).reshape(6, 1, 1)
        extended_std = np.array(PSR_STD + [1.0, 1.0, 1.0]).reshape(6, 1, 1)

        # Unnormalize the masked_img
        unnormalized_masked_img = masked_img * extended_std + extended_mean
        unnormalized_masked_img = np.clip(
            unnormalized_masked_img, 0, 1
        )  # Clip values to [0, 1]

        # Convert from [C, H, W] to [H, W, C]
        unnormalized_masked_img = unnormalized_masked_img.transpose(
            1, 2, 0
        )  # This line is correct as is

        # Visualize each channel
        # Changed to 3,4 to accommodate RGB image, RGB mask, and analysis plots
        num_fpn = 4
        fig, axes = plt.subplots(2 + num_fpn, 4, figsize=(24, 18))
        # ind_masks = data["ind_masks"][0].numpy()
        # part_centers = (
        #     data["masks_bbox_center"][0][ind_masks == 1].cpu().numpy()
        # )  # Corrected filtering and added [0] for batch dim
        # part_scales = (
        #     data["masks_bbox_wh"][0][ind_masks == 1].cpu().numpy()
        # )  # Corrected filtering and added [0] for batch dim
        # print(part_centers)
        # print(part_scales)

        axes = axes.flatten()
        titles = [
            "Original RGB Image",
            "Red Channel (Original)",
            "Green Channel (Original)",
            "Blue Channel (Original)",
            "Colored Masks RGB",
            "Red Channel (Masks)",
            "Green Channel (Masks)",
            "Blue Channel (Masks)",
        ]

        # Display original RGB image (combined)
        ax = axes[0]
        original_rgb = unnormalized_masked_img[:, :, :3]
        ax.imshow(original_rgb)
        ax.set_title(titles[0])
        ax.axis("off")

        # Display individual original RGB channels
        for channel_idx in range(3):
            ax = axes[channel_idx + 1]
            ax.imshow(unnormalized_masked_img[:, :, channel_idx], cmap="gray")
            ax.set_title(titles[channel_idx + 1])
            ax.axis("off")

        # Display colored masks RGB (combined)
        ax = axes[4]
        mask_rgb = unnormalized_masked_img[:, :, 3:6]
        ax.imshow(mask_rgb)
        ax.set_title(titles[4])
        ax.axis("off")

        # Add part centers and bounding boxes to the colored masks view
        # for idx in range(len(part_centers)):
        #     ax.scatter(
        #         part_centers[idx][0],
        #         part_centers[idx][1],
        #         color="white",
        #         marker="x",
        #         s=100,
        #         linewidths=2,
        #         label=f"Part {idx}",
        #     )
        #     width_bbox = part_scales[idx][0]
        #     height_bbox = part_scales[idx][1]
        #     x_min = part_centers[idx][0] - width_bbox / 2
        #     y_min = part_centers[idx][1] - height_bbox / 2
        #
        #     rect = plt.Rectangle(
        #         (x_min, y_min),
        #         width_bbox,
        #         height_bbox,
        #         linewidth=2,
        #         edgecolor="white",
        #         facecolor="none",
        #     )
        #     ax.add_patch(rect)
        # if len(part_centers) > 0:
        #     ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Display individual mask RGB channels
        for channel_idx in range(3):
            ax = axes[channel_idx]
            ax = axes[channel_idx + 5]
            ax.imshow(unnormalized_masked_img[:, :, channel_idx + 3], cmap="gray")
            ax.set_title(titles[channel_idx + 5])
            ax.axis("off")
        # Select a specific FPN level for visualization (e.g., the coarsest level, which is the first one in the list)
        for fpn_level_idx in range(num_fpn):
            hmap_level = data["hmap"][fpn_level_idx][
                0
            ].numpy()  # Added [0] for batch dim
            # print(data["hmap"][1].shape)
            regs_level = data["regs"][fpn_level_idx][
                0
            ].numpy()  # Added [0] for batch dim
            wh_level = data["w_h_"][fpn_level_idx][0].numpy()  # Added [0] for batch dim
            reg_inds_level = data["reg_inds"][fpn_level_idx][
                0
            ].numpy()  # Added [0] for batch dim
            wh_inds_level = data["wh_inds"][fpn_level_idx][
                0
            ].numpy()  # Added [0] for batch dim
            ind_masks_level = data["ind_masks"][fpn_level_idx][0]
            # print(wh_level[wh_inds_level >= 1])
            # print(wh_inds_level)

            # Display Hmap for a specific category (e.g., 'door' which is index 11)
            ax = axes[7 + fpn_level_idx * 4 + 1]
            # Assuming 'door' is at index 11 in PSR_FUNC_CAT
            # You might want to make this dynamic or configurable, using 'housing' as an example
            door_hmap_idx = dataset.func_cat_ids.get("door", -1)
            if door_hmap_idx != -1 and hmap_level.shape[0] > door_hmap_idx:
                ax.imshow(hmap_level[door_hmap_idx], cmap="hot", alpha=1)
                ax.set_title(f"door hmap p{5 - fpn_level_idx}")
                ax.axis("off")

                # Overlay center points from inds and regs on the heatmap
                # Calculate fmap_size for the current FPN level
                fmap_w = int(
                    dataset.img_size["w"]
                    / dataset.down_ratio[list(dataset.down_ratio.keys())[fpn_level_idx]]
                )
                fmap_h = int(
                    dataset.img_size["h"]
                    / dataset.down_ratio[list(dataset.down_ratio.keys())[fpn_level_idx]]
                )

                for obj_idx in range(len(reg_inds_level)):
                    if ind_masks_level[obj_idx] == 1:
                        center_fmap_x = (
                            reg_inds_level[obj_idx] % fmap_w + regs_level[obj_idx, 0]
                        )
                        center_fmap_y = (
                            reg_inds_level[obj_idx] // fmap_w + regs_level[obj_idx, 1]
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
                ax.set_title(f"door hmap p{5 - fpn_level_idx} (not found)")
                ax.axis("off")

            # # Display w_h_ and regs
            # ax = axes[7 + fpn_level_idx * 4 + 2]
            # ax.set_title(titles[9])
            # ax.axis("off")
            #
            # # Filter out zero entries (where ind_masks is 0)
            # valid_indices = np.where(ind_masks_level == 1)[
            #     0
            # ]  # ind_masks is not per-FPN level, but common for all objects
            # valid_regs = regs_level[valid_indices]
            #
            # if len(valid_indices) > 0:
            #     table_data = []
            #     for j in range(len(valid_indices)):
            #         table_data.append(
            #             [
            #                 f"{part_centers[j, 0]:.2f}",
            #                 f"{part_centers[j, 1]:.2f}",
            #                 f"{part_scales[j, 0]:.2f}",
            #                 f"{part_scales[j, 1]:.2f}",
            #                 f"{valid_regs[j, 0]:.2f}",
            #                 f"{valid_regs[j, 1]:.2f}",
            #             ]
            #         )
            #     col_labels = ["X", "Y", "Width", "Height", "Reg_X", "Reg_Y"]
            #     ax.table(
            #         cellText=table_data,
            #         colLabels=col_labels,
            #         loc="center",
            #         cellLoc="center",
            #     )
            #     ax.set_title(titles[9])
            # else:
            #     ax.text(
            #         0.5,
            #         0.5,
            #         "No valid objects",
            #         horizontalalignment="center",
            #         verticalalignment="center",
            #         transform=ax.transAxes,
            #     )

            # Display RAF Field Magnitude (Fixed)
            ax = axes[7 + fpn_level_idx * 4 + 3]
            raf_field = (
                data["gt_relations"][fpn_level_idx][0].cpu().numpy()
            )  # Added [0] for batch dim
            fixed_rel_idx = PSR_KR_CAT_IDX.get("fixed", -1)
            if fixed_rel_idx != -1 and fixed_rel_idx < raf_field.shape[0]:
                raf_field_fixed = raf_field[fixed_rel_idx]
                # Calculate magnitude of the 2D vectors
                raf_magnitude = np.linalg.norm(raf_field_fixed, axis=0)
                ax.imshow(raf_magnitude, cmap="viridis")
                ax.set_title("fixed kaf mag")
                ax.axis("off")
            else:
                ax.set_title("fixed kaf mag (Not Available)")
                ax.axis("off")

            # Display RAF Weights (Fixed)
            ax = axes[7 + fpn_level_idx * 4 + 4]
            raf_weights = (
                data["gt_relations_weights"][fpn_level_idx][0].cpu().numpy()
            )  # Added [0] for batch dim
            if fixed_rel_idx != -1 and fixed_rel_idx < raf_weights.shape[0]:
                raf_weights_fixed = raf_weights[fixed_rel_idx]
                # Sum the weights across the 2 channels (x and y components)
                raf_weights_sum = np.sum(raf_weights_fixed, axis=0)
                ax.imshow(raf_weights_sum, cmap="hot")
                ax.set_title("weights")
                ax.axis("off")
            else:
                ax.set_title("weights (Not Available)")
                ax.axis("off")

        plt.suptitle(f"Dataset Visualization Sample {i + 1}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        if i == 5:  # Visualize 5 images for demonstration
            break


if __name__ == "__main__":
    root_directory = "/home/cyl/Reconst/Data/PSR_v2/train/"
    visualize_dataset(root_directory)
