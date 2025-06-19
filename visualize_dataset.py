from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from datasets.psr import PSRDataset, PSR_MEAN, PSR_STD


def visualize_dataset(root_dir, img_size=512):
    dataset = PSRDataset(root_dir=root_dir, split="train", img_size=img_size)
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
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Red Channel", "Green Channel", "Blue Channel", "Mask Channel"]

        for channel_idx in range(4):
            ax = axes[channel_idx]
            # Display each channel as grayscale
            ax.imshow(unnormalized_masked_img[:, :, channel_idx], cmap="gray")
            ax.set_title(titles[channel_idx])
            ax.axis("off")

            # Spot the center point on the mask channel (channel_idx == 3)
            if channel_idx == 3:  # Mask channel
                part_centers = data["masks_bbox"]
                for part_name, bbox in part_centers.items():
                    # center_coords are (x, y)
                    ax.scatter(
                        bbox["center"][0],
                        bbox["center"][1],
                        color="red",
                        marker="x",
                        s=100,
                        linewidths=2,
                        label=part_name,
                    )
                    # Draw bounding box rectangle
                    # bbox["scale"] is [width, height]
                    # bbox["center"] is [x, y]
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
                ax.legend()  # Show legend for part names

        plt.suptitle(f"Masked Image {i + 1} - Channel by Channel")
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to prevent suptitle overlap
        plt.show()

        if i == 5:  # Visualize 5 images for demonstration
            break


if __name__ == "__main__":
    # Replace 'path/to/your/psr_dataset' with the actual path to your dataset
    # Example: root_dir = '/home/user/data/psr_dataset'
    root_directory = "/home/cyl/Reconst/KAF-Net/data/Sample PSR/"
    visualize_dataset(root_directory)
