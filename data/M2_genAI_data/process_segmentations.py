# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

plt.style.use("dark_background")

data_path = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/M2_genAI_data/"
)
# Check that the data is there
assert data_path.exists()

output_path = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/segmentations/"
)
if not output_path.exists():
    output_path.mkdir()

image_path = data_path / "images"
image_list = list(image_path.glob("*.npy"))
print(len(image_list))

# # Plot some images
# fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# for i, ax in enumerate(axs.ravel()):
#     ax.imshow(np.load(image_list[i], allow_pickle=True))
#     ax.axis("off")
# plt.show()

seg_path = data_path / "points_4CH.json"
# Load the segmentation
with open(seg_path, "r") as f:
    try:
        segmentations = json.load(f)
    except Exception as e:
        print("Error in loading the json file:", str(e))

    for i, seg in tqdm(enumerate(segmentations)):
        first_seg = segmentations[i]
        id = first_seg["id"]
        annotations = first_seg["annotations"]
        result = annotations[0]["result"]
        # original_width = result["original_width"]
        # original_height = result["original_height"]
        fig = plt.figure(figsize=(512 / 256, 512 / 256), dpi=256)
        for val in range(len(result)):
            value = result[val]["value"]
            points = np.array(value["points"])
            # Plot the convex hull  of the points
            hull = ConvexHull(points)
            # plot area of the convex hull
            plt.fill(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                c="white",
                alpha=0.5,
            )
        plt.axis("off")
        fig.savefig(
            output_path / f"{id}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=256,
        )
        plt.close(fig)
