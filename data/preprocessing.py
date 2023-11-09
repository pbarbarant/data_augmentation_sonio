# %%
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, exposure, filters, io, measure, morphology
from tqdm import tqdm

plt.rcParams["figure.dpi"] = 100
data_path = Path(os.path.dirname(__file__) + "/raw/data_M2_healthy")
output_path = Path(os.path.dirname(__file__) + "/processed")
if not os.path.exists(output_path):
    os.makedirs(output_path)


images_path = sorted(glob.glob(str(data_path / "data_healthy/*.jpg")))
constructor_dict = json.load(open(data_path / "data_constructor_healthy.json", "r"))


def crop_top_panel(image):
    im_gray = color.rgb2gray(image)
    im_gray_equalized = exposure.equalize_hist(im_gray)

    grad = filters.sobel(im_gray_equalized)
    y_hist = np.sum(grad, axis=1)
    cutoff = np.argmax(y_hist > 0.1 * np.max(y_hist))

    return image[cutoff:, 50:-50, :]


def remove_annotations(image, min_size=10000):
    im_gray = color.rgb2gray(image)
    im_gray_equalized = exposure.equalize_hist(im_gray)
    mask = im_gray_equalized > im_gray_equalized.mean()
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    label_img = measure.label(mask)
    regions = measure.regionprops(label_img)
    props = regions[0]
    minr, minc, maxr, maxc = props.bbox

    relative_padding = 0.03 * max(maxr - minr, maxc - minc)
    minr = int(max(0, minr - relative_padding))
    minc = int(max(0, minc - relative_padding))
    maxr = int(min(image.shape[0], maxr + relative_padding))
    maxc = int(min(image.shape[1], maxc + relative_padding))

    return image[minr:maxr, minc:maxc]


def morphological_opening(image, radius=3):
    disk = morphology.disk(radius)

    # Select color channels
    im_r = image[:, :, 0]
    im_g = image[:, :, 1]
    im_b = image[:, :, 2]

    # Morphological opening
    im_r = morphology.opening(im_r, disk)
    im_g = morphology.opening(im_g, disk)
    im_b = morphology.opening(im_b, disk)

    return np.stack([im_r, im_g, im_b], axis=-1)


def sharpen_image(image, radius=2, amount=1):
    # Select color channels
    im_r = image[:, :, 0]
    im_g = image[:, :, 1]
    im_b = image[:, :, 2]

    # Sharpen image
    im_r = filters.unsharp_mask(im_r, radius=radius, amount=amount)
    im_g = filters.unsharp_mask(im_g, radius=radius, amount=amount)
    im_b = filters.unsharp_mask(im_b, radius=radius, amount=amount)

    return np.stack([im_r, im_g, im_b], axis=-1)


def pad_and_resize(image, size=512):
    pil_img = Image.fromarray((255 * image).astype(np.uint8), "RGB")
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), (0, 0, 0))
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), (0, 0, 0))
        result.paste(pil_img, ((height - width) // 2, 0))
    result = result.resize((size, size))
    image = np.array(result)
    return image


def detect_segmentation(image, min_blob_area=50):
    # Convert the image to the HSV color space
    hsv_image = color.rgb2hsv(image)

    # Define the lower and upper bounds for the red and blue colors
    color_lower_red = np.array([0, 0.62, 0.57])
    color_upper_red = np.array([0.05, 1, 1])
    color_lower_blue = np.array([0.55, 0.4, 0.5])
    color_upper_blue = np.array([0.7, 1, 1])

    # Create a mask to isolate the specified colors
    color_mask_red = np.all(
        (hsv_image >= color_lower_red) & (hsv_image <= color_upper_red), axis=-1
    )
    color_mask_blue = np.all(
        (hsv_image >= color_lower_blue) & (hsv_image <= color_upper_blue), axis=-1
    )

    color_mask = (color_mask_red + color_mask_blue) > 0

    # Apply morphological operations to clean up the mask
    color_mask = morphology.binary_closing(color_mask, morphology.disk(3))

    # Label connected components in the mask to identify blobs
    label_image = measure.label(color_mask, connectivity=2)

    # Filter for blobs of sufficient size
    for region in measure.regionprops(label_image):
        if region.area >= min_blob_area:
            return None

    return image


def image_processing_pipeline(image):
    # Crop top panel
    # image = crop_top_panel(image)
    # Apply morphological opening
    image = morphological_opening(image, radius=3)
    # Sharpen image
    image = sharpen_image(image, radius=2, amount=1)
    # Remove text and annotations
    image = remove_annotations(image, min_size=1e5)
    # Center and resize
    image = pad_and_resize(image, size=512)
    # Detect segmentation and reject if any
    image = detect_segmentation(image)

    return image


plotting = False
error_list = []
if __name__ == "__main__":
    for path in tqdm(images_path):
        im_name = path.split("/")[-1]
        im_id = im_name.split(
            "_",
        )[1][:-4]
        if im_id not in constructor_dict:
            print(f"Image {im_id} not in constructor dict")
        else:
            try:
                original_image = io.imread(path)
                image = np.copy(original_image)
                image = image_processing_pipeline(image)

                if image is not None:
                    # Save image
                    io.imsave(output_path / (im_id + ".jpg"), image)

                    if plotting:
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(original_image)
                        ax[1].imshow(image)
                        ax[0].axis("off")
                        ax[1].axis("off")
                        ax[0].set_title(f"Constructor: {constructor_dict[im_id]}")
                        ax[1].set_title("Processed image")
                        plt.show()
                else:
                    error_list.append(im_id)
            except Exception as e:
                error_list.append(im_id)
                del e
    if len(error_list) > 0:
        print("Number of errors:\n")
        print("Error list:\n")
        print(*error_list, sep="\n")
