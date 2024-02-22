# %%
from diffusers import AutoPipelineForText2Image
import torch
from torchvision import transforms
from pathlib import Path

NUM_SAMPLES = 50

device = "cuda:3"


def add_speckle_noise(image, mean=0, std=0.15, patch_size=512):
    """Adds more realistic speckle noise to an image with larger areas.

    Args:
        image: A PyTorch tensor of shape (C, H, W) representing the image.
        mean: The mean of the Gaussian distribution for generating noise.
        std: The standard deviation of the Gaussian distribution.
        patch_size: Size of the patches for applying localized noise (larger for bigger areas).

    Returns:
        A PyTorch tensor of the same shape as the input image with more realistic speckle noise.
    """

    # Generate Gaussian noise with low std for overall noise
    base_noise = torch.randn(image.shape, device=image.device) * 0.05 + mean

    # Generate additional localized noise with higher std and larger patches
    patch_noise = torch.zeros_like(image)
    for i in range(0, image.shape[1], patch_size):
        for j in range(0, image.shape[2], patch_size):
            patch_noise[:, i : i + patch_size, j : j + patch_size] = (
                torch.randn(patch_size, patch_size, device=image.device) * std + mean
            )

    # Combine both noise components
    noisy_image = image + image * base_noise + patch_noise

    # Clamp the noisy image values to the valid range [0, 1]
    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image


# Target folder
synthetic_data_folder = Path("../../data/synthetic_healthy")
# synthetic_data_folder = Path("../../data/synthetic_patho")
if not synthetic_data_folder.exists():
    synthetic_data_folder.mkdir()

# Load the model
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16
).to(device)

# Check that the file exists
path = Path("../../models/sd_lora/sd_lora_output")
assert path.exists(), "File not found"

# Load the textual inversion embeddings
pipeline.load_lora_weights(path, weight_name="pytorch_lora_weights.safetensors")

modalities = ["Philipps", "General-Electric", "Samsung"]
for modality in modalities:
    for i in range(NUM_SAMPLES):
        print(f"Generating image {i}/{NUM_SAMPLES} for {modality} scanner.")
        # Generate the image
        prompt = f"A healthy cardiac image from a {modality} scanner. The image should exhibit a high level of speckle noise like on the ultrasound or sar images."
        sample = pipeline(
            prompt,
            num_inference_steps=100,
        ).images[0]

        # Create the folder if it does not exist
        output_folder = synthetic_data_folder / modality
        if not output_folder.exists():
            output_folder.mkdir()

        noisy_sample = add_speckle_noise(
            transforms.ToTensor()(transforms.GaussianBlur(kernel_size=1)(sample))
        )

        # Convert to PIL
        noisy_sample = transforms.ToPILImage()(noisy_sample)

        # Save the image
        noisy_sample.save(output_folder / f"{i}.png")
