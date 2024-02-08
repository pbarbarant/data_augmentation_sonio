# %%
from diffusers import AutoPipelineForText2Image
import torch
from pathlib import Path

NUM_SAMPLES = 100

# Target folder
# synthetic_data_folder = Path("../../data/synthetic_healthy")
synthetic_data_folder = Path("../../data/synthetic_patho")
if not synthetic_data_folder.exists():
    synthetic_data_folder.mkdir()

# Load the model
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16
).to("cuda")

# Check that the file exists
path = Path("../../models/sd_lora/sd_lora_output")
assert path.exists(), "File not found"

# Load the textual inversion embeddings
pipeline.load_lora_weights(path, weight_name="pytorch_lora_weights.safetensors")

modalities = ["Philipps", "General-Electric", "Samsung"]
for modality in modalities:
    for i in range(NUM_SAMPLES):
        print(f"Generating image {i}/{NUM_SAMPLES} for {modality} scanner")
        # Generate the image
        # prompt = f"A healthy cardiac image from a {modality} scanner"
        prompt = f"A pathological cardiac image from a {modality} scanner"
        sample = pipeline(
            prompt,
            num_inference_steps=100,
        ).images[0]

        # Create the folder if it does not exist
        output_folder = synthetic_data_folder / modality
        if not output_folder.exists():
            output_folder.mkdir()

        # Save the image
        sample.save(output_folder / f"{i}.png")
