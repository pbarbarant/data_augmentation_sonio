# %%
import glob
import os
from pathlib import Path
import json
import pandas as pd
import shutil
from tqdm import tqdm
from datasets import load_dataset

source_folder = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/processed_patho/"
)
target_folder = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/processed_by_modality/patho/"
)
modalities_path = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/raw/data_patho/data_constructor.json"
)

assert source_folder.exists(), "Source folder does not exist"
assert modalities_path.exists(), "JSON file with modalities does not exist"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Load modalities
modalities_dict = json.load(
    open(
        "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/raw/data_patho/data_constructor.json",
        "r",
    )
)
# Load pathologies names
patho_dict = json.load(
    open(
        "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/raw/data_patho/data_pathology.json",
        "r",
    )
)
assert len(modalities_dict) > 0, "No modalities found"
assert len(patho_dict) > 0, "No pathologies found"


filenames_list = glob.glob(str(source_folder / "*.jpg"))

# Create json file with filenames and modalities
dict_names_captions = {
    "file_name": [],
    "caption": [],
}

# Replace identifiers by file paths and assign unique number for each image according to modality
number_modality = {
    "GE": 0,
    "Philips": 0,
    "Samsung": 0,
}

for filename in tqdm(filenames_list):
    identifier = filename.split("/")[-1].split(".")[0]
    modality = modalities_dict[identifier]
    patho = patho_dict[identifier]
    if patho == "Cardiomégalie":
        patho = "Cardiomegaly"
    else:
        patho = "Interventricular communication"

    output_filename = f"{modality}{number_modality[modality]}.jpg"
    number_modality[modality] += 1
    target_path = target_folder / output_filename

    # Add modality to dataframe
    if modality == "GE":
        modality = "General-Electric"

    dict_names_captions["file_name"].append(output_filename)
    dict_names_captions["caption"].append(
        f"A pathological cardiac image with {patho} from a {modality} scanner"
    )
    # Copy file to target folder
    shutil.copy(filename, target_path)

# Create dataframe
df = pd.DataFrame.from_dict(dict_names_captions)
# Save to csv
df.to_csv(target_folder / "metadata.csv", index=False)

# Create a hf dataset
dataset = load_dataset(str(target_folder))
# Save to disk
dataset.save_to_disk(
    str(
        "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/hf_dataset_patho"
    )
)
