# %%
import glob
from pathlib import Path
import json
import shutil
from tqdm import tqdm

source_folder = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/processed/"
)
target_folder = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/processed_by_modality/"
)
modalities_path = Path(
    "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/raw/data_M2_healthy/data_constructor_healthy.json"
)

assert source_folder.exists(), "source_folder does not exist"
assert target_folder.exists(), "target_folder does not exist"
assert modalities_path.exists(), "modalities_path does not exist"

# Load modalities
modalities_dict = json.load(
    open(
        "/data/parietal/store3/work/pbarbara/data_augmentation_sonio/data/raw/data_M2_healthy/data_constructor_healthy.json",
        "r",
    )
)

filenames_list = glob.glob(str(source_folder / "*.jpg"))

# Replace identifiers by file paths and assign unique number for each image according to modality
number_modality = {
    "GE": 0,
    "Philips": 0,
    "Samsung": 0,
}
for filename in tqdm(filenames_list):
    identifier = filename.split("/")[-1].split(".")[0]
    modality = modalities_dict[identifier]
    output_filename = f"{modality}{number_modality[modality]}.jpg"
    number_modality[modality] += 1
    target_path = target_folder / output_filename
    # Copy file to target folder
    shutil.copy(filename, target_path)
