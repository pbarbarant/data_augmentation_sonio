# %%
import json
import glob
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from pythae.data.datasets import BaseDataset
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig

from cvae import CVAEConfig, Custom_encoder, Custom_decoder, CVAE

# print(torch.cuda.is_available())
# torch.cuda.set_device("cuda:1")
# print(torch.cuda.current_device())
device = "cpu"


dataset = "../../data/processed/"
label_path = "../../data/raw/data_M2_healthy/data_constructor_healthy.json"

# Load all images
image_list = []
label_list = []
with open(label_path) as f:
    for line in f:
        dict_label = json.loads(line)

for ext in ["*.jpg", "*.png"]:
    for f in glob.glob(os.path.join(dataset, ext)):
        image_list.append(f)

for image_path in image_list:
    image_id = image_path.split("/")[-1].split(".")[0]
    if image_id in dict_label.keys():
        label_list.append(dict_label[image_id])

# One-hot encoding
labels = torch.zeros(len(label_list), 3)
for i in range(len(label_list)):
    if label_list[i] == "Philips":
        labels[i, :] = torch.tensor([1, 0, 0])
    elif label_list[i] == "GE":
        labels[i, :] = torch.tensor([0, 1, 0])
    elif label_list[i] == "Samsung":
        labels[i, :] = torch.tensor([0, 0, 1])

# %%
# Load images into a tensor
images = torch.stack(
    [torch.from_numpy(plt.imread(f)).permute(2, 0, 1) for f in image_list]
)
print("Loaded {} images".format(len(images)))
# Normalize images
images = images / 255.0

# Split into training and validation sets
training_images = images[:700].to(device)
validation_images = images[700:].to(device)

# Resize images to 64x64
transform = transforms.Compose([transforms.Resize((128, 128))])
resized_images = torch.stack([transform(img) for img in images])


fig, ax = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    ax[i].imshow(training_images[i].permute(1, 2, 0))
    ax[i].set_title(f"Image {i}")
    ax[i].axis("off")
plt.show()

# %%
# Set up the training configuration
my_training_config = BaseTrainerConfig(
    output_dir="../models/cvae",
    num_epochs=2000,
    learning_rate=1e-3,
    per_device_train_batch_size=200,
    per_device_eval_batch_size=200,
    train_dataloader_num_workers=2,
    eval_dataloader_num_workers=2,
    steps_saving=20,
    optimizer_cls="AdamW",
    optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5},
)
# Set up the model configuration
cvae_config = CVAEConfig(class_size=3, latent_dim=10)
# Build the model
model = CVAE(
    model_config=cvae_config,
    encoder=Custom_encoder(cvae_config),
    decoder=Custom_decoder(cvae_config),
)
# Build the Pipeline
pipeline = TrainingPipeline(training_config=my_training_config, model=model)

#
print(f"resized_images.shape: {resized_images.shape}")
print(f"labels.shape: {labels.shape}")
# %%
# Launch the Pipeline
resized_images = resized_images.view(len(resized_images), -1)
train_data = BaseDataset(data=resized_images[:700], labels=labels[:700])
eval_data = BaseDataset(data=resized_images[700:], labels=labels[700:])
pipeline(
    train_data=training_images,  # must be torch.Tensor, np.array or torch datasets
    eval_data=validation_images,  # must be torch.Tensor, np.array or torch datasets
)
# Use fake data for testing
# model(eval_data)
