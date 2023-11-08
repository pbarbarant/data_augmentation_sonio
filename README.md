# Data Augmentation for Ultrasound Foetal Echography Images using Conditional Variational Autoencoders

## Overview

This GitHub repository contains code and resources for data augmentation of ultrasound foetal echography images using conditional variational autoencoders (cVAEs). Data augmentation is a critical step in machine learning, especially in medical imaging tasks, where obtaining a large and diverse dataset is often challenging. This repository provides a solution to enhance the dataset for training ultrasound foetal echography image analysis models using cVAEs.

### What is a Conditional Variational Autoencoder (cVAE)?

A conditional variational autoencoder (cVAE) is a type of generative model that extends the capabilities of a standard VAE. It allows us to generate new data points based on specific conditions or labels. In this context, we can use cVAEs to generate augmented ultrasound foetal echography images conditioned on various factors like gestational age, pathology type, and more.

## Features

- **Data Augmentation**: Generate synthetic ultrasound foetal echography images to diversify your dataset.
- **Conditioned Generation**: Generate images based on specific conditions, such as machine constructor, pathology type, etc.
- **Variational Latent Space**: Explore the latent space of the cVAE model to understand the image generation process.

## Repository Structure

The repository is structured as follows:

- `data/`: This directory contains example ultrasound foetal echography images for training and evaluation.
- `notebooks/`: Jupyter notebooks with example code for using the cVAE for data augmentation and latent space exploration.
- `src/`: Python source code for the conditional variational autoencoder implementation.
- `models/`: Pretrained cVAE models for ultrasound foetal echography data.
- `pyproject.toml`: List of Python dependencies to run the code in this repository.
- `README.md`: This README file, providing an overview of the repository and usage instructions.

## Usage

To use the code and resources in this repository, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/pbarbarant/data_augmentation_sonio.git
   ```
2. Install the requirements in a dedicated Python environment:

   ```bash
   pip install -e .
   ```
3. For development purposes, contributors should run:

   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
4. Explore the provided Jupyter notebooks in the `notebooks/` directory to understand how to train and use cVAE models for data augmentation.
5. Use the pretrained cVAE models in the `models/` directory for generating augmented ultrasound foetal echography images.

## Getting Started

If you're new to conditional variational autoencoders or need a quick introduction to the concept, please refer to the documentation and resources below:

- [Conditional Variational Autoencoders - Understanding cVAEs](https://link-to-cvae-tutorial.com)
- [Deep Learning with PyTorch - Official PyTorch Tutorials](https://pytorch.org/tutorials/)

## Contributing

If you would like to contribute to this project or report issues, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, concise commit messages.
4. Create a pull request to merge your changes into the main branch of this repository.

## License

This project is licensed under the MIT License. For more details, please see the [LICENSE](LICENSE) file.

## Acknowledgments

We would like to acknowledge the contributions of the open-source community, as this project builds upon the work of many researchers and developers in the fields of computer vision, deep learning, and medical imaging.

## Contact

For any questions or inquiries related to this project, feel free to contact the project maintainers:

Thank you for your interest in our project!

**Happy data augmentation and ultrasound foetal echography image analysis!**
