# Weights & Biases Tutorial

Introduction to Weights and Biases I held at our research group. Covers the subset of wandb that I use in my daily research.
Descriptions of the covered topics can be found in the uploaded PDF file.
If you spot any errors, feel free to raise an issue in the project :)

## Part 0: Setup

1. Create a wandb account and log in with `wandb login`.
2. Install the provided mamba environment using `mamba env create -f environment.yml`.
3. Run the code. Information about the used features can be found in the PDF.

The example project can be found [here](https://wandb.ai/timo_kk/Wandb%20Tutorial).

## Part 1: Basics

- Initializing wandb
- Using wandb log to sync data to the cloud
- Most common CLI commands
- Debugging with wandb
- Basic dashboard functionality
- Logging various types of data
- Saving model parameters or other arbitrary files in the cloud

## Part 2: "Advanced"

- Logging using multiple x-axis
- Efficiently scraping data from the cloud
- Using wandb sweep to perform hyperparameter optimization
- wandb in code with multiple processes
- Creating reports (not advanced, but takes some time)

## Credits

- [Weights & Biases](https://wandb.ai/site)
- [Pytorch beta-VAE example](https://github.com/pytorch/examples/tree/main/vae)
