# Autoencoder-based Dimensionality Reduction for EMOD data

This project performs dimensionality reduction (via autoencoder) on data from the EMOD project, which is a simulation of infectious disease spread in a population. The data is a mix of real-valued and binary features, and the goal is to reduce the dimensionality of the data while preserving the most important information.

# Setup Instructions

## Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/emod_dim_red.git
cd emod_dim_red
```

## Create the Conda Environment:

```bash
# The environment.yml file contains all the necessary dependencies to run this project. 
# To create the environment, use the following command:
conda env create -f environment.yml

# This will create a Conda environment with all the specified dependencies.

# 3. Activate the Environment

# After the environment is created, activate it using:
conda activate autoencoder-env
```


## Run the main script
```python autoencoder_pipeline.py```
