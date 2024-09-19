import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_mask(mask, title="Mask Visualization"):
    """
    Visualize a 2D mask as an image where 1s are different from 0s.
    
    Args:
    - mask (np.ndarray or torch.Tensor): The mask to visualize, assumed to be 2D (batch size x features).
    - title (str): The title of the plot.
    """
    # If the mask is a torch tensor, convert it to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Plot the mask using matplotlib's imshow
    plt.figure(figsize=(10, 6))
    plt.imshow(mask, cmap='gray', aspect='auto', interpolation='nearest')
    plt.colorbar(label="Mask Value (1=Available, 0=Missing)")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.show()

def plot_missingness(data):
    # --- eCDF Plot for Missingness ---
    # Calculate percentage of missing values for each column
    missing_counts = data.isna().mean()
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(missing_counts, bins=30, edgecolor='black')
    plt.xlabel("Number of Missing Values per Column")
    plt.ylabel("Count of Columns")
    plt.title("Histogram of Feature Missingness Counts")
    plt.grid(True)
    plt.show()
