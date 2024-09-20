import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score
from utils.data_utils import prepare_data
from pdb import set_trace as bp


class ResidualAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, hidden_dims, activation_fn, use_residual, 
                 identity_if_no_compression=False, use_batch_norm=False, use_layer_norm=False,
                 real_weight=1.0,
                 binary_weight=1.0,
                 dropout_rate=0.3,
                 learning_rate=1e-4,
                 max_epochs=10,
                 log_plots=True,
                 log_grads=False,
                 **kwargs):
        super(ResidualAutoencoder, self).__init__()

        # print the kwargs and point out that they will be ignored
        print(f"kwargs: {kwargs}")
        print("Note: The kwargs will be ignored in this model, but were hopefully used elsewhere.")

        self.log_plots = log_plots
        self.log_grads = log_grads
        self.use_residual = use_residual
        self.identity_if_no_compression = identity_if_no_compression
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.real_weight = real_weight
        self.binary_weight = binary_weight
        # recommend to rescale MSE_loss to be of similar magnitude to BCE_loss
        # e.g. mse_loss_weight = 8.32 allows losses to be equal
        # also need to account for the number of features in each group
        # SO: total_loss = real_weight * MSE_loss + binary_weight * BCE_loss
        # where real_weight = 8.32 * num_real_features / num_total_features
        # and binary_weight = num_binary_features / num_total_features

        # Ensure that BatchNorm and LayerNorm are mutually exclusive
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("You can only use one of BatchNorm or LayerNorm, not both.")

        self.return_identity = self.latent_dim == self.input_dim and self.identity_if_no_compression

        if self.return_identity:
            self.scalar_param = nn.Parameter(torch.ones(1))
        else:
            # Encoder with optional residual connections
            self.encoder_layers = nn.ModuleList()
            self.encoder_projections = nn.ModuleList()

            # First layer (input to first hidden)
            self.encoder_layers.append(self._create_layer(input_dim, hidden_dims[0], activation_fn))
            if self.use_residual:
                self.encoder_projections.append(
                    nn.Linear(input_dim, hidden_dims[0]) if input_dim != hidden_dims[0] else nn.Identity()
                )

            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                self.encoder_layers.append(self._create_layer(hidden_dims[i], hidden_dims[i + 1], activation_fn))
                if self.use_residual:
                    self.encoder_projections.append(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if hidden_dims[i] != hidden_dims[i + 1] else nn.Identity()
                    )

            # Latent layer
            self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)

            # Decoder with optional residual connections
            self.decoder_layers = nn.ModuleList()
            self.decoder_projections = nn.ModuleList()

            # First layer (latent to first hidden)
            self.decoder_layers.append(self._create_layer(latent_dim, hidden_dims[-1], activation_fn))
            if self.use_residual:
                self.decoder_projections.append(
                    nn.Linear(latent_dim, hidden_dims[-1]) if latent_dim != hidden_dims[-1] else nn.Identity()
                )

            # Hidden layers
            for i in range(len(hidden_dims) - 1, 0, -1):
                self.decoder_layers.append(self._create_layer(hidden_dims[i], hidden_dims[i - 1], activation_fn))
                if self.use_residual:
                    self.decoder_projections.append(
                        nn.Linear(hidden_dims[i], hidden_dims[i - 1]) if hidden_dims[i] != hidden_dims[i - 1] else nn.Identity()
                    )

            # Output layer
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[0], input_dim),
                nn.Sigmoid()  # Use Sigmoid since data is normalized between [0,1]
            )

    def _create_layer(self, in_dim, out_dim, activation_fn):
        """Helper method to create a single layer with optional BatchNorm or LayerNorm."""
        layers = [nn.Linear(in_dim, out_dim)]

        # Apply BatchNorm or LayerNorm based on the flags
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        elif self.use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))

        layers.append(activation_fn())  # Add the activation function

        # Add dropout layer
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x_binary, x_real):
        # Concatenate binary and real data before encoding
        x = torch.cat([x_binary, x_real], dim=1)

        if self.return_identity:
            # Identity mapping case
            reconstructed = x * self.scalar_param  # Return input directly as reconstruction
        else:
            # Standard encoder-decoder with residuals
            x_encoded = self.encode(x)
            reconstructed = self.decode(x_encoded)

        # Split the reconstructed output back into binary and real parts
        x_hat_binary = reconstructed[:, :x_binary.shape[1]]
        x_hat_real = reconstructed[:, x_binary.shape[1]:]

        return x_hat_binary, x_hat_real

    def encode(self, x):
        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            identity = self.encoder_projections[i](x) if self.use_residual else 0
            x = layer(x) + identity
        return self.latent_layer(x)

    def decode(self, x):
        # Pass through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            identity = self.decoder_projections[i](x) if self.use_residual else 0
            x = layer(x) + identity
        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        x_binary, x_real, mask_binary, mask_real = batch
        x_hat_binary, x_hat_real = self(x_binary, x_real)
        loss = self.compute_loss(x_binary, x_real,
                x_hat_binary, x_hat_real,
                mask_binary, mask_real,
                stage="train")

        # Log the true and reconstructed data as images to wandb
        if self.current_epoch % 50 == 0 and batch_idx == 0:  # Log only once per epoch
            # log binary images
            self.log_images_to_wandb(x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        x_binary, x_real, mask_binary, mask_real = batch
        x_hat_binary, x_hat_real = self(x_binary, x_real)
        loss = self.compute_loss(x_binary, x_real,
                x_hat_binary, x_hat_real,
                mask_binary, mask_real,
                stage="val")

        # Log the true and reconstructed data as images to wandb
        if self.current_epoch % 50 == 0 and batch_idx == 0:  # Log only once per epoch
            # log binary images
            self.log_images_to_wandb(x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, 'val')

    def configure_optimizers(self):
        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define the number of warmup steps
        num_warmup_steps = 50  # Adjust this based on your needs

        # Define the warmup schedule using LambdaLR
        def warmup_scheduler_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0  # Return the full learning rate after warmup period

        # LambdaLR for warmup
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler_lambda)

        # Return the optimizer and the warmup scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,  # Warmup scheduler
                "interval": "step",  # Apply the scheduler per step (batch)
                "frequency": 1,  # Apply it at every step
                "name": "warmup_lr_scheduler"  # Give a name for tracking in WandB
            },
        }

    def compute_loss(self, x_binary, x_real, x_hat_binary, x_hat_real, mask_binary, mask_real, stage):

        # MSE Loss for real-valued columns
        mse_loss = F.mse_loss(x_hat_real, x_real, reduction='none')
        mse_loss = (mse_loss * mask_real).mean()

        # Binary Cross-Entropy Loss for binary columns
        bce_loss = F.binary_cross_entropy(x_hat_binary, x_binary, reduction='none')
        bce_loss = (bce_loss * mask_binary).mean()

        # Combine losses
        total_loss = self.real_weight * mse_loss + self.binary_weight * bce_loss

        # a note that a random prediction (0.5) would have a BCE loss of 0.69
        # for real-valued data, unclear what the MSE loss would be for 0.5,
        # as it depends on the distribution of the data.
        # if the data are uniformly distributed between 0 and 1, then the MSE loss for 0.5
        # would be the variance of the distribution, which is 1/12 = 0.0833.
        # This would suggest a rescaling of:
        # total_loss = lambda * MSE_loss + BCE_loss
        # where lambda = 0.69 / 0.0833 = 8.32

        # Log losses to wandb
        self.log(f"{stage}_mse_loss", mse_loss.item())
        self.log(f"{stage}_bce_loss", bce_loss.item())
        self.log(f"{stage}_loss", total_loss.item())

        # compute AUROC for binary columns
        # Compute AUROC only on valid data (non-missing values where mask == 1)
        x_binary_cpu = x_binary.detach().cpu().numpy()
        x_hat_binary_cpu = (
            x_hat_binary.detach().cpu().numpy()
        )  # No need for additional sigmoid
        mask_binary_cpu = mask_binary.detach().cpu().numpy()

        # try:
        # Compute AUROC for each binary variable (mask out missing values)
        auroc_per_label = [
            roc_auc_score(
                x_binary_cpu[mask_binary_cpu[:, i] == 1, i],
                x_hat_binary_cpu[mask_binary_cpu[:, i] == 1, i],
            )
            for i in range(x_binary_cpu.shape[1])
            if len(np.unique(x_binary_cpu[mask_binary_cpu[:, i] == 1, i]))
            > 1  # Only compute if variance exists
        ]

        # Macro-Averaged AUROC
        if len(auroc_per_label) > 0:
            macro_auroc = np.mean(auroc_per_label)
        else:
            macro_auroc = float("nan")  # Handle cases where no AUROC could be computed

        # Flattened true and predicted arrays for micro-averaged AUROC
        y_true_flat = x_binary_cpu[mask_binary_cpu == 1].ravel()
        y_pred_flat = x_hat_binary_cpu[mask_binary_cpu == 1].ravel()

        if len(np.unique(y_true_flat)) > 1:  # Ensure there's variance in flattened data
            micro_auroc = roc_auc_score(y_true_flat, y_pred_flat)
        else:
            micro_auroc = float("nan")

        # Log AUROCs to wandb
        self.log(f"{stage}_macro_auroc", macro_auroc)
        self.log(f"{stage}_micro_auroc", micro_auroc)

        # except Exception as e:
        #     bp()
        #     print(f"Error calculating AUROC: {e}")

        return total_loss

    def log_gradient_stats(self):
        """
        Logs gradient statistics (mean, max, min) for each layer in the model.
        """
        gradient_stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradient_stats[f"grad_mean/{name}"] = param.grad.mean().item()
                gradient_stats[f"grad_max/{name}"] = param.grad.max().item()
                gradient_stats[f"grad_min/{name}"] = param.grad.min().item()

        # Log gradient stats to wandb
        wandb.log(gradient_stats)

    def on_after_backward(self):
        """Hook to log gradient stats after the backward pass."""

        if self.log_grads and self.current_epoch % 10 == 0:
            self.log_gradient_stats()

    def log_images_to_wandb(self, x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, stage):
        """
        Create imshow-style plots of the true x_binary and x_real vs reconstructed x_hat_binary and x_hat_real,
        with the mask applied as black color. Log them as a single figure to wandb.
        
        Args:
            x_binary (torch.Tensor): The true binary data.
            x_hat_binary (torch.Tensor): The reconstructed binary data.
            x_real (torch.Tensor): The true real-valued data.
            x_hat_real (torch.Tensor): The reconstructed real-valued data.
            mask_binary (torch.Tensor): The mask applied to binary data.
            mask_real (torch.Tensor): The mask applied to real-valued data.
            stage (str): Either 'train' or 'val', indicating the current stage of training.
        """

        if not self.log_plots:
            return

        # Convert to NumPy for plotting
        x_binary = x_binary.cpu().detach().numpy()
        x_hat_binary = x_hat_binary.cpu().detach().numpy()
        # convert to binary by thresholding at 0.5
        x_hat_binary_thresholded = (x_hat_binary > 0.5).astype(float)

        x_real = x_real.cpu().detach().numpy()
        x_hat_real = x_hat_real.cpu().detach().numpy()
        mask_binary = mask_binary.cpu().detach().numpy()
        mask_real = mask_real.cpu().detach().numpy()

        # Create a figure with 4 subplots: true and reconstructed for binary and real
        fig, ax = plt.subplots(3, 2, figsize=(18, 12))

        # Blue-White-Red colormap with black for masked regions
        cmap = plt.cm.bwr
        cmap.set_bad(color='black')

        # Define a function to replace masked areas with black
        def apply_mask(data, mask, vmin, vmax):
            masked_data = np.where(mask == 0, np.nan, data)  # Replace masked values with NaN
            return masked_data

        # Plot true binary data with mask applied
        true_binary_with_mask = apply_mask(x_binary, mask_binary, vmin=0, vmax=1)
        my_ax = ax[0, 0]
        my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"True Binary Data ({stage})")
        my_ax.set_xlabel("Features")
        my_ax.set_ylabel("Samples")
        plt.colorbar(my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot reconstructed binary data with mask applied
        my_ax = ax[0, 1]
        reconstructed_logit_with_mask = apply_mask(x_hat_binary, mask_binary, vmin=0, vmax=1)
        my_ax.imshow(reconstructed_logit_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"Reconstructed Logit Data ({stage})")
        my_ax.set_xlabel("Features")
        plt.colorbar(my_ax.imshow(reconstructed_logit_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot true binary data with mask applied
        my_ax = ax[1, 0]
        true_binary_with_mask = apply_mask(x_binary, mask_binary, vmin=0, vmax=1)
        my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"True Binary Data ({stage})")
        my_ax.set_xlabel("Features")
        my_ax.set_ylabel("Samples")
        plt.colorbar(my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot reconstructed binary data with mask applied
        my_ax = ax[1, 1]
        reconstructed_binary_with_mask = apply_mask(x_hat_binary_thresholded, mask_binary, vmin=0, vmax=1)
        my_ax.imshow(reconstructed_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"Reconstructed Binary Data ({stage})")
        my_ax.set_xlabel("Features")
        plt.colorbar(my_ax.imshow(reconstructed_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)


        # Plot true real data with mask applied
        my_ax = ax[2, 0]
        true_real_with_mask = apply_mask(x_real, mask_real, vmin=0, vmax=1)
        my_ax.imshow(true_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"True Real Data ({stage})")
        my_ax.set_xlabel("Features")
        my_ax.set_ylabel("Samples")
        plt.colorbar(my_ax.imshow(true_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot reconstructed real data with mask applied
        my_ax = ax[2, 1]
        reconstructed_real_with_mask = apply_mask(x_hat_real, mask_real, vmin=0, vmax=1)
        my_ax.imshow(reconstructed_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"Reconstructed Real Data ({stage})")
        my_ax.set_xlabel("Features")
        plt.colorbar(my_ax.imshow(reconstructed_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Adjust layout and log the figure to wandb
        plt.tight_layout()
        wandb.log({f"{stage}_reconstruction": wandb.Image(fig)})

        # Close the figure to free up memory
        plt.close(fig)


# 4. Training setup function with wandb logging
def train_model(model, train_loader, test_loader, config):

    # Detect device: MPS for macOS, GPU for other systems if available
    if torch.backends.mps.is_available():
        config["accelerator"] = "mps"
    elif torch.cuda.is_available():
        config["accelerator"] = "gpu"
    else:
        config["accelerator"] = "cpu"

    output_dir = "outputs"
    os.makedirs(f"{output_dir}", exist_ok=True)

    wandb_logger = WandbLogger(project="autoencoder_project", save_dir=output_dir)

    # Log the hyperparameters to wandb
    wandb_logger.experiment.config.update(config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=model.max_epochs,
        # gradient_clip_val=0.5,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),  # was "epoch"
            # pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{output_dir}/checkpoints/{wandb.run.name}",  # Directory to save checkpoints
                filename="{epoch}-{val_loss:.2f}",  # Naming pattern for checkpoint files
                save_top_k=1,  # Save top 3 checkpoints
                monitor="val_loss",  # Monitor validation loss
                mode="min",  # Save when validation loss decreases
            ),
        ],
        accelerator=config["accelerator"],  # mps, gpu, cpu
        # can configure devices= if desired to choose specific GPUs
    )
    trainer.fit(model, train_loader, test_loader)


# 5. Main function to run the script
if __name__ == "__main__":

    # preliminary setup
    batch_size = 2048 # setting to None uses full dataset
    shuffle_train = True

    # Prepare data
    # filename = "data/combined_data_subset_1_percent.csv"
    # filename = "data/combined_data_subset_10_percent.csv"
    filename = "data/combined_data.csv"

    train_loader, test_loader = prepare_data(filename, batch_size, shuffle_train)

    # determine MSE vs BCE loss weights by counting number of features (and possibly doing further re-weighting)
    num_binary_features = train_loader.dataset.binary_data.shape[1]
    num_real_features = train_loader.dataset.real_data.shape[1]
    num_total_features = num_binary_features + num_real_features
    print(f"Input dimension: {num_total_features}")

    binary_weight = num_binary_features / num_total_features
    real_weight = 8.3 * num_real_features / num_total_features

    # then, further upweight the MSE loss to be of similar magnitude to BCE loss
    # e.g. mse_loss_weight = 8.3 allows losses to be equal when
    # the data are uniformly distributed between 0 and 1
    # and the BCE loss is 0.69 for a random prediction of 0.5
    # whereas the MSE loss for 0.5 would be 1/12 = 0.0833
    # their ratio is 0.69 / 0.0833 = 8.3

    # total_loss = real_weight * MSE_loss + binary_weight * BCE_loss

    # Define configuration for wandb
    config = {
        "input_dim": num_total_features,
        "latent_dim": 1000,
        "hidden_dims": [2000, 1500],
        "activation_fn": nn.GELU,
        "use_residual": True,
        "identity_if_no_compression": False,
        "use_batch_norm": True,
        "use_layer_norm": False,
        "learning_rate": 1e-4,
        "dropout_rate": 0.0,
        "batch_size": batch_size,  # None uses full dataset
        "shuffle_train": shuffle_train,
        "max_epochs": 1000,
        "real_weight": real_weight,
        "binary_weight": binary_weight,
        "log_plots": True,
        "log_grads": False,
    }

    # Initialize model
    model = ResidualAutoencoder(**config)

    # Train model
    train_model(model, train_loader, test_loader, config)

    '''
    NOTE:
    With latent_dim==input_dim and GELU activation, we are able to get VERY close to identity mapping.
    A few comments:
        1. We are using residual connections (Projection based) and batch norm; can try without these.
        2. Needed to run for MANY epochs (set to 1000).

    '''
