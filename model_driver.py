import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torchvision import datasets, transforms
from torch.optim import AdamW, Adam
from torch.cuda.amp import autocast, GradScaler

class ModelDriver:
    def __init__(self, checkpoint_path, model):
        self.checkpoint_path = checkpoint_path
        self.model = model

    def load_checkpoint(self):
        """Load the checkpoint and state_dict."""
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        print("Model loaded successfully.")

    def remove_last_layer_weights(self):
        """
        Reset the weights and biases of the last layer in the model.
        The last layer remains intact, but its parameters are cleared or re-initialized.
        """
        if isinstance(self.model, torch.nn.Module):
            # Identify the last layer
            if hasattr(self.model, "head"):  # Assuming the SwinTransformer has a "head" attribute
                if hasattr(self.model.head, "weight"):
                    self.model.head.weight.data.zero_()  # Reset weights to zero
                    print("Last layer weights reset to zero.")
                if hasattr(self.model.head, "bias") and self.model.head.bias is not None:
                    self.model.head.bias.data.zero_()  # Reset biases to zero
                    print("Last layer biases reset to zero.")
            else:
                raise AttributeError("The model does not have a 'head' attribute.")
        else:
            raise TypeError("Model is not a valid PyTorch nn.Module instance.")
        return self.model