import torch
import torch.optim as optim

from models.discriminator import Discriminator
from models.generator import AE

import numpy as np

input = torch.rand(2048)
np.savetxt('./input.txt', input.detach().numpy())

# Initializing models

img_size = 2048 # Need to change to check length on its self (hardcoded)

netG = AE(img_size) # Generator -> AutoEncoder model
netD = Discriminator() # Discriminator -> Fully Connected Layer


# Training parameters
lr = 0.002 # Learning rate (original is 0.00002)
epochs = 1 # Epochs (original is 8192)

# Optimizers
optimizerG = optim.RMSprop(
        netG.parameters(),
        lr = lr,
        momentum = 0.6,
    )
optimizerD = optim.RMSprop(
        netD.parameters(),
        lr = lr,
        momentum = 0.6,
    )
