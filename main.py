import torch
from models.discriminator import Discriminator
from models.generator import AE
import numpy as np



input = torch.rand(2048)
np.savetxt('./input.txt', input.detach().numpy())


