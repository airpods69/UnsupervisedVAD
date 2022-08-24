import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from models.generator import AE
from dataset import *

import matplotlib.pyplot as plt
from tqdm import tqdm


normal_train_dataset = Normal_Loader(is_train=1)
normal_train_loader = DataLoader(normal_train_dataset, batch_size=2, shuffle=True)
print("Normal train data loaded")

normal_test_dataset = Normal_Loader(is_train=0)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=2, shuffle=True)
print("Normal test data loaded")

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=2, shuffle=True)
print("Anomaly train data loaded")

anomaly_test_dataset = Anomaly_Loader(is_train=0)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=2, shuffle=True)
print("Anomaly test data loaded")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Model dec
netG = AE()

# reconstruction loss (MSE)
loss_function = torch.nn.MSELoss()

#RMSprop optimizer with a learning rate of 0.00002,
#momentum 0.60, for 15 epochs on training data with batch size 8192
optimizer = RMSprop(netG.parameters(), lr=0.00002, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.6, centered=False)


epochs = 1
outputs = []
losses = []
for epoch in range(epochs):

    running_loss = 0.

    print("Epoch No.: {}".format(epoch))

    for (image, inputs) in tqdm(normal_train_loader):

        # Output of Autoencoder
        reconstructed = netG.forward(image)

        # Calculating the loss function
        loss = loss_function(reconstructed, image)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss)

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(normal_train_dataset)
    # epoch_acc = running_corrects / len(normal_train_dataset) * 100.
    print("Loss: {}".format(epoch_loss))

    outputs.append((epochs, image, reconstructed))

# Defining the Plot Style
# plt.style.use('fivethirtyeight')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')

print(type(losses))

# Plotting the last 100 values
# print(torch.tensor.detach().numpy().array(losses[-100:]).shape)
losses = [i.item() for i in losses]

# plt.plot(losses[-100:])
# plt.show()

losses = np.array(losses)

np.save("./losses.npy", losses)



















