import torch
from velocity_dataset import VelocityDataset
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torch.optim import SGD
import numpy as np
from model import MLP
from matplotlib import pyplot as plt

dataset = VelocityDataset('../data/data_y.txt')
dps = dataset.__len__()

input_dim = dataset.__getitem__(1)[0].shape[0]
output_dim = dataset.__getitem__(1)[1].shape[0]

train, test = random_split(dataset, [int(np.floor(dps * 0.9)), int(np.ceil(dps * 0.1))])


train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)

model = MLP(input_dim, 32, output_dim)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.99)

losses = []
num_epochs = 1000
for epoch in range(num_epochs):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
    losses.append(loss.item())
    print('Epoch: ' + str(epoch) + '/' + str(num_epochs)  + '. Loss: ' + str(loss.item()))
        
plt.plot(losses)
plt.savefig('loss.png')