# import libraries
import datasets.cnn_dataset as cnd
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import time

# Train the model using the cell below. Hyperparameters are given.
# Hyperparameters
# lr = 0.01
# max_epochs=10
# gamma = 0.95
# batch_size=6
exec_start=time.time()
lr=float(sys.argv[1])
max_epochs=int(sys.argv[2])
gamma=float(sys.argv[3])
batch_size=int(sys.argv[4])

# when did we start execution?
print(f'Execution started at:{exec_start}')

# path and device
PATH="YOUR PATH"
path=Path(PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Path.BASE_path= path
print(f'Device being used: {DEVICE}')

# getting data 
training_data=cnd.GalaxyImageDataset(
    annotations_file= (path/"data/train.csv"),
    img_dir=path/"data/train",
    transform=transforms.ToTensor()
)
testing_data=cnd.GalaxyImageDataset(
    annotations_file= (path/"data/test.csv"),
    img_dir=path/"data/test",
    transform=transforms.ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# trying to see if we have the right size
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

from models.cnn_improv_avp import CNN4
input_channels=3
model=CNN4(input_channels).to(DEVICE)

# sanity check
print(model)
from torchsummary import summary
summary(model, (3,424,424))

import trainer
from tqdm import trange
import torch.nn as nn

# Recording data
log_interval = 10

# Instantiate optimizer (model was created in previous cell)
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
train_counter = []
test_losses = []
mse_train=[]
mse_test=[]
for epoch in trange(max_epochs, leave=True, desc='Epochs'):
    epoch_start=time.time()
    train_loss, counter,metrics = trainer.train_one_epoch(train_dataloader, model, DEVICE, optimizer,\
                                                    log_interval, epoch,batch_size,nn.BCEWithLogitsLoss())
    test_loss,mse= trainer.test_one_epoch(test_dataloader, model, DEVICE,loss_func=nn.BCEWithLogitsLoss())
    print(f'Time taken for the epoch:{time.time()-epoch_start}')

    # Record results
    # print(train_loss)
    # print(test_loss)
    train_losses.extend(train_loss)
    train_counter.extend(counter)
    test_losses.append(test_loss)
    mse_train.extend(metrics)
    mse_test.append(mse)

    # print test RMSE 
    print(f"Test RMSE: {mse_test[-1]}")

# print out a prediction and corresposding label
# print('Average train loss:',train_losses/len(train_losses)) #not working
# print('Average test loss:',test_losses/len(test_losses)) #not working
train_features=train_features.to(DEVICE)
train_labels=train_labels.to(DEVICE)

print(nn.Sigmoid()(model(train_features)[1]))
print(train_labels[1])
print(metrics)
print('Final rmse:',mse)
print(f'Total execution time:{time.time()-exec_start}')