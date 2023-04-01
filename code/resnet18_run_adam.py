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
image_size=int(sys.argv[5])

# when did we start execution?
print(f'Execution started at:{exec_start}')

# path and device
PATH="YOUR PATH"
path=Path(PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Path.BASE_path= path
print(f'Device being used: {DEVICE}')

# getting data 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size,image_size)),
])
training_data=cnd.GalaxyImageDataset(
    annotations_file= (path/"data/train.csv"),
    img_dir=path/"data/train",
    transform=transform
)
testing_data=cnd.GalaxyImageDataset(
    annotations_file= (path/"data/test.csv"),
    img_dir=path/"data/test",
    transform=transform
)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# trying to see if we have the right size
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
input_channels=3
model=resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc=nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=512, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1), 
        nn.Dropout(p=0.5), 
        nn.Linear(in_features=2048, out_features=37)
    )

# sanity check
print(model)
model.cuda()
from torchsummary import summary
summary(model, (3,image_size,image_size))

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

# save the model for inference later
torch.save(model.state_dict(),path/"trained_models/resnet18.pt")