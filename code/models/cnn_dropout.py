# importing some dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# In cnn_baseline.py, we implemented a basic 2 layer conv network.
# Now, in this file we are going to improve the Conv network
# by adding a few things:
# 1. Dropout
# 2. Two more fully connected layers

class CNN2(torch.nn.Module):
    
    def __init__(self,input_channels):
        
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        # ------------------
        # Write your implementation here.
        self.conv1 = nn.Conv2d(input_channels,10,5,padding=(1,1)) # first convolutional layer
        self.conv2 = nn.Conv2d(10,20,5,padding=(1,1)) # second convolutional layer
        self.fc = nn.Linear(20*104*104, 4096) # first fully-connected layer
        self.fc2 = nn.Linear(4096, 2048) # second fully-connected layer
        self.fc3 = nn.Linear(2048, 37) # third fully-connected layer
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2=nn.Dropout(0.5)

        # ------------------
    
    def forward(self, x):
        # Input image is of shape [batch_size, 1, 28, 28]
        # Need to flatten to [batch_size, 784] before feeding to fc1

        # ------------------
        # Write your implementation here. 
        x= F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # print(x.shape)
        x= F.max_pool2d(F.relu(self.conv2(x)),(2,2))

        # print(x.shape)
        x= self.flatten(x)
        # print(x.shape)

        # first fc layer
        x=F.relu(self.fc(x))
        x=self.dropout1(x)
        
        # second fc layer
        x=F.relu(self.fc2(x))
        x=self.dropout2(x)

        #give output
        y_output = self.fc3(x)
        
        return y_output
        # ------------------