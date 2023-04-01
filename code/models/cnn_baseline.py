# importing some dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    
    def __init__(self,input_channels):
        
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        # ------------------
        # Write your implementation here.
        self.conv1 = nn.Conv2d(input_channels,10,5,padding=(1,1)) # first convolutional layer
        self.conv2 = nn.Conv2d(10,20,5,padding=(1,1)) # second convolutional layer
        self.fc = nn.Linear(20*104*104, 37) # third fully-connected layer
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
        x= self.fc(x)
        y_output = x
        
        return y_output
        # ------------------