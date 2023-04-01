# importing some dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# In cnn_baseline.py, we implemented a basic 2 layer conv network.
# Now, in this file we are going to improve the Conv network
# by adding a few things:
# 1. Batch Norm
# 2. Adaptive average pooling

class CNN4(torch.nn.Module):
    
    def __init__(self,input_channels):
        
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        # ------------------
        # Write your implementation here.
        
        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                                out_channels=16,
                                kernel_size=5,
                                stride=2,
                                padding=1)

        # Batch Norm 1st layer
        self.bn1=nn.BatchNorm2d(16)
        self.dp1=nn.Dropout2d(0.3)

        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=5,
                                stride=2,
                                padding=1)

        # Batch Norm 2nd layer
        self.bn2=nn.BatchNorm2d(32)
        self.dp2=nn.Dropout2d(0.3)

        # third convolutional layer                       
        self.conv3= nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        # Batch Norm 3rd layer 
        self.bn3=nn.BatchNorm2d(64)
        self.dp3=nn.Dropout2d(0.5)

        # fourth convolutional layer                       
        self.conv4= nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.bn4=nn.BatchNorm2d(128)
        self.dp4=nn.Dropout2d(0.5)

        self.fc = nn.Linear(128, 37) # first fully-connected layer

        # ------------------
    
    def forward(self, x):

        # ------------------
        # Write your implementation here. 
        # convolutional layer
        x= F.max_pool2d(F.relu(self.bn1(self.conv1(x))),(2,2))
        # print(x.shape)

        x= F.max_pool2d(F.relu(self.bn2(self.conv2(x))),(2,2))
        # print(x.shape)

        x= F.max_pool2d(F.relu(self.bn3(self.conv3(x))),(2,2))
        x=self.dp3(x)
        # print(x.shape)

        x= F.relu(self.bn4(self.conv4(x)))
        x=self.dp4(x)
        # print(x.shape)

        # adaptive average pooling
        x=F.adaptive_avg_pool2d(x,1)
        # print(x.shape)

        x= self.flatten(x)
        # print(x.shape)

        # first fc layer
        y_output=self.fc(x)
        
        return y_output
        # ------------------

if __name__=="__main__":
    from torchsummary import summary
    model=CNN4(input_channels=3)
    summary(model, (3,424,424))