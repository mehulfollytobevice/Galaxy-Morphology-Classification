# importing some dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# In cnn_baseline.py, we implemented a basic 2 layer conv network.
# Now, in this file we are going to improve the Conv network
# by adding a few things:
# 1. Dropout
# 2. Two more fully connected layers

class CNN3(torch.nn.Module):
    
    def __init__(self,input_channels):
        
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        # ------------------
        # Write your implementation here.
        
        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                                out_channels=10,
                                kernel_size=5,
                                stride=2,
                                padding=1)

        # Batch Norm 1st layer
        self.bn1=nn.BatchNorm2d(10)

        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=10,
                                out_channels=20,
                                kernel_size=5,
                                stride=2,
                                padding=1)

        # Batch Norm 2nd layer
        self.bn2=nn.BatchNorm2d(20)

        # third convolutional layer                       
        self.conv3= nn.Conv2d(in_channels=20,
                                out_channels=50,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        # Batch Norm 3rd layer 
        self.bn3=nn.BatchNorm2d(50)

        self.fc = nn.Linear(50*13*13, 256) # first fully-connected layer
        self.fc2 = nn.Linear(256, 128) # second fully-connected layer
        self.fc3 = nn.Linear(128, 37) # third fully-connected layer
        self.dense_bn=nn.BatchNorm1d(256)
        self.dense_bn2=nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2=nn.Dropout(0.5)

        # ------------------
    
    def forward(self, x):

        # ------------------
        # Write your implementation here. 
        x= F.max_pool2d(F.relu(self.bn1(self.conv1(x))),(2,2))
        #print(x.shape)

        x= F.max_pool2d(F.relu(self.bn2(self.conv2(x))),(2,2))
        #print(x.shape)

        x= F.max_pool2d(F.relu(self.bn3(self.conv3(x))),(2,2))
        #print(x.shape)
        
        x= self.flatten(x)
        # print(x.shape)

        # first fc layer
        x=F.relu(self.dense_bn(self.fc(x)))
        x=self.dropout1(x)
        
        # second fc layer
        x=F.relu(self.dense_bn2(self.fc2(x)))
        x=self.dropout2(x)

        #give output
        y_output = self.fc3(x)
        
        return y_output
        # ------------------

if __name__=="__main__":
    from torchsummary import summary
    model=CNN3(input_channels=3)
    summary(model, (3,424,424))