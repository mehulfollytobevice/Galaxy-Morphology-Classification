# importing some dependencies
import torch

hidden_size = 128         # width of hidden layer
output_size = 37          # number of output neurons

class FNN(torch.nn.Module):
    
    def __init__(self, input_size):
        
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.act = torch.nn.functional.relu
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.flatten(x)

        y_output = self.act(self.fc1(x))
        y_output = self.fc2(y_output)
        return y_output
