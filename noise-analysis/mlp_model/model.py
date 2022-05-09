import torch
import torch.nn.functional as F
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            output = self.out(x)
            return output