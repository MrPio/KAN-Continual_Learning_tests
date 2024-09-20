import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_kan.kan import KAN
from kan.KAN_batch import KAN as PyKAN


# MLP
class MLP(nn.Module):
    def __init__(self, strategy, device):
        super(MLP, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 28 * 28, 256, output_size]
        self.fc1 = nn.Linear(self.layers[0], self.layers[1])
        self.fc2 = nn.Linear(self.layers[1], self.layers[2])
        self.fc3 = nn.Linear(self.layers[2], self.layers[3])

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        # x = (x / 0.5 - 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class MLPBig(nn.Module):
    def __init__(self, strategy, device):
        super(MLPBig, self).__init__()
        self.fc1 = nn.Linear(28*28, 28*28)
        self.fc2 = nn.Linear(28*28, 285)
        self.fc3 = nn.Linear(285, 256)
        if strategy == "taskIL":
            self.fc4 = nn.Linear(256, 2)
        else:
            self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        # x = (x / 0.5 - 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x
    
# KAN
class Efficient_KAN(nn.Module):
    def __init__(self, strategy, device, grid=5):
        super(Efficient_KAN, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 128, output_size]
        self.model = KAN(self.layers, grid_size=grid, sb_trainable=True, sp_trainable=True).to(device)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
    
# KAN
class Efficient_KAN_Fix(nn.Module):
    def __init__(self, strategy, device, grid=5):
        super(Efficient_KAN_Fix, self).__init__()
        # self.layers = [input_size, 103, output_size] if dataset == datasets.MNIST \
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 128, output_size]
        self.model = KAN(self.layers, grid_size=grid, sb_trainable=False, sp_trainable=True).to(device)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

# KAN
class Efficient_KAN_Fixall(nn.Module):
    def __init__(self, strategy, device, grid=5):
        super(Efficient_KAN_Fixall, self).__init__()
        # self.layers = [input_size, 103, output_size] if dataset == datasets.MNIST \
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 128, output_size] 
        self.model = KAN(self.layers, grid_size=grid, sb_trainable=False, sp_trainable=False).to(device)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class Py_KAN_Fix(nn.Module):
    def __init__(self, strategy, device, grid=5):
        super(Py_KAN_Fix, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 128, output_size]
        self.model = PyKAN(self.layers, grid=grid, device=device, sb_trainable=False,
                           sp_trainable=True, bias_trainable=False, symbolic_enabled=False).to(device)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class Py_KAN(nn.Module):
    def __init__(self, strategy, device, grid=5):
        super(Py_KAN, self).__init__()
        # self.layers = [input_size, 73, 10] if dataset == datasets.MNIST \
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 128, output_size]
        self.model = PyKAN(self.layers, grid=grid, device=device, sb_trainable=True,
                           sp_trainable=False, bias_trainable=False, symbolic_enabled=False).to(device)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class Py_KAN_smallFix(nn.Module):
    def __init__(self, strategy, device, grid=5):
        super(Py_KAN_smallFix, self).__init__()
        # self.layers = [input_size, 73, 10] if dataset == datasets.MNIST \
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [28*28, 28, output_size]
        self.model = PyKAN(self.layers, grid=grid, device=device, sb_trainable=True,
                           sp_trainable=False, bias_trainable=False, symbolic_enabled=False).to(device)

    def forward(self, x):
        x = x.view(-1, self.layers[0])
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x