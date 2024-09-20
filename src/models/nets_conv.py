import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv2d + MLP + (Dropout)
class ConvNet(nn.Module):
    def __init__(self, strategy, device):
        super(ConvNet, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [1, 980, 161, output_size]

        self.conv1 = nn.Conv2d(self.layers[0], 5, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5, padding='same')

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.layers[1], self.layers[2])
        self.fc2 = nn.Linear(self.layers[2], self.layers[3])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.dropout3(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x
    

# Conv2d + MLP + (Dropout)
class ConvNet_BN(nn.Module):
    def __init__(self, strategy, device):
        super(ConvNet_BN, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.layers = [1, 980, 161, output_size]

        self.conv1 = nn.Conv2d(self.layers[0], 5, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(5)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.layers[1], self.layers[2])
        self.fc2 = nn.Linear(self.layers[2], self.layers[3])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.dropout3(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x