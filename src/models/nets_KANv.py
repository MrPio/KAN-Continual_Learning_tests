import torch
import torch.nn as nn
import torch.nn.functional as F

from kan_convolutional import KAN_Convolutional_Layer

# KAN_Convolutional_Layer + MLP
class KANvNet_BN_Trainable(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super().__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )
        self.bn1 = nn.BatchNorm2d(5)

        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )
        self.bn2 = nn.BatchNorm2d(25)

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)

        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

# KAN_Convolutional_Layer + MLP
class KANvNet_BN_Fixed(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super().__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )
        self.bn1 = nn.BatchNorm2d(5)

        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )
        self.bn2 = nn.BatchNorm2d(25)

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)

        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x
    

# KAN_Convolutional_Layer + MLP (without Batch Norm)
class KANvNet_Trainable(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super().__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
            
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    
# KAN_Convolutional_Layer + MLP (without Batch Norm)
class KANvNet_Fixed(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super().__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device,
            base_w_trainable=wb_train,
            spline_s_trainable=ss_train,
            spline_w_trainable=ws_train
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x