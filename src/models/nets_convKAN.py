import torch.nn as nn
import torch.nn.functional as F

from kan_convolutional import KAN_Convolutional_Layer, KANLinear


# Conv2d + KAN
class ConvKANLinBN_Trainable(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super(ConvKANLinBN_Trainable, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(5)

        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(5)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        self.kan1 = KANLinear(
                980, # 245, # 90
                20,
                grid_size=5,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0, 1],
                base_w_trainable=wb_train,
                spline_w_trainable=ws_train,
                spline_s_trainable=ss_train)
        self.kan2 = KANLinear(
            20, # 245, # 90
            output_size,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
            base_w_trainable=wb_train,
            spline_w_trainable=ws_train,
            spline_s_trainable=ss_train)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    

# Conv2d + KAN
class ConvKANLinBN_Fixed(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super(ConvKANLinBN_Fixed, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        self.kan1 = KANLinear(
                980, # 245, # 90
                20,
                grid_size=5,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0, 1],
                base_w_trainable=wb_train,
                spline_w_trainable=ws_train,
                spline_s_trainable=ss_train)
        self.kan2 = KANLinear(
            20, # 245, # 90
            output_size,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
            base_w_trainable=wb_train,
            spline_w_trainable=ws_train,
            spline_s_trainable=ss_train)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    


# Conv2d + KAN
class ConvKANLinTrainable(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super(ConvKANLinTrainable, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        self.kan1 = KANLinear(
                980, # 245, # 90
                20,
                grid_size=5,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0, 1],
                base_w_trainable=wb_train,
                spline_w_trainable=ws_train,
                spline_s_trainable=ss_train)
        self.kan2 = KANLinear(
            20, # 245, # 90
            output_size,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
            base_w_trainable=wb_train,
            spline_w_trainable=ws_train,
            spline_s_trainable=ss_train)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    
# Conv2d + KAN
class ConvKANLinFixed(nn.Module):
    def __init__(self, strategy, device, wb_train=True, ws_train=True, ss_train=True):
        super(ConvKANLinFixed, self).__init__()
        output_size = 10
        if strategy == "taskIL":
            output_size = 2
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        self.kan1 = KANLinear(
                980, # 245, # 90
                20,
                grid_size=5,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0, 1],
                base_w_trainable=wb_train,
                spline_w_trainable=ws_train,
                spline_s_trainable=ss_train)
        self.kan2 = KANLinear(
            20, # 245, # 90
            output_size,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
            base_w_trainable=wb_train,
            spline_w_trainable=ws_train,
            spline_s_trainable=ss_train)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x