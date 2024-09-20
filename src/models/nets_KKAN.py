import torch.nn as nn
import torch.nn.functional as F

from kan_convolutional import KAN_Convolutional_Layer, KANLinear


# KAN_Convolutional_Layer + KAN
class KKAN_BN_Trainable(nn.Module):
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

        self.kan1 = KANLinear(
            625,
            31,
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
            spline_s_trainable=ss_train
        )
        self.kan2 = KANLinear(
            31,
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
            spline_s_trainable=ss_train
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        
        x = self.flat(x)

        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    

# KAN_Convolutional_Layer + KAN
class KKAN_BN_Fixed(nn.Module):
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

        self.kan1 = KANLinear(
            625,
            31,
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
            spline_s_trainable=ss_train
        )
        self.kan2 = KANLinear(
            31,
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
            spline_s_trainable=ss_train
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.flat(x)

        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    

# KAN_Convolutional_Layer + KAN
class KKAN_Trainable(nn.Module):
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

        self.kan1 = KANLinear(
            625,
            31,
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
            spline_s_trainable=ss_train
        )
        self.kan2 = KANLinear(
            31,
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
            spline_s_trainable=ss_train
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.flat(x)

        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    

# KAN_Convolutional_Layer + KAN
class KKAN_Fixed(nn.Module):
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

        self.kan1 = KANLinear(
            625,
            31,
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
            spline_s_trainable=ss_train
        )
        self.kan2 = KANLinear(
            31,
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
            spline_s_trainable=ss_train
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)

        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x