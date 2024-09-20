import os
import math
import torch
import torch.optim as optim

from nets_conv import *
from nets_KANv import *
from nets_KKAN import *
from nets_convKAN import *
from train_functs import train, onEpochEnd
from dataset import train_loader_tasks, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_path = os.path.join('/mnt/c/Users/aless/Desktop/pykan/KAN-Continual_Learning', 'results', 'mnist', 'convs')
strategy = "classIL"     # ["taskIL", "classIL"] classIL è quello tosto
lrs = [1e-5]
decays = [1]
epochss = [10]
models = [
    # ConvNet(strategy, device), ConvNet_BN(strategy, device),
    # KKAN_Fixed(strategy, device), 
    # KKAN_BN_Fixed(strategy, device), KKAN_Trainable(strategy, device), KKAN_BN_Trainable(strategy, device),
    KANvNet_Fixed(strategy, device), KANvNet_BN_Fixed(strategy, device), KANvNet_BN_Trainable(strategy, device), KANvNet_Trainable(strategy, device),
    ConvKANLinFixed(strategy, device), ConvKANLinBN_Fixed(strategy, device), ConvKANLinTrainable(strategy, device), ConvKANLinBN_Trainable(strategy, device)
    ]
particolari = ""
reverse_taks = False

reverse_path = ""
if reverse_taks:
    reverse_path = "reverse_tasks"
    train_loader_tasks.reverse()

out_path = os.path.join(out_path, strategy, particolari, reverse_path)
cfgs = []
for model in models:
    for lr in lrs:
        for decay in decays:
            for epochs in epochss:
                cfgs.append([model, epochs, lr, decay])

for cfg in cfgs:
    model = cfg[0]
    epochs = cfg[1]
    lr = cfg[2]
    decay_f = cfg[3]
    if decay_f == 1:
        lr_decay = False
    else:
        lr_decay = True
    start_epochs_list = [int(epochs + epochs*i[0]) for i in enumerate(train_loader_tasks)]#+start_epochs_list[-1])
    start_epochs_list.insert(0, 0)
    naam = model.__class__.__name__
    isKAN = False
    print("\n\n", naam)
    print(epochs, lr, decay_f, "\n")
    if 'Py_KAN' in naam:
        isKAN = True
    for i, task in enumerate(train_loader_tasks):
        epochs_act = epochs
        if particolari == "long_last_tasks" and i > 3:
            epochs_act = epochs + epochs

        str_print = f'\t\t\t\tTRAINING ON TASK {i}'
        str_print +=  f' for {epochs_act} epochs' 
        print(str_print)
        # str_epoch = f"ep{epochs}_10fin_"
        str_epoch = f"ep{epochs}"
        str_lr = f"_lr{round(math.log10(lr))}"
        str_decay = '_dec'+ str(decay_f) if lr_decay else ''
        lr_act = lr * decay_f**(i)
        train(model, task, test_loader, strategy, os.path.join(out_path,f"{str_epoch}{str_lr}{str_decay}", naam), device=device,
                epochs=epochs_act, start_epoch=start_epochs_list[i], optimizer=optim.Adam(model.parameters(), lr=lr_act),
                on_epoch_end=onEpochEnd, lr=lr, isKAN=isKAN)
    torch.cuda.empty_cache()