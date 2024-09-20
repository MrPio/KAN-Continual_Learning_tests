import os
import time
import torch
import pickle
import torch.nn as nn

from tqdm import tqdm


def train(model, train_loader, test_loader, strategy, save_dir, optimizer, device, start_epoch=0, epochs=5,
          on_epoch_end=None, lr=0, task_id=None, isKAN=False):
    criterion = nn.NLLLoss()
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)
    for epoch in range(start_epoch, epochs + start_epoch):
        if not isKAN:
            model.train()
        model.to(device)
        epoch_start = time.time_ns()
        with tqdm(train_loader) as pbar:
            for images, labels in pbar:
                labels = labels.to(device)  #(labels % 2 if task_id is not None else labels).to(device)
                images = images.to(device)
                optimizer.zero_grad()
                output = model(images)
                if strategy == "taskIL":
                    labels = labels % 2
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step(closure=lambda: loss)
                accuracy = (output.argmax(dim=1) == labels).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
                # scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        epoch_duration = (time.time_ns() - epoch_start) // 1000000
        if on_epoch_end is not None:
            on_epoch_end(model, test_loader, strategy, device, save_dir, epoch, loss.item(), epoch_duration, lr, task_id, isKAN)
        # torch.save(model.state_dict(), f'{checkpoint}_ep{epoch + 1}.pth')


def test(model, test_loader, strategy, device, isKAN=False):
    if not isKAN:
        model.eval()
    criterion = nn.NLLLoss()
    predictions = []
    ground_truths = []
    val_accuracy = 0
    loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)  #(labels % 2 if model.layers[-1] == 2 else labels).to(device)
            images = images.to(device)
            output = model(images)
            if strategy == "taskIL":
                    labels = labels % 2
            loss = criterion(output, labels)
            predictions.extend(output.argmax(dim=1).to('cpu').numpy())
            ground_truths.extend(labels.to('cpu').numpy())
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    val_accuracy /= len(test_loader)
    print(f"Accuracy: {val_accuracy}")
    return loss.item(), ground_truths, predictions


class EpochStat:
    @staticmethod
    def loadModelStats(name, dir=f'/mnt/c/Users/aless/Desktop/pykan/KAN-Continual_Learning/results/mnist/classIL', subdir='') -> list['EpochStat']:
        return sorted([pickle.load(open(os.path.join(dir, subdir, file), 'rb')) for file in
                       filter(lambda e: name == '_'.join(e.split('_')[:-1]), os.listdir(os.path.join(dir, subdir)))],
                      key=lambda e: e.epoch)

    def __init__(self, name, save_dir, epoch, train_loss=0, test_loss=0, labels=None, predictions=None, epoch_duration=0, lr=0,
                 train_losses=None, train_accuracies=None, task_id=None):
        self.name = name
        self.save_dir = save_dir
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.epoch = epoch
        self.predictions = predictions
        self.labels = labels
        self.epoch_duration = epoch_duration
        self.lr = lr
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.task_id = task_id

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        pickle.dump(self, open(os.path.join(self.save_dir, self.name + '_e' + str(self.epoch) + '.pickle'), 'wb'))

    def get_accuracy(self):
        accuracy = 0
        for label, prediction in zip(self.labels, self.predictions):
            if label == prediction:
                accuracy += 1
        return accuracy / len(self.labels)


def onEpochEnd(model, test_loader, strategy, device, save_dir, epoch, train_loss, epoch_duration, lr, task_id, isKAN):
    test_loss, labels, predictions = test(model, test_loader, strategy, device, isKAN)
    stat = EpochStat(model.__class__.__name__, save_dir, epoch, train_loss, test_loss, labels, predictions, epoch_duration,
                     lr, [], [], task_id)
    stat.save()

# def train_until_decrease(model, train_loader, save_dir, optimizer, device, start_epoch=0, epochs=5,
#                          on_epoch_end=None, lr=0, loader=None, task_id=None, isKAN=False):
#     if loader is None:
#         loader = train_loader
#     criterion = nn.NLLLoss()
#     # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)
#     for epoch in range(start_epoch, epochs + start_epoch):
#         if not isKAN:
#             model.train()
#         epoch_start = time.time_ns()
#         best_loss = 0
#         enough
#         with tqdm(loader) as pbar:
#             for images, labels in pbar:
#                 labels = labels.to(device)  #(labels % 2 if task_id is not None else labels).to(device)
#                 images = images.to(device)
#                 optimizer.zero_grad()
#                 output = model(images)
#                 loss = criterion(output, labels)
#                 loss.backward()
#                 if loss > best_loss:
#                     best_loss = loss
#                 else:
#                     enough = True 
#                 optimizer.step(closure=lambda: loss)
#                 accuracy = (output.argmax(dim=1) == labels).float().mean()
#                 pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
#                 # scheduler.step()
#         print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
#         epoch_duration = (time.time_ns() - epoch_start) // 1000000
#         if on_epoch_end is not None:
#             on_epoch_end(model, save_dir, epoch, loss.item(), epoch_duration, lr, task_id, isKAN)
#         # torch.save(model.state_dict(), f'{checkpoint}_ep{epoch + 1}.pth')

