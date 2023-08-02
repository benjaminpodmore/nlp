import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import RNN
from tqdm import tqdm


class Trainer:
    def __init__(self, device, model: RNN, epochs, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler,
                 criterion, train_dataloder: DataLoader, train_steps, val_dataloader, val_steps):
        self.device = device
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloder
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps

        self.loss = {"train": [], "val": []}

    def train(self):
        for i in range(self.epochs):
            self.train_epoch()
            self.validate_epoch()
            print(f"Epoch {i+1}: training loss: {self.loss['train'][-1]}, validation loss: {self.loss['val'][-1]}")
            self.lr_scheduler.step()

    def train_epoch(self):
        self.model.train()
        running_loss = []
        for i, batch_data in enumerate(self.train_dataloader):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
