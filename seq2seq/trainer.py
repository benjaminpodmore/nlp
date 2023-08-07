import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class Trainer:
    def __init__(self, device, model, epochs, optimizer, lr_scheduler, criterion, train_dataloader, train_steps, val_dataloader, val_steps):
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps

        self.loss = {"train": [], "val": [0]}

    def train(self):
        for i in range(self.epochs):
            self.train_epoch()
            # self.validate_epoch()
            print(f"Epoch {i+1}: training loss: {self.loss['train'][-1]:.4f} validation loss: {self.loss['val'][-1]:.4f}")
            self.lr_scheduler.step()

    def train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, labels, 0.5)
            loss = self.criterion(outputs.permute(0, 2, 1), labels)
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
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs, labels, 0)
                loss = self.criterion(outputs.permute(0, 2, 1), labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
