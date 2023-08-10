import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, device, epochs, optimizer, criterion, train_dataloader, train_steps,
                 validation_dataloader, validation_steps, clip):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.validation_dataloader = validation_dataloader
        self.validation_steps = validation_steps
        self.clip = clip

        self.loss = {"train": [], "val": []}

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.validate_epoch()
            print(f"Epoch: {epoch + 1} train loss: {self.loss['train'][-1]} validation loss: {self.loss['val'][-1]}")

    def train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in tqdm(enumerate(self.train_dataloader, 1)):
            self.optimizer.zero_grad()

            inputs = batch_data[0].squeeze(0).permute(1, 0).to(self.device)
            labels = batch_data[1].squeeze(0).permute(1, 0).to(self.device)

            outputs = self.model(inputs, labels, 0.5)
            output_size = outputs.shape[2]
            outputs = outputs[1:].view(-1, output_size)
            labels = labels[1:].reshape(-1)

            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        self.loss["train"].append(np.mean(running_loss))

    def validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(self.validation_dataloader, 1)):
                inputs = batch_data[0].squeeze(0).permute(1, 0).to(self.device)
                labels = batch_data[1].squeeze(0).permute(1, 0).to(self.device)

                outputs = self.model(inputs, labels, 0.5)
                output_size = outputs.shape[2]
                outputs = outputs[1:].view(-1, output_size)
                labels = labels[1:].reshape(-1)

                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.validation_steps:
                    break

            self.loss["val"].append(np.mean(running_loss))