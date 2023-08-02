import torch
import numpy as np


class Trainer:
    def __init__(self, model, epochs, optimizer, criterion, train_dataloader, val_dataloader, device):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        self.loss = {"train": [], "val": [-1]}

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            # self.validate_epoch()
            print(f"Epoch: {epoch+1}/{self.epochs}, Train loss={self.loss['train'][-1]:.5f}, Val loss={self.loss['val'][-1]:.5f}, ")

    def train_epoch(self):
        self.model.train()
        running_loss = []

        hidden_state = self.model.init_hidden().to(self.device)

        # for each minibatch
        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs, hidden_state = self.model(inputs, hidden_state.detach())
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == 10:
                break

        self.loss["train"].append(np.mean(running_loss))

    def validate_epoch(self):
        self.model.eval()
        running_loss = []

        hidden_state = self.model.init_hidden()

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0]
                labels = batch_data[1]

                outputs, hidden_state = self.model(inputs, hidden_state)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
