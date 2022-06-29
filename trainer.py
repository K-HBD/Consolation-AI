import torch
import numpy as np

from copy import deepcopy

from torch.utils.data import DataLoader
from fastprogress.fastprogress import master_bar, progress_bar


class Trainer():
    def __init__(self, model, optimizer, crit, device, epochs):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.device = device
        self.epochs = epochs

        super().__init__()

    
    def train(self, train_loader, train_dataset, valid_loader, valid_dataset):
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.99, end_factor=0.0001, total_iters=len(train_loader)*self.epochs, last_epoch=- 1, verbose=False)

        train_losses = []
        train_accuracy = []
        valid_losses = []
        valid_accuracy = []

        mb = master_bar(range(self.epochs))
        lowest_loss = np.inf
        
        for epoch in mb:
            correct_train = 0
            train_loss = 0
            correct_valid = 0
            valid_loss = 0

            self.model.train()
            for x, y in progress_bar(train_loader, parent=mb):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                y_hat = self.model(x)
                t_loss = self.crit(y_hat, y.squeeze())
                train_loss += t_loss.item()
                t_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                _, pred_idxs = torch.topk(y_hat, 1)
                correct_train += torch.eq(y, pred_idxs.squeeze()).sum().item()
                scheduler.step()

            self.model.eval()
            with torch.no_grad():
                for x, y in progress_bar(valid_loader, parent=mb):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    y_hat = self.model(x)
                    v_loss = self.crit(y_hat, y.squeeze())
                    valid_loss += v_loss.item()

                    _, pred_idxs = torch.topk(y_hat, 1)
                    correct_valid += torch.eq(y, pred_idxs.squeeze()).sum().item()

                    if lowest_loss >= v_loss:
                        lowest_loss = v_loss
                        best_model = deepcopy(self.model.state_dict())

            train_acc = correct_train/len(train_dataset)
            valid_acc = correct_valid/len(valid_dataset)
            loss_train = train_loss/len(train_loader)
            loss_valid = valid_loss/len(valid_loader)

            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)
            train_losses.append(loss_train)
            valid_losses.append(loss_valid)

            print("Epoch: {: d}  train_accuracy: {: .4f}  valid_accuracy: {: .4f} train_loss: {: .4f} valid_loss: {: .4f} lowest_loss: {: .4f}".format(epoch, train_acc, valid_acc, loss_train, loss_valid, lowest_loss))
            self.model.load_state_dict(best_model)