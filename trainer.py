import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy


class Trainer():
    def __init__(self, model, optimizer, crit, device):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.device = device

        super().__init__()


    def _train(self, train_loader):
        self.model.train()
        total_loss = 0
        i = 0

        for x, y in iter(train_loader):
            i = i + 1
            x, y = x.to(self.device), y.to(self.device)
            
            y_hat = self.model(x)
            loss = self.crit(y_hat, y.squeeze())
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        #   if config.verbose >= 2:
            print("Train Iteration(%d/%d): loss=%.4e" % (i, len(train_loader), float(loss)))

            total_loss += float(loss)

        return total_loss / len(x)


    def _validate(self, valid_loader):
        self.model.eval()
        total_loss = 0
        i = 0

        with torch.no_grad():
            for x, y in iter(valid_loader):
                i = i + 1
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x)
                loss = self.crit(y_hat, y.squeeze())

        #       if config.verbose >= 2:
                print("Valid Iteration(%d/%d): loss=%.4e" % (i, len(valid_loader), float(loss)))
                
                total_loss += float(loss)

        return total_loss / len(x)


    def train(self, train_loader, valid_loader):
        lowest_loss = np.inf
    
        for epoch_index in range(20):
            train_loss = self._train(train_loader)
            valid_loss = self._validate(valid_loader)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                100,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        self.model.load_state_dict(best_model)