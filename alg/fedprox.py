# coding=utf-8
from alg.fedavg import fedavg
from util.traineval import train, train_prox, test
from util.modelsel import modelsel
from alg.core.comm import communication
import torch
import torch.nn as nn
import torch.optim as optim
class fedprox(torch.nn.Module):
    def __init__(self, args):
        super(fedprox, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args

    def client_train(self, c_idx, dataloader, round):
        if round > 0:
            train_loss, train_acc = train_prox(
                self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    def fedprox_server_aggre_part1(self):
        server_model=self.server_model
        models =self.client_model
        client_weights=self.client_weight
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
        self.server_model =server_model
    def fedprox_server_aggre_part2(self):
        for key in self.server_model.state_dict().keys():
            for client_idx in range(self.args.n_clients):
                self.client_model[client_idx].state_dict()[key].data.copy_(self.server_model.state_dict()[key])
    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc