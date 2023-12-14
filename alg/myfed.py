import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


from util.modelsel import modelsel
from util.traineval import trainwithteacher, trainwithstudents, test, trainwithfinetune
from datautil.prepare_data import get_soft_dataset
from datautil.datasplit import  DataPartitioner

class myfed(torch.nn.Module):
    def __init__(self, args):
        super(myfed, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args

    def pretrain(self, train_loaders):
        #pretrain
        client_num = self.args.n_clients
        optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=self.args.lr) for idx in range(client_num)]

        for idx in range(client_num):
            client_idx = idx
            model, train_loader, optimizer, tmodel = self.client_model[client_idx], train_loaders[client_idx], optimizers[client_idx], None
            for _ in range(1):
                _, _ = trainwithteacher(
                    model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, 1, self.args, False)

        self.distribution_dist = self.get_client_distribution_dist()

        template_model=self.server_model
        for key in template_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                pass
            else:
                temp = torch.zeros_like(template_model.state_dict()[key])
                for client_idx in range(self.args.n_clients):
                    temp += 1/self.args.n_clients * self.client_model[client_idx].state_dict()[key]
                template_model.state_dict()[key].data.copy_(temp)
        self.server_models=template_model


    def get_center_distribution_dist(self):
        data = get_soft_dataset(self.args.dataset)(self.args)
        data_partitioner = DataPartitioner(
            self.args,
            data,
            [1],
            partition_type="evenly",
            # consistent_indices=False,
        )
        data_soft = torch.utils.data.DataLoader(
            data_partitioner.use(0), batch_size=self.args.batch, shuffle=True)
        distribution_dist = []
        temp_list = list(np.zeros(self.args.num_classes, dtype=float))
        temp_model = copy.deepcopy(
            self.server_models).to(self.args.device)
        temp_model.eval()
        with torch.no_grad():
            for data, target in data_soft:
                data = data.to(self.args.device).float()
                output = temp_model(data)
                pred = output.data.max(1)[1]
                for j in pred:
                    temp_list[j] += 1
        summ = sum(temp_list)
        for k in range(len(temp_list)):
            temp_list[k] = temp_list[k] / summ
        distribution_dist = temp_list
        return distribution_dist


    def get_client_distribution_dist(self):
        data = get_soft_dataset(self.args.dataset)(self.args)
        data_partitioner = DataPartitioner(
            self.args,
            data,
            [1],
            partition_type="evenly",
            # consistent_indices=False,
        )
        data_soft = torch.utils.data.DataLoader(
            data_partitioner.use(0), batch_size=self.args.batch, shuffle=True)
        distribution_dist = {i: np.array([]) for i in range(self.args.n_clients)}
        for i in range(self.args.n_clients):
            temp_list = list(np.zeros(self.args.num_classes, dtype=float))
            temp_model = copy.deepcopy(
                self.client_model[i]).to(self.args.device)
            temp_model.eval()
            with torch.no_grad():
                for data, target in data_soft:
                    data = data.to(self.args.device).float()
                    output = temp_model(data)
                    pred = output.data.max(1)[1]
                    for j in pred:
                        temp_list[j] += 1
            summ = sum(temp_list)
            for k in range(len(temp_list)):
                temp_list[k] = temp_list[k] / summ
            distribution_dist[i] = temp_list
        return distribution_dist


    def update_weight(self):
        weight_list=[]
        center=self.center
        for idx in range(self.args.n_clients):
            client = self.distribution_dist[idx]
            dists = [k for k in range(len(center))]
            tmp = scipy.stats.wasserstein_distance(dists, dists, center, client)
            weight_list.append(tmp)
        summ = sum(weight_list)
        for k in range(len(weight_list)):
            weight_list[k] = weight_list[k] / summ
        return weight_list


    def client_train(self, dataloader,val_loader):
        self.center = self.get_center_distribution_dist()
        self.client_weight=self.update_weight()
        center = self.server_models
        for cid in range(self.args.n_clients):
            client = self.client_model[cid]
            flag = False
            model, train_loader, optimizer, tmodel = client, dataloader[cid], self.optimizers[cid], center

            trainwithteacher(model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, 1,self.args, flag)
            for _ in range(self.args.wk_iters-1):
                trainwithteacher(model, train_loader, optimizer, self.loss_fun, self.args.device, None, 1,
                                self.args, flag)


    def server_train(self):
        template_model = self.server_model
        for key in template_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                pass
            else:
                temp = torch.zeros_like(template_model.state_dict()[key])
                for client_idx in range(self.args.n_clients):
                    temp += 1 / self.args.n_clients * self.client_model[client_idx].state_dict()[key]
                template_model.state_dict()[key].data.copy_(temp)
        self.server_models = template_model

        self.distribution_dist=self.get_client_distribution_dist()
        self.center = self.get_center_distribution_dist()
        self.client_weight=self.update_weight()

        data = get_soft_dataset(self.args.dataset)(self.args)
        data_partitioner = DataPartitioner(
            self.args,
            data,
            [1],
            partition_type="evenly",
            # consistent_indices=False,
        )
        data_soft = torch.utils.data.DataLoader(
            data_partitioner.use(0), batch_size=self.args.batch, shuffle=True)

        center = self.server_models
        clients = []
        for cid in range(self.args.n_clients):
            clients.append(self.client_model[cid])
        train_loader, optimizer =  data_soft, optim.SGD(params=center.parameters(),lr=self.args.lr)
        lam = []
        for j in range(len(self.client_weight)):
            lam.append(self.client_weight[j] * self.args.agg_lam)
        for _ in range(self.args.wk_iters):
            trainwithstudents(center, train_loader, optimizer, self.loss_fun, self.args.device, clients, lam, self.args)

    def get_similarity_matrix(self):
        client_num = self.args.n_clients
        weight_m = np.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    weight_m[i, j] = 0
                else:
                    dists = [k for k in range(len(self.distribution_dist[i]))]
                    tmp = scipy.stats.wasserstein_distance(dists, dists, self.distribution_dist[i],
                                                           self.distribution_dist[j])
                    if tmp == 0:
                        weight_m[i, j] = 100000000000000
                    else:
                        weight_m[i, j] = 1 / tmp
        weight_s = np.sum(weight_m, axis=1)
        weight_s = np.repeat(weight_s, client_num).reshape(
            (client_num, client_num))
        weight_m = (weight_m / weight_s) * self.args.finetune_lam
        return weight_m


    def fine_tune(self,dataloader):
        print(f"============ finetune ============")
        self.center = self.get_center_distribution_dist()
        self.client_weight=self.update_weight()

        self.sim_matrix=self.get_similarity_matrix()

        center = self.server_models

        data = get_soft_dataset(self.args.dataset)(self.args)
        data_partitioner = DataPartitioner(
            self.args,
            data,
            [1],
            partition_type="evenly",
            # consistent_indices=False,
        )
        data_soft = torch.utils.data.DataLoader(
            data_partitioner.use(0), batch_size=self.args.batch*5, shuffle=True)

        clients = []
        for cid in range(self.args.n_clients):
            clients.append(self.client_model[cid])

        for cid in range(self.args.n_clients):
            client = self.client_model[cid]
            model, train_loader, dist_loader , optimizer, tmodel = client, dataloader[cid], data_soft, self.optimizers[cid], clients

            trainwithteacher(model, train_loader, optimizer, self.loss_fun, self.args.device, center, 1, self.args,False)
            for _ in range(self.args.wk_iters-1):
                trainwithfinetune(model,train_loader,dist_loader,optimizer,self.loss_fun, self.args.device,tmodel,self.sim_matrix[cid], self.args)


    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc
