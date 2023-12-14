import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import torch.nn.functional as F
from util.modelsel import modelsel
from util.traineval import trainwithteacher, trainwithstudents, trainwithfinetune, test
from datautil.prepare_data import get_soft_dataset
from datautil.datasplit import  DataPartitioner

class myfedmode(torch.nn.Module):
    def __init__(self, args):
        super(myfedmode, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args
    '''
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
        #注释注释注释注释注释注释注释
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

    
    def update(self):
        server_list=[]
        client_dist = {i: [] for i in range(self.args.n_clients)}
        server_model=copy.deepcopy(
                        self.server_models).to(self.args.device)
        data = get_soft_dataset(self.args.dataset)(self.args)
        data_partitioner = DataPartitioner(
            self.args,
            data,
            [1],
            partition_type="evenly",
            # consistent_indices=False,
        )
        data_soft = torch.utils.data.DataLoader(data_partitioner.use(0), batch_size=self.args.batch, shuffle=True)

        with torch.no_grad():
            for data, target in data_soft:
                data = data.to(self.args.device).float()
                server_model.eval()
                list1=F.softmax(server_model(data)).tolist()
                for o in range(len(list1)):
                    server_list.append(list1[o])
                for i in range(self.args.n_clients):
                    temp_model = copy.deepcopy(
                        self.client_model[i]).to(self.args.device)
                    temp_model.eval()
                    list2 = F.softmax(temp_model(data)).tolist()
                    for o in range(len(list2)):
                        client_dist[i].append(list2[o])
        weight_list = []
        for i in range(len(client_dist)):
            tmp_list=[]
            for j in range(len(client_dist[i])):
                dists = [k for k in range(len(client_dist[i][j]))]
                tmp_list.append(scipy.stats.wasserstein_distance(dists, dists, client_dist[i][j], server_list[j]))
            weight_list.append(sum(tmp_list)/len(tmp_list))
        summ=sum(weight_list)
        for m in range(len(weight_list)):
            weight_list[m]=weight_list[m]/summ

        client_num = self.args.n_clients
        similarity=np.zeros((client_num, client_num))

        for i in range(client_num):
            temp_list = []
            for j in range(client_num):
                if(i==j):
                    similarity[i][j]=0
                else:
                    for k in range(len(client_dist[i])):
                        dists = [m for m in range(len(client_dist[i][k]))]
                        temp_list.append(scipy.stats.wasserstein_distance(dists, dists, client_dist[i][k], client_dist[j][k]))
                    similarity[i][j]=sum(temp_list)/len(temp_list)

        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    similarity[i, j] = 0
                else:
                    tmp=similarity[i, j]
                    if tmp == 0:
                        similarity[i, j] = 100000000000000
                    else:
                        similarity[i, j] = 1 / tmp
        weight_s = np.sum(similarity, axis=1)
        weight_s = np.repeat(weight_s, client_num).reshape(
            (client_num, client_num))
        weight_m = (similarity / weight_s) * self.args.finetune_lam
        self.client_weight = weight_list

        self.sim_matrix = weight_m
    




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
    '''
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
        weight_list = []
        if(self.args.adaptive=='full' or self.args.adaptive=='server'):
            center=self.center
            for idx in range(self.args.n_clients):
                client = self.distribution_dist[idx]
                dists = [k for k in range(len(center))]
                tmp = scipy.stats.wasserstein_distance(dists, dists, center, client)
                weight_list.append(tmp)
            summ = sum(weight_list)
            for k in range(len(weight_list)):
                weight_list[k] = weight_list[k] / summ
        else:
            for idx in range(self.args.n_clients):
                weight_list.append(1/self.args.n_clients)
        return weight_list

    def get_similarity_matrix(self):
        client_num = self.args.n_clients
        weight_m = np.zeros((client_num, client_num))
        if(self.args.adaptive=='full' or self.args.adaptive=='client'):
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
            weight_m = (weight_m / weight_s)
        else:
            for i in range(client_num):
                for j in range(client_num):
                    if i == j:
                        weight_m[i, j] = 0
                    else:
                        weight_m[i, j] = 1/(client_num-1)
        return weight_m

    def client_train(self, dataloader):

        # 注释注释注释注释注释注释注释
        if (self.args.adaptive):
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

        clients = []
        for cid in range(self.args.n_clients):
            clients.append(self.client_model[cid])

        for cid in range(self.args.n_clients):
            center.eval()
            with torch.no_grad():
                for key in center.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif self.args.nosharebn and 'bn' in key:
                        pass
                    else:
                        self.client_model[cid].state_dict()[key].data.copy_(
                            center.state_dict()[key])
            client = self.client_model[cid]

            # 客户端中本地数据的batch数量要和参考数据batch数量一致
            data_soft = torch.utils.data.DataLoader(
                data_partitioner.use(0), batch_size=int(len(data_partitioner.use(0)) / (
                        len(dataloader[cid]) * self.args.batch)), shuffle=True)

            if(self.args.mode=='server' or self.args.mode=='null'):
                model, train_loader, optimizer = client, dataloader[cid], self.optimizers[cid]
                for _ in range(self.args.wk_iters):
                    trainwithteacher(model, train_loader, optimizer, self.loss_fun, self.args.device, None, 1,self.args, False)
            elif(self.args.mode=='client' or self.args.mode=='full'):
                model, train_loader, dist_loader, optimizer, tmodel = client, dataloader[cid], data_soft, self.optimizers[cid], clients
                for _ in range(self.args.wk_iters):
                    trainwithfinetune(model, train_loader, dist_loader, optimizer, self.loss_fun, self.args.device, tmodel, self.sim_matrix[cid], self.args)



    def server_train(self):
        template_model = self.server_model
        for key in template_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                pass
            #elif self.args.nosharebn and 'bn' in key:
                #pass
            else:
                temp = torch.zeros_like(template_model.state_dict()[key])
                for client_idx in range(self.args.n_clients):
                    temp += 1 / self.args.n_clients * self.client_model[client_idx].state_dict()[key]
                template_model.state_dict()[key].data.copy_(temp)
        self.server_models = template_model
        if(self.args.adaptive):
            self.distribution_dist = self.get_client_distribution_dist()
            self.center = self.get_center_distribution_dist()
        self.client_weight = self.update_weight()

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
        train_loader, optimizer =  data_soft, optim.SGD(params=center.parameters(),lr=self.args.server_lr)
        lam = []
        for j in range(len(self.client_weight)):
            lam.append(self.client_weight[j])
        if(self.args.mode=='server' or self.args.mode=='full'):
            for _ in range(self.args.wk_iters):
                trainwithstudents(center, train_loader, optimizer, self.loss_fun, self.args.device, clients, lam, self.args)
        elif(self.args.mode=='client'):
            for _ in range(self.args.wk_iters):
                trainwithteacher(center, train_loader, optimizer, self.loss_fun, self.args.device, None, lam, self.args, False)
        elif(self.args.mode=='null'):
            '''
            for _ in range(self.args.wk_iters):
                trainwithteacher(center, train_loader, optimizer, self.loss_fun, self.args.device, None, lam,
                                self.args, 1)
            '''



    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc
    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_models, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc