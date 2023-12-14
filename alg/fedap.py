import os
import copy
import math
import numpy as np
import torch

from alg.fedavg import fedavg
from util.traineval import pretrain_model


class fedap(fedavg):
    def __init__(self, args):
        super(fedap, self).__init__(args)

    def set_client_weight(self, train_loaders):
        os.makedirs('./checkpoint/'+'pretrained/', exist_ok=True)
        preckpt = './checkpoint/'+'pretrained/' + \
            self.args.dataset+'_'+str(self.args.batch)
        self.pretrain_model = copy.deepcopy(
            self.server_model).to(self.args.device)
        if not os.path.exists(preckpt):
            #pretrain:30% of data to train 1 epoch
            pretrain_model(self.args, self.pretrain_model,
                           preckpt, self.args.device)
        self.preckpt = preckpt
        self.client_weight = get_weight_preckpt(
            self.args, self.pretrain_model, self.preckpt, train_loaders, self.client_weight,self.args.device)

    def fedap_server_aggre_part1(self):
        pass
    def fedap_server_aggre_part2(self):
        client_num = self.args.n_clients
        models= self.client_model
        client_weights = self.client_weight
        server_model =self.server_model
        tmpmodels = []
        for i in range(client_num):
            tmpmodels.append(copy.deepcopy(models[i]).to(self.args.device))
        with torch.no_grad():
            for cl in range(client_num):
                for key in self.server_model.state_dict().keys():
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[cl, client_idx] * tmpmodels[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    if 'bn' not in key:
                        models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])

        for cl in range(client_num):
            for key in models[cl].state_dict().keys():
                if 'bn' not in key:
                        self.client_model[cl].state_dict()[key].data.copy_(models[cl].state_dict()[key])


def get_form(model):
    #the mean and var of batch normalization 是基于batch的，(m,f,p,q)，m为min-batch sizes，f为特征图个数，p、q分别为特征图的宽高
    #则该batch normalization层的mean和var均为f个
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


def get_weight_matrix1(args, bnmlist, bnvlist, client_weights):
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = 1/tmp
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m/weight_s)*(1-args.model_momentum)
    for i in range(client_num):
        weight_m[i, i] = args.model_momentum
    return weight_m


def get_weight_preckpt(args, model, preckpt, trainloadrs, client_weights, device='cpu'):
    model.load_state_dict(torch.load(preckpt)['state'])
    model.eval()
    bnmlist1, bnvlist1 = [], []
    for i in range(args.n_clients):
        avgmeta = metacount(get_form(model)[0])
        with torch.no_grad():
            for data, _ in trainloadrs[i]:
                data = data.to(device).float()
                fea = model.getallfea(data) #batch normalization前的x
                nl = len(data)     #batch size
                tm, tv = [], []
                for item in fea:
                    if len(item.shape) == 4:
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                    else:
                        tm.append(torch.mean(
                            item, dim=0).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=0).detach().to('cpu').numpy())
                avgmeta.update(nl, tm, tv)
        #bnmlist1和bnvlist1中的第i个元素为第i个client在每个batch normalization中的mean和var组成的list
        bnmlist1.append(avgmeta.getmean())
        bnvlist1.append(avgmeta.getvar())
    weight_m = get_weight_matrix1(args, bnmlist1, bnvlist1, client_weights)
    return weight_m


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count+m
        for i in range(self.bl):
            tmpm = (self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i] = (self.count*(self.var[i]+np.square(tmpm -
                           self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var