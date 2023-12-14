import os
import copy
import math
import numpy as np
import torch
import scipy.stats

from alg.fedavg import fedavg
from datautil.prepare_data import get_soft_dataset
from datautil.datasplit import  DataPartitioner

class fedsoft(fedavg):
    def __init__(self, args):
        super(fedsoft, self).__init__(args)

    def update_client_weight(self):
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
        distribution_dist={i: np.array([]) for i in range(self.args.n_clients)}
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
                        temp_list[j]+=1
            summ = sum(temp_list)
            for k in range(len(temp_list)):
                temp_list[k] = temp_list[k] / summ
            distribution_dist[i]=temp_list
        self.client_weight=update_weight_matrix1(self.args,distribution_dist)


def update_weight_matrix1(args, distribution_dist):
    client_num = args.n_clients
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                dists = [k for k in range(len(distribution_dist[i]))]
                tmp = scipy.stats.wasserstein_distance(dists,dists,distribution_dist[i], distribution_dist[j])
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
