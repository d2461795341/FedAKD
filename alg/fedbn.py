from alg.fedavg import fedavg
import torch
import torch.nn as nn
class fedbn(fedavg):
    def __init__(self,args):
        super(fedbn, self).__init__(args)

    def fedbn_server_aggre_part1(self):
        server_model=self.server_model
        models =self.client_model
        client_weights=self.client_weight
        for key in server_model.state_dict().keys():
            if 'bn' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(self.args.n_clients):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
        self.server_model =server_model
    def fedbn_server_aggre_part2(self):
        for key in self.server_model.state_dict().keys():
            if 'bn' not in key:
                for client_idx in range(self.args.n_clients):
                    self.client_model[client_idx].state_dict()[key].data.copy_(self.server_model.state_dict()[key])
