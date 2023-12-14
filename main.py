# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import argparse
import numpy as np

from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from util.evalandprint import evalandprint, evalandprintglo
from alg import algs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    parser.add_argument('--datapercent', type=float,
                        default=1e-1, help='data percent to use')
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnist]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to savethe checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--server_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=20, help='number of clients')
    '''
    #*********************************************************使用前需要和data一起修改
    parser.add_argument('--soft_ratio', type=float,
                        default=0.2, help='soft data ratio')
    #*********************************************************使用前需要和data一起修改
    '''
    # *********************************************************消融实验
    parser.add_argument('--adaptive', type=str,
                        default='full', help='full | server | client| null adaptive or 1/N')
    # *********************************************************消融实验
    #迪利克雷分布使用
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.1, help='data split for label shift')
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')
    #choose the feature type???
    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')
    #iterations for pretrained models???
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    #本身的影响因素 1-model_momentum为邻居影响因素之和 仅影响收敛速度
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    parser.add_argument('--mode', type=str,
                        default='full', help='[full | server | client | null]')
    parser.add_argument('--agg_lam', type=float,
                        default=0.01, help='hyperparameter for myfed')
    parser.add_argument('--finetune_lam', type=float,
                        default=0.01, help='hyperparameter for myfed')
    '''
    parser.add_argument('--train_data_ratio', type=float,
                        default=0.5, help='hyperparameter for myfed')
    '''
    args = parser.parse_args()

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    if args.dataset in ['vlcs', 'pacs', 'off_home']:
        args = img_param_init(args)
        args.n_clients = 4

    exp_folder = f'fed_{args.dataset}_{args.alg}_client_lr{args.lr}_server_lr{args.server_lr}_{args.n_clients}_{args.non_iid_alpha}_{args.iters}_{args.wk_iters}'
    if args.nosharebn:        exp_folder += '_nosharebn'
    if args.alg=='myfedmode' or args.alg=='fedprox_FedAKD':
        exp_folder += '_adaptive_'
        exp_folder += str(args.adaptive)
        exp_folder += '_mode_'
        exp_folder += args.mode
        if(args.mode=='server'or args.mode=='full'):
            exp_folder += '_'
            exp_folder += str(args.agg_lam)
        if(args.mode=='client'or args.mode=='full'):
            exp_folder += '_'
            exp_folder += str(args.finetune_lam)

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = args.save_path

    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)
    algclass = algs.get_algorithm_class(args.alg)(args)

    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters-1
        print('Common knowledge accumulation stage')
    elif args.alg == 'myfed' or args.alg == 'myfedmode' or args.alg == 'fedprox_FedAKD':
        algclass.pretrain(train_loaders)
    per_best_changed = False
    per_record_trigger = True
    glo_best_changed = False
    glo_record_trigger = True


    per_best_acc = [0] * args.n_clients
    per_best_tacc = [0] * args.n_clients
    glo_best_acc = [0] * args.n_clients
    glo_best_tacc = [0] * args.n_clients
    start_iter = 0

    for a_iter in range(start_iter, args.iters):
        print(f"============ Train round {a_iter} ============")

        if args.alg == 'metafed':
            for c_idx in range(args.n_clients):
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
            algclass.update_flag(val_loaders)
        elif args.alg == 'myfedmode' or args.alg =='fedprox_FedAKD':
            algclass.client_train(train_loaders)
            algclass.server_train()
        elif args.alg == 'myfed':
            algclass.fine_tune(train_loaders)
            algclass.server_train()
        else:
            # local client training
            for wi in range(args.wk_iters):
                for client_idx in range(args.n_clients):
                    algclass.client_train(
                        client_idx, train_loaders[client_idx], a_iter)
            if args.alg == 'fedsoft':
                algclass.update_client_weight()
            # server aggregation
            if(args.alg == 'fedprox'):
                algclass.fedprox_server_aggre_part1()
            if(args.alg == 'fedbn'):
                algclass.fedbn_server_aggre_part1()
            if(args.alg == 'fedap'):
                algclass.fedap_server_aggre_part1() #什么都不做，不要global model

        per_best_acc, per_best_changed, per_record_trigger= evalandprint(
            args, algclass, train_loaders, test_loaders, SAVE_PATH, per_best_acc, a_iter, per_best_changed, per_record_trigger)
        '''
        if(args.alg == 'myfedmode' or args.alg == 'fedprox_FedAKD' or args.alg == 'fedprox'):
            glo_best_acc, glo_best_changed, glo_record_trigger = evalandprintglo(
                args, algclass, train_loaders, test_loaders, SAVE_PATH, glo_best_acc, a_iter,glo_best_changed, glo_record_trigger)
        '''
        if(args.alg == 'fedprox'):
            algclass.fedprox_server_aggre_part2()
        if (args.alg == 'fedbn'):
            algclass.fedbn_server_aggre_part2()
        if (args.alg == 'fedap'):
            algclass.fedap_server_aggre_part2()



    if args.alg == 'metafed':
        print('Personalization stage')
        for c_idx in range(args.n_clients):
            algclass.personalization(
                c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
        per_best_acc, per_best_changed, per_record_trigger = evalandprint(
            args, algclass, train_loaders, test_loaders, SAVE_PATH, per_best_acc, a_iter, per_best_changed,
            per_record_trigger)
    '''
    s = 'Personalized test acc for each client: '
    for item in per_best_acc:
        s += f'{item:.4f},'
    mean_acc_test = np.mean(np.array(per_best_acc))
    s += f'\nAverage accuracy: {mean_acc_test:.4f}'
    print(s)
    '''
