import enum
import numpy as np
import torch
import os
import pickle


def evalandprint(args, algclass, train_loaders, test_loaders, SAVE_PATH, best_acc, a_iter, best_changed, record_trigger):
    print(f"============ Personalized {a_iter} ============")
    temp_list=[0] * args.n_clients
    # evaluation on training data
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.client_eval(
            client_idx, train_loaders[client_idx])
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    #record the test
    for client_idx in range(args.n_clients):
        _, test_acc = algclass.client_eval(
            client_idx, test_loaders[client_idx])
        temp_list[client_idx]=test_acc
    file_name = os.path.join(SAVE_PATH, args.alg+'_pertest.pkl')
    if record_trigger:
        record_trigger=False
        with open(file_name, 'wb+') as f:
            pickle.dump([np.mean(np.array(temp_list))], f)
    else:
        with open(file_name, 'rb') as f:
            test_list=pickle.load(f)
        test_list.append(np.mean(np.array(temp_list)))
        with open(file_name, 'wb+') as f:
            pickle.dump(test_list, f)   

    if np.mean(np.array(temp_list)) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = temp_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {best_acc[client_idx]:.4f}')
        print(f'Test macc:{np.mean(np.array(best_acc)):.4f}')
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_macc': np.mean(np.array(best_acc))}
        for i,tmodel in enumerate(algclass.client_model):
            tosave['client_model_'+str(i)]=tmodel.state_dict()
        if (args.alg=='myfed' or args.alg=='myfedmode'):
            tosave['server_model'] = algclass.server_models.state_dict()
        else:
            tosave['server_model']=algclass.server_model.state_dict()
        k = 'per'+args.alg
        torch.save(tosave, os.path.join(SAVE_PATH, k))

    return best_acc, best_changed, record_trigger

def evalandprintglo(args, algclass, train_loaders, test_loaders, SAVE_PATH, best_acc, a_iter, best_changed, record_trigger):
    print(f"============ Global {a_iter} ============")
    temp_list=[0] * args.n_clients
    # evaluation on training data
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.server_eval(train_loaders[client_idx])
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    #record the test
    for client_idx in range(args.n_clients):
        _, test_acc = algclass.server_eval(test_loaders[client_idx])
        temp_list[client_idx]=test_acc
    file_name = os.path.join(SAVE_PATH, args.alg+'_glotest.pkl')
    if record_trigger:
        record_trigger=False
        with open(file_name, 'wb+') as f:
            pickle.dump([np.mean(np.array(temp_list))], f)
    else:
        with open(file_name, 'rb') as f:
            test_list=pickle.load(f)
        test_list.append(np.mean(np.array(temp_list)))
        with open(file_name, 'wb+') as f:
            pickle.dump(test_list, f)

    if np.mean(np.array(temp_list)) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = temp_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {best_acc[client_idx]:.4f}')
        print(f'Test macc:{np.mean(np.array(best_acc)):.4f}')
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_macc': np.mean(np.array(best_acc))}
        if (args.alg=='myfed' or args.alg=='myfedmode'):
            tosave['server_model'] = algclass.server_models.state_dict()
        else:
            tosave['server_model']=algclass.server_model.state_dict()
        k = 'glo' + args.alg
        torch.save(tosave, os.path.join(SAVE_PATH, k))

    return best_acc, best_changed, record_trigger
