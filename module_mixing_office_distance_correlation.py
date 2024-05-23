import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from datasets.office import data_load
import networks.resnet_EFT as resnet_EFT
import networks.hidden_mixing_clf_layerwise as hid_mix
from utils.utils_load_model import *
import copy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from DC_criterion import *


class args():
    dataset = "office"
    data_root = './data/office'
    total_epoch = 100
    batch_size = 128
    seed = 3
    trte = "val"
    model_path = 'style_param_models/office_100ep_lr0.001_bs128_save_best_val_model/{}_2_tsk_to_use_layerwise'.format(str(seed))
    prev_model_path = 'style_param_models/office_100ep_lr0.001_bs128_save_best_val_model/resneteft_independent'


if __name__ == '__main__':
    random_seed = args.seed
    seed_torch(random_seed)
    device = 'cuda'
    args = args()

    dataset_name = args.dataset
    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.num_classes = 31

    for args.t in [0, 1, 2]:  # choose which domain to set as target {0 to len(names)-1}
        args.src = []
        for i in range(len(names)):
            if i == args.t:
                continue
            else:
                args.src.append(names[i])

        args.s_dset_path = os.path.join(args.data_root, names[args.t], 'image_unida_list.txt')

        chosen_models = []
        test_acc_final = []

        pre_tasks = args.src
        print("pre tasks", pre_tasks)
        cur_task = names[args.t]

        dset_loaders = data_load(args)
        train_loader, val_loader, test_loader = dset_loaders["source_tr"], dset_loaders["source_val"], dset_loaders[
            "source_te"]
        dataloader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
        print(len(dataloader['train'].dataset.imgs))

        nn_cls_acc = []
        model = resnet_EFT.Net(args)
        num_classes, num_ftrs = args.num_classes, model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))
        knn = KNeighborsClassifier(n_neighbors=5)
        for m in range(len(pre_tasks)):
            path = os.path.join(args.prev_model_path, pre_tasks[m]+'.pth')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['net'])
            model.to('cuda')
            # criterion = nn.CrossEntropyLoss()
            X_train, y_train, X_test, y_test = [], [], [], []
            # train_loss, val_loss = 0, 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    if inputs.shape[0] != 0:
                        outputs, e = model(inputs)
                        # loss = criterion(outputs, targets)
                        # train_loss += loss.item()
                    X_train.append(e)
                    y_train.append(targets)
                # train_loss = train_loss / (batch_idx + 1)
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    if inputs.shape[0] != 0:
                        outputs, e = model(inputs)
                        # loss = criterion(outputs, targets)
                        # val_loss += loss.item()
                    X_test.append(e)
                    y_test.append(targets)
                # val_loss = val_loss / (batch_idx + 1)
            X_train, y_train = torch.cat(X_train), torch.cat(y_train)
            X_test, y_test = torch.cat(X_test), torch.cat(y_test)
            knn.fit(X_train.cpu(), y_train.cpu())
            acc = knn.score(X_test.cpu(), y_test.cpu())
            # nn_cls_acc.append((m, acc))
            nn_cls_acc.append(acc)
            # print(train_loss, val_loss)
        print(nn_cls_acc)
        max_val, max_idx = torch.topk(torch.tensor(nn_cls_acc), largest=True, k=1, sorted=True)
        print(max_val, max_idx)
        partial_pre_tasks = [pre_tasks[idx] for idx in sorted(max_idx.cpu().numpy())]
        # partial_pre_tasks = pre_tasks
        print("pre tasks", partial_pre_tasks)

        # grid search weight decay {0, 1e-4, 1e-5}, learing rate {0.01, 0.001}
        wgt_decays = [1e-5]
        learn_rates = [0.001]
        test_acc_grid_search = np.zeros((1, 1))
        models_grid_search = [[], [], []]
        for i, lr in enumerate(learn_rates):
            for j, wgt_decay in enumerate(wgt_decays):
                # pick out related tasks with a threshold
                hidden_mixing_model = hid_mix.Net(partial_pre_tasks, args)
                dict = torch.load('resnet18.pth')
                hidden_mixing_model = model_equal_part_embed(hidden_mixing_model, dict)
                # Modify fc layers to match num_classes
                num_classes = args.num_classes
                num_ftrs = hidden_mixing_model.fc.in_features
                hidden_mixing_model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))
                # layerwise module-mixing
                hid_mix.grad_false(hidden_mixing_model, len(partial_pre_tasks))

                get_parameter_number(hidden_mixing_model)
                hidden_mixing_model = hidden_mixing_model.to(device)

                # Loss function
                # criterion = nn.CrossEntropyLoss()
                criterion = Loss_DC(0.05).to('cuda')
                # Optimizer
                optimizer = optim.Adam(hidden_mixing_model.parameters(), lr=lr, weight_decay=wgt_decay)
                # Learning rate decay
                exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

                best_acc = 0
                best_model = copy.deepcopy(hidden_mixing_model)
                for epoch in range(args.total_epoch):
                    hid_mix.train(train_loader, epoch, hidden_mixing_model, args, optimizer, criterion)
                    best_acc, best_model = hid_mix.val_test(val_loader, epoch, cur_task, hidden_mixing_model, args,
                                                            criterion, best_acc, best_model, pre_tasks)
                    if epoch % 99 == 0:
                        print(best_model.task_weights.weight.exp() / torch.sum(best_model.task_weights.weight.exp()))
                    exp_lr_scheduler.step()

                best_test_acc = hid_mix.test(test_loader, cur_task, best_model, args, criterion)
                test_acc_grid_search[i][j] = best_test_acc
                models_grid_search[i].append(best_model)
        best_test_ind = np.unravel_index(np.argmax(test_acc_grid_search, axis=None), test_acc_grid_search.shape)
        best_test_acc = test_acc_grid_search[best_test_ind[0]][best_test_ind[1]]
        lr, wgt_decay = learn_rates[best_test_ind[0]], wgt_decays[best_test_ind[1]]
        print(test_acc_grid_search, best_test_ind, best_test_acc, lr, wgt_decay)
        hid_mix.save_model_grid_search('task_' + names[args.t], best_test_acc,
                                       models_grid_search[best_test_ind[0]][best_test_ind[1]],
                                       args, lr, wgt_decay, pre_tasks, None)
        print(' Done!')
        test_acc_final.append(best_test_acc)
        print(test_acc_final)
