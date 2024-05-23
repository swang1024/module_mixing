import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from datasets.main import load_dataset, get_test_loader, get_train_valid_loader
from datasets.create_cl_ds import load_ds_info
from datasets.office import data_load
import networks.hidden_mixing_clf_layerwise as hid_mix
from utils.utils_load_model import *
import networks.resnet_EFT as resnet_EFT
import pandas as pd
from collections import defaultdict
import pickle
import copy
import numpy as np
import ctrl
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from DC_criterion import *
import umap
import plotly.express as px
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt



class args():
    dataset = "office"
    total_epoch = 100
    batch_size = 128
    seed = 3
    trte = "val"
    model_type = "independent"
    # model_type = "ours"
    t = 0  # choose which domain to set as target {0 to len(names)-1}
    model_path = 'style_param_models/office_100ep_lr50_128_save_best_val_model/{}_2_tsk_to_use_layerwise'.format(str(seed))


if __name__ == '__main__':
    # this version uses validation set during training to reduce bias
    random_seed = args.seed
    seed_torch(random_seed)
    device = 'cuda'
    args = args()
    # Load data
    dataset_name = args.dataset

    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.num_classes = 31

    for args.t in [0]:
        args.src = []
        for i in range(len(names)):
            if i == args.t:
                continue
            else:
                args.src.append(names[i])

        folder = './data/'
        args.s_dset_path = folder + args.dataset + '/' + names[args.t] + '_list.txt'

        chosen_models = []
        test_acc_final = []

        print(args.model_path)

        pre_tasks = args.src
        print("pre tasks", pre_tasks)
        cur_task = names[args.t]

        dset_loaders = data_load(args)
        train_loader, val_loader, test_loader = dset_loaders["source_tr"], dset_loaders["source_val"], dset_loaders["source_te"]
        dataloader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
        print(len(dataloader['train'].dataset.imgs))

        # partial_pre_tasks = ['dslr', 'webcam'] #0
        partial_pre_tasks = ['amazon', 'webcam'] #1
        # partial_pre_tasks = ['amazon', 'dslr'] #2
        #
        # partial_pre_tasks = pre_tasks
        print("pre tasks", partial_pre_tasks)

        # ours
        if args.model_type == "ours":
            model_path = 'style_param_models/office_100ep_lr50_128_save_best_val_model/3_2_tsk_to_use_layerwise_use_val_' \
                 'in_knn_shrink_suff_prev_1_1_bs128_wgt_decay_grid_search/task_amazon.pth'
            model = hid_mix.Net(partial_pre_tasks, args)
            num_classes = args.num_classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))
            dict = torch.load(model_path)['net']
            model.load_state_dict(dict)
            model = model.to(device)
        elif args.model_type == "finetune":
            model_path = 'style_param_models/office_100ep_lr50_128_save_best_val_model/3_2_tsk_to_use_layerwise_use_val_' \
                         'in_knn_shrink_suff_prev_1_1_bs128_wgt_decay_grid_search/task_dslr.pth'
            model = hid_mix.Net(partial_pre_tasks, args)
            num_classes = args.num_classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))
            dict = torch.load(model_path)['net']
            model.load_state_dict(dict)
            model = model.to(device)
        elif args.model_type == "independent":
            model_path = '/media/sijia/project/ComplexityofData/style_param_models/office_100ep_lr50_128_save_best_val_model' \
                      '/resneteft_independent_corrected_mixdatasets_val_0.001/' + 'amazon.pth'
            model = resnet_EFT.Net(args)
            dict = torch.load(model_path)['net']
            num_classes = args.num_classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))
            model.load_state_dict(dict)
            model = model.to(device)

        features = []
        labels = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                if inputs.shape[0] != 0:
                    if args.model_type == 'ours':
                        outputs, learned_feature, ref_features, first_feats = model(inputs)
                    else:
                        outputs, learned_feature = model(inputs)
                    features.append(learned_feature)
                    labels.append(targets)

        features = torch.concat(features).cpu()
        features = torch.squeeze(features).numpy()
        print(features.shape)
        labels = torch.concat(labels).cpu()
        labels = torch.squeeze(labels).numpy()
        print(labels)

        umap_2d = umap.UMAP(n_components=2, random_state=0)
        proj_2d = umap_2d.fit_transform(features)
        print(proj_2d.shape)
        n_colors = 31
        # colors = px.colors.sample_colorscale("turbo", [2* n / (n_colors - 1) for n in range(n_colors)])

        # fig_2d = px.scatter(
        #     proj_2d, x=0, y=1,
        #     color=labels.astype(str), color_discrete_sequence=colors, labels={'color': 'classes'}
        # )
        # fig_2d.show()
        palette = sns.color_palette(cc.glasbey, n_colors=n_colors)

        sns.scatterplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=labels, data=proj_2d, palette=palette)
        plt.legend(ncol=5, bbox_to_anchor=(1, 1))
        plt.show()