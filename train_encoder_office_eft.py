import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datasets.office import data_load
import networks.resnet_EFT as resnet_EFT
import networks.resnet as resnet
torch.set_printoptions(precision=5,sci_mode=False)
from utils.utils_load_model import *


class args():
    dataset = "office"
    data_root = './data/office'
    total_epoch = 100
    batch_size = 128
    seed = 3
    trte = "val"
    model_path = 'style_param_models/office_100ep_lr0.001_bs128_save_best_val_model' \
                 '/resneteft_independent'


if __name__ == '__main__':
    random_seed = 3
    seed_torch(random_seed)
    args = args()

    root = args.model_path
    # Create missing directories
    if not os.path.exists(root):
        os.makedirs(root)

    # Load data
    dataset_name = args.dataset

    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

    args.src = []
    for i in range(len(names)):
        args.src.append(names[i])

    model_name = "resneteft"
    task_model = []
    test_final_acc = []
    task_index = 0
    for ii, cur_task in enumerate(args.src):
        folder = './data/'
        args.s_dset_path = os.path.join(folder, args.dataset, cur_task, 'image_unida_list.txt')
        args.t = ii  # which domain to train on

        dset_loaders = data_load(args)
        train_loader, val_loader, test_loader = dset_loaders["source_tr"], dset_loaders["source_val"], dset_loaders[
            "source_te"]
        dataloader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}
        print(len(dataloader['train'].dataset.imgs))

        print(len(train_loader.dataset))
        # train resnet for feature extraction
        if model_name == 'resneteft':
            print("\nLoading resnet18 for finetuning ...\n")
            # Load a pretrained model - Resnet18
            model_ft = resnet_EFT.Net(args)
            dict = torch.load('resnet18.pth')
            model_ft = model_equal_part_embed(model_ft, dict)

            # Modify fc layers to match num_classes
            num_classes = args.class_num
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=num_classes)
            )
        else:
            model_ft = resnet.resnet18(pretrained=True)
            # Modify fc layers to match num_classes
            num_classes = args.class_num
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))

        # Transfer the model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('GPU')
            model_ft.to(device)
        else:
            print("CPU")

        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Optimizer
        optimizer = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-5)
        # Learning rate decay
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

        # Model training routine
        print("\nTraining:-\n")
        # Train the mo del

        # Number of epochs
        dataloader = {'train': train_loader, 'valid': test_loader}

        resnet_EFT.grad_false(model_ft)
        get_parameter_number(model_ft)

        best_acc = 0
        best_model = copy.deepcopy(model_ft)
        since = time.time()
        for epoch in range(args.total_epoch):
            resnet_EFT.train(train_loader, epoch, model_ft, args, optimizer, criterion)
            best_acc, best_model = resnet_EFT.val_test(val_loader, epoch, model_ft, criterion, best_acc, best_model)
            exp_lr_scheduler.step()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        task_model.append(copy.deepcopy(model_ft))
        print("best validation acc: ", best_acc)
        best_test_acc = resnet_EFT.test(test_loader, best_model, criterion)
        resnet_EFT.save_model(cur_task, best_test_acc, best_model, args)
        test_final_acc.append(best_test_acc)
        print(test_final_acc)
        print('Done!')
        ###################################
