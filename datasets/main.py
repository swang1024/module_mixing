from .cifar100 import CIFAR100_Dataset, MyCIFAR100_mix
from .TinyImageNet import TinyImageNet_Dataset, MyTinyImageNet
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def load_dataset(dataset_name, data_path, normal_class, test_class, subset_list=None):
    """Loads the dataset."""

    implemented_datasets = ('tiny_imagenet_50_overlap', 'cifar100_mix')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'cifar100_mix':
        dataset = CIFAR100_Dataset(root=data_path, task_id=normal_class, class_list=test_class)

    if dataset_name == 'tiny_imagenet_50_overlap':
        dataset = TinyImageNet_Dataset(root=data_path, task_id=normal_class, class_list=test_class)

    return dataset


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=(collate_fn or default_collate),
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )


def get_train_valid_loader(dataset_name, data_dir, class_list, task_id, batch_size, augment, random_seed, valid_size=0.1,
                           shuffle=True, num_workers=3, pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

    # define transforms
    valid_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            # lambda x: torch.clamp(x, 0, 1)
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            # lambda x: torch.clamp(x, 0, 1)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])

    target_transform = transforms.Lambda(lambda x: class_list.index(x))

    # load the dataset
    if dataset_name == "cifar100_mix":
        train_dataset = MyCIFAR100_mix(root=data_dir, train=True, download=False,
                                      transform=train_transform, target_transform=target_transform, task_id=task_id,
                                      class_list=class_list)

        valid_dataset = MyCIFAR100_mix(root=data_dir, train=True, download=False,
                                      transform=valid_transform, target_transform=target_transform, task_id=task_id,
                                      class_list=class_list)
    elif dataset_name == "tiny_imagenet_50_overlap":
        size = 64
        train_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        train_dataset = MyTinyImageNet(root=data_dir, mode="train", download=False,
                                      transform=train_transform, target_transform=target_transform, task_id=task_id,
                                      class_list=class_list)

        valid_dataset = MyTinyImageNet(root=data_dir, mode="train", download=False,
                                      transform=valid_transform, target_transform=target_transform, task_id=task_id,
                                      class_list=class_list)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print("train set size", len(train_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        # drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        # drop_last=True
    )

    return (train_loader, valid_loader)


def get_test_loader(dataset_name, data_dir, class_list, task_id, batch_size, shuffle=True, num_workers=4, pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
        # lambda x: torch.clamp(x, 0, 1),
    ])
    target_transform = transforms.Lambda(lambda x: class_list.index(x))

    if dataset_name == "cifar100_mix":
        dataset = MyCIFAR100_mix(root=data_dir, train=False, download=False,
                               transform=transform, target_transform=target_transform, task_id=task_id,
                               class_list=class_list)
    elif dataset_name == "tiny_imagenet_50_overlap":
        size = 64
        common_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        dataset = MyTinyImageNet(root=data_dir, mode="val", download=False,
                                      transform=common_transform, target_transform=target_transform, task_id=task_id,
                                      class_list=class_list)
        print("test set size", len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )

    return data_loader