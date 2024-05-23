from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR100
from base.torchvision_dataset import TorchvisionDataset
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets.utils import check_integrity


class CIFAR100_noaug(TorchvisionDataset):

    def __init__(self, root: str, task_id=1, class_list=[]):
        super().__init__(root)
        self.size = 32
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Lambda(lambda x: class_list.index(x))

        self.train_set = MyCIFAR100(root=self.root, train=True, download=False,
                                   transform=transform, target_transform=target_transform, task_id=task_id,
                                   class_list=class_list)
        # print(len(self.train_set.targets))
        # get selected class from test set
        self.test_set = MyCIFAR100(root=self.root, train=False, download=False,
                                  transform=transform_test, target_transform=target_transform, task_id=task_id,
                                  class_list=class_list)


class CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root: str, task_id=1, class_list=[]):
        super().__init__(root)
        self.size = 32
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                 [0.2673, 0.2564, 0.2762]),
            # lambda x: torch.clamp(x, 0, 1)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                 [0.2673, 0.2564, 0.2762]),
            # lambda x: torch.clamp(x, 0, 1)
        ])

        target_transform = transforms.Lambda(lambda x: class_list.index(x))

        self.train_set = MyCIFAR100(root=self.root, train=True, download=False,
                                   transform=transform, target_transform=target_transform, task_id=task_id,
                                   class_list=class_list)
        # print(len(self.train_set.targets))
        # get selected class from test set
        self.test_set = MyCIFAR100(root=self.root, train=False, download=False,
                                  transform=transform_test, target_transform=target_transform, task_id=task_id,
                                  class_list=class_list)


class MyCIFAR100(CIFAR100):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(self, *args, task_id=1, class_list=[], **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

        if class_list == []:
            return

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        # print(self.targets)
        if self.train:
            labels = np.array(self.targets)
            exclude = np.array(class_list).reshape(1, -1)
            mask = (labels.reshape(-1, 1) == exclude).any(axis=1)

            self.data = self.data[mask]
            train_labels = labels[mask].tolist()
            num_partitions = 2
            data_size = len(train_labels)
            partition_size = int(data_size / num_partitions)
            if task_id in range(1, num_partitions + 1):
                self.data = self.data[(task_id - 1) * partition_size:task_id * partition_size]
                self.targets = train_labels[(task_id - 1) * partition_size:task_id * partition_size]
            else:
                assert "task id not in range"
        else:
            labels = np.array(self.targets)
            exclude = np.array(class_list).reshape(1, -1)
            mask = (labels.reshape(-1, 1) == exclude).any(axis=1)

            self.data = self.data[mask]
            test_labels = labels[mask].tolist()
            num_partitions = 2
            data_size = len(test_labels)
            partition_size = int(data_size / num_partitions)
            if task_id in range(1, num_partitions + 1):
                self.data = self.data[(task_id - 1) * partition_size:task_id * partition_size]
                self.targets = test_labels[(task_id - 1) * partition_size:task_id * partition_size]
            else:
                assert "task id not in range"

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """Override the original method of the CIFAR100 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class MyCIFAR100_mix(CIFAR100):
    def __init__(self, *args, task_id=1, class_list=[], subset_list=None, **kwargs):
        super(MyCIFAR100_mix, self).__init__(*args, **kwargs)

        if class_list == []:
            return
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        # print(self.targets)
        if self.train:
            data_subset = []
            train_labels = []
            for cls in class_list:
                labels = np.array(self.targets)
                exclude = np.array([cls]).reshape(1, -1)
                mask = (labels.reshape(-1, 1) == exclude).any(axis=1)

                data_tmp = self.data[mask]
                tmp_labels = labels[mask].tolist()
                num_partitions = 2
                data_size = len(tmp_labels)
                partition_size = int(data_size / num_partitions)
                if task_id in range(1, num_partitions + 1):
                    data_subset.extend(data_tmp[(task_id - 1) * partition_size:task_id * partition_size])
                    train_labels.extend(tmp_labels[(task_id - 1) * partition_size:task_id * partition_size])
                else:
                    assert "task id not in range"
        else:
            data_subset = []
            train_labels = []
            for cls in class_list:
                labels = np.array(self.targets)
                exclude = np.array([cls]).reshape(1, -1)
                mask = (labels.reshape(-1, 1) == exclude).any(axis=1)

                data_tmp = self.data[mask]
                tmp_labels = labels[mask].tolist()
                num_partitions = 2
                data_size = len(tmp_labels)
                partition_size = int(data_size / num_partitions)
                if task_id in range(1, num_partitions + 1):
                    data_subset.extend(data_tmp[(task_id - 1) * partition_size:task_id * partition_size])
                    train_labels.extend(tmp_labels[(task_id - 1) * partition_size:task_id * partition_size])
                else:
                    assert "task id not in range"
        self.data = data_subset
        self.targets = train_labels

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """Override the original method of the CIFAR100 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, index  # only line changed
        return img, target
