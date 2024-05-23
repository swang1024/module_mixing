import os
import imageio
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from base.torchvision_dataset import TorchvisionDataset
from PIL import Image
import tqdm


class TinyImageNet_Dataset(TorchvisionDataset):

    def __init__(self, root: str, task_id=1, class_list=[]):
        super().__init__(root)
        self.size = 64
        train_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            # transforms.RandomRotation(20),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])

        common_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])

        target_transform = transforms.Lambda(lambda x: class_list.index(x))

        self.train_set = MyTinyImageNet(root=self.root, mode='train', download=False,
                                   transform=train_transform, target_transform=target_transform, task_id=task_id,
                                   class_list=class_list)

        # get selected class from test set
        self.test_set = MyTinyImageNet(root=self.root, mode='val', download=False,
                                  transform=common_transform, target_transform=target_transform, task_id=task_id,
                                  class_list=class_list)
    def loaders():
        return 

    # def loaders(self, batch_size: int, shuffle_train=True, shuffle_val=False,shuffle_test=False, num_workers: int = 0) -> (
    #         DataLoader, DataLoader, DataLoader):
    #     """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
    #     train = DataLoader(self.train_set, batch_size, shuffle_train)
    #     val = DataLoader(self.val_set, batch_size, shuffle_val)
    #     test = DataLoader(self.test_set, batch_size, shuffle_test)
    #     return train, val, test

class MyTinyImageNet(Dataset):
    def __init__(self, root, mode='train', preload=False, load_transform=None,
               transform=None, target_transform=None, download=False, max_samples=None, task_id=1, class_list=[]):
        tinp = TinyImageNetPaths(root, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.target_transform = target_transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        # self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.RandomState(seed=3).permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                   dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = Image.fromarray(img)
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

                # print(self.targets)

        self.data = np.array([x[0] for x in self.samples])
        self.targets = np.array([x[1] for x in self.samples])

        self.classes = tinp.ids
        self.class_to_idx = tinp.class_to_idx

        if self.mode == 'train':
            # part = (int(task_id)+1) % 2 + 1
            part = int(task_id)
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
                if part in range(1, num_partitions + 1):
                    data_subset.extend(data_tmp[(part - 1) * partition_size:part * partition_size])
                    train_labels.extend(tmp_labels[(part - 1) * partition_size:part * partition_size])
                else:
                    assert "task id not in range"
        elif self.mode == 'val':
            # part = (int(task_id)+1) % 2 + 1
            part = int(task_id)
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
                if part in range(1, num_partitions + 1):
                    data_subset.extend(data_tmp[(part - 1) * partition_size:part * partition_size])
                    train_labels.extend(tmp_labels[(part - 1) * partition_size:part * partition_size])
                else:
                    assert "task id not in range"
        
        self.data = data_subset
        self.targets = train_labels
        self.samples_num = len(self.data)

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            # lbl = None if self.mode == 'test' else self.label_data[idx]
            lbl = self.targets[idx]
        else:
            s = self.data[idx]
            img = imageio.imread(s)
            img = Image.fromarray(img)
            # lbl = None if self.mode == 'test' else self.targets[idx]
            lbl = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            lbl = self.target_transform(lbl)

        return img, lbl


"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', root_dir)
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
        self.ids = []
        self.class_to_idx = {}
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        # self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                        #   os.listdir(test_path)))
        # Get the validation paths and labels as test
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['test'].append((fname, label_id, nid, bbox))

        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            self.class_to_idx[nid] = label_id
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))