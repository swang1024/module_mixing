import os
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
import torch.nn.functional as F
import networks.resnet_EFT as resnet_EFT
from utils.train import feature_extractor
import copy
import time
from utils.utils_load_model import *
import gc
from DC_criterion import *


class ExpLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ExpLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        # print(self.weight.exp()/torch.sum(self.weight.exp()), self.bias.exp()/torch.sum(self.bias.exp()))
        # print(len(self.weight.exp()))
        return nn.functional.linear(input, self.weight.exp() / torch.sum(self.weight.exp()),
                                    self.bias.exp() / torch.sum(self.bias.exp()))

    def __call__(self, input):
        return self.forward(input)


class ConvAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvAdapt, self).__init__()
        gp = 8
        pt = 16
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=int(p / gp), bias=True)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=int(p / pt), bias=True)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            eft_modules: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.lhc1_list = nn.ModuleList()
        for m in range(len(eft_modules)):
            tmp1 = ConvAdapt(planes, planes, planes)
            layer_dict = tmp1.state_dict()
            dict_fix = {k.split('lhc1.')[1]: v for k, v in eft_modules[m].items() if
                        k.find('lhc1') >= 0}
            layer_dict.update(dict_fix)
            tmp1.load_state_dict(layer_dict)
            self.lhc1_list.append(tmp1)
        self.lhc1 = ConvAdapt(planes, planes, planes)
        self.lhc1_list.append(self.lhc1)
        self.lhc1_weights = ExpLinear(len(eft_modules) + 1, 1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.lhc2_list = nn.ModuleList()
        for m in range(len(eft_modules)):
            tmp2 = ConvAdapt(planes, planes, planes)
            layer_dict = tmp2.state_dict()
            dict_fix = {k.split('lhc2.')[1]: v for k, v in eft_modules[m].items() if
                        k.find('lhc2') >= 0}
            layer_dict.update(dict_fix)
            tmp2.load_state_dict(layer_dict)
            self.lhc2_list.append(tmp2)
        self.lhc2 = ConvAdapt(planes, planes, planes)
        self.lhc2_list.append(self.lhc2)
        self.lhc2_weights = ExpLinear(len(eft_modules) + 1, 1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = self._shortcut(x)

        out = self.conv1(x)
        out = [module(out) for module in self.lhc1_list]
        out = torch.stack(out)
        out = out.permute(1, 2, 3, 4, 0)
        out = self.lhc1_weights(out)
        out = torch.flatten(out, -2)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = [module(out) for module in self.lhc2_list]
        out = torch.stack(out)
        out = out.permute(1, 2, 3, 4, 0)
        out = self.lhc2_weights(out)
        out = torch.flatten(out, -2)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

    def _shortcut(self, x):
        if self.downsample is not None:
            x_s = self.downsample(x)
        else:
            x_s = x
        return x_s


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def load_eft_module(model):
    dict_model = model
    dict_eft = OrderedDict()
    for k, v in dict_model.items():
        if k.find('gwc') >= 0 or k.find('pwc') >= 0 or k.find('task_weights') >= 0 or k.find('fc') >= 0:
            dict_eft[k] = v
    return dict_eft


def load_backbone_module(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if k in model_dict}
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model


def add_eft_to_backbone(model, dict_eft):
    model_dict = model.state_dict()
    model_dict.update(dict_eft)
    model.load_state_dict(model_dict)
    model.eval()
    return model


class hidden_mix_clf(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000, groups: int = 1,
            width_per_group: int = 64, pre_tasks=None, args=None,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(hidden_mix_clf, self).__init__()
        self.pre_tasks = pre_tasks
        self.args = args
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.model_paths = []
        prev_model_path = self.args.prev_model_path

        for m in range(len(self.pre_tasks)):
            path = prev_model_path + '/' + self.pre_tasks[m] + '.pth'
            self.model_paths.append(path)

        self.eft_modules_lys = []
        for i in range(4):
            self.eft_modules_lys.append([])
            for j in range(2):
                self.eft_modules_lys[i].append([])
                for k in range(len(self.pre_tasks)):
                    new_state_dict = OrderedDict()
                    self.eft_modules_lys[i][j].append(new_state_dict)

        for m in range(len(self.pre_tasks)):
            model_dict = torch.load(self.model_paths[m])['net']
            for k, v in model_dict.items():
                if k.find('layer1.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[0][0][m][k] = v
                if k.find('layer1.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[0][1][m][k] = v
                if k.find('layer2.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[1][0][m][k] = v
                if k.find('layer2.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[1][1][m][k] = v
                if k.find('layer3.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[2][0][m][k] = v
                if k.find('layer3.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[2][1][m][k] = v
                if k.find('layer4.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[3][0][m][k] = v
                if k.find('layer4.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[3][1][m][k] = v

        self.layer1 = self._make_layer(block, 64, layers[0], eft_modules=self.eft_modules_lys[0][:][:])#self.eft_modules_lys[0][:][:])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       eft_modules=self.eft_modules_lys[1][:][:])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       eft_modules=self.eft_modules_lys[2][:][:])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       eft_modules=self.eft_modules_lys[3][:][:])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.task_weights = ExpLinear(len(self.pre_tasks)+1, 1)
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU(inplace=True)

        self.pre_model = resnet_EFT.Net(self.args)
        if args.dataset == 'office':
            num_classes = 31
        elif args.dataset == 'tiny_imagenet_50' or args.dataset == 'tiny_imagenet_50_overlap':
            num_classes = 50
        else:
            num_classes = 5
        num_ftrs = self.pre_model.fc.in_features
        self.pre_model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))
        self.fc = nn.Linear(512 * block.expansion, 1000)
        for m in self.modules():
            if isinstance(m, ExpLinear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, eft_modules=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                ConvAdapt(planes * block.expansion, planes * block.expansion, int(planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, eft_modules=eft_modules[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, eft_modules=eft_modules[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_list = []
        old_list = []
        for m in range(len(self.pre_tasks)):
            checkpoint = torch.load(self.model_paths[m])
            self.pre_model.load_state_dict(checkpoint['net'])
            self.pre_model.to('cuda')
            with torch.no_grad():
                _, e = self.pre_model(x)
            x_list.append(e)
            old_list.append(e)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        probs = torch.flatten(x, 1)
        x_list.append(probs)

        x_list = torch.stack(x_list)
        x_list = x_list.permute(1, 2, 0)
        x_list = self.task_weights(x_list)
        feats = self.flatten(x_list)
        out = self.fc(feats)

        return out, feats, old_list, probs


class cross_dataset_soup(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000, groups: int = 1,
            width_per_group: int = 64, tasks=None, args=None,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(cross_dataset_soup, self).__init__()
        self.tasks = tasks
        self.args = args
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.model_paths = []
        model_path = self.args.prev_model_path

        for m in range(len(self.tasks)):
            path = model_path + '/' + self.tasks[m] + '.pth'
            self.model_paths.append(path)

        self.eft_modules_lys = []
        for i in range(4):
            self.eft_modules_lys.append([])
            for j in range(2):
                self.eft_modules_lys[i].append([])
                for k in range(len(self.tasks)):
                    new_state_dict = OrderedDict()
                    self.eft_modules_lys[i][j].append(new_state_dict)

        for m in range(len(self.tasks)):
            model_dict = torch.load(self.model_paths[m])['net']
            for k, v in model_dict.items():
                if k.find('layer1.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[0][0][m][k] = v
                if k.find('layer1.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[0][1][m][k] = v
                if k.find('layer2.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[1][0][m][k] = v
                if k.find('layer2.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[1][1][m][k] = v
                if k.find('layer3.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[2][0][m][k] = v
                if k.find('layer3.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[2][1][m][k] = v
                if k.find('layer4.0') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[3][0][m][k] = v
                if k.find('layer4.1') >= 0 and k.find('lhc') >= 0:
                    self.eft_modules_lys[3][1][m][k] = v

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       eft_modules=self.eft_modules_lys[0][:][:])  # self.eft_modules_lys[0][:][:])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       eft_modules=self.eft_modules_lys[1][:][:])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       eft_modules=self.eft_modules_lys[2][:][:])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       eft_modules=self.eft_modules_lys[3][:][:])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.task_weights = ExpLinear(len(self.tasks), 1)
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if args.dataset == 'office':
            num_classes = 31
        else:
            num_classes = 5
        self.pre_model = resnet_EFT.Net(self.args)
        num_ftrs = self.pre_model.fc.in_features
        self.pre_model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes))

        for n, m in self.named_modules():
            if n.find('lhc1_list.{}'.format(str(0))) >= 0 or n.find('lhc2_list.{}'.format(str(0))) >= 0:
                continue
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, ExpLinear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, eft_modules=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                ConvAdapt(planes * block.expansion, planes * block.expansion, int(planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, eft_modules=eft_modules[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, eft_modules=eft_modules[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_list = []
        old_list = []
        for m in range(len(self.tasks)-1):
            # random_seed = self.args.seed
            # seed_torch(random_seed)
            checkpoint = torch.load(self.model_paths[m])
            self.pre_model.load_state_dict(checkpoint['net'])
            self.pre_model.to('cuda')
            with torch.no_grad():
                _, e = self.pre_model(x)
            x_list.append(e)
            old_list.append(e)

        x_c1 = self.conv1(x)
        x1 = self.bn1(x_c1)
        x2 = self.relu(x1)
        x3 = self.maxpool(x2)
        x_1 = self.layer1(x3)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        xf = self.avgpool(x_4)
        probs = torch.flatten(xf, 1)
        x_list.append(probs)

        x_list = torch.stack(x_list)
        x_list = x_list.permute(1, 2, 0)
        x_list = self.task_weights(x_list)
        feats = self.flatten(x_list)

        out = self.fc(feats)

        return out, feats, old_list, probs


def Net(pre_tasks, args, **kwargs):
    return hidden_mix_clf(BasicBlock, [2, 2, 2, 2], pre_tasks=pre_tasks, args=args, **kwargs)


def model_soup_Net(tasks, args, **kwargs):
    return cross_dataset_soup(BasicBlock, [2, 2, 2, 2], tasks=tasks, args=args, **kwargs)


def save_model(cur_task_name, acc, model, args, pre_tasks, loss):
    print('Saving..', acc, loss)
    statem = {
        'net': model.state_dict(),
        'acc': acc,
        'pre_tasks': pre_tasks,
    }
    fname = args.model_path
    if not os.path.isdir(fname):
        os.makedirs(fname)
    torch.save(statem, fname + '/' + str(cur_task_name) + '.pth')


def save_model_grid_search(cur_task_name, acc, model, args, lr, wgt_decay, pre_tasks, loss):
    print('Saving..', acc, loss)
    statem = {
        'net': model.state_dict(),
        'acc': acc,
        'lr': lr,
        'wgt_decay': wgt_decay,
        'pre_tasks': pre_tasks,
    }
    fname = args.model_path
    if not os.path.isdir(fname):
        os.makedirs(fname)
    torch.save(statem, fname + '/' + str(cur_task_name) + '.pth')


def load_model(prev_task_name, model, args):
    fname = args.model_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(fname + '/' + str(prev_task_name) + '.pth')
    checkpoint = torch.load(fname + '/' + str(prev_task_name) + '.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    return best_acc


def grad_false(model, n):
    for name, param in model.named_parameters():
        if name.find('prev_model') < 0 and \
                ((name.find('layer4') == -1 and name.find('lhc1_list.{}'.format(str(n))) >= 0) \
                or (name.find('layer4') == -1 and name.find('lhc2_list.{}'.format(str(n))) >= 0) \
                or (name.find('layer4') >= 0 and name.find('lhc1_list.{}'.format(str(n))) >= 0) \
                or (name.find('layer4') >= 0 and name.find('lhc2_list.{}'.format(str(n))) >= 0) \
                or (name.find('downsample') >= 0 and (name.find('gwc') >= 0 or name.find('pwc') >= 0)) \
                or name.find('lhc1_weights') >= 0 \
                or name.find('lhc2_weights') >= 0 \
                or name.find('task_weights') >= 0 \
                or name.find('fc') >= 0):
            param.requires_grad = True
        else:
            param.requires_grad = False


def grad_false_only_weights(model):
    for name, param in model.named_parameters():
        if name.find('lhc1_weights') >= 0 \
                or name.find('lhc2_weights') >= 0 \
                or name.find('task_weights') >= 0 \
                or (name.find("pre_model") < 0 <= name.find('fc')):
            param.requires_grad = True
        else:
            param.requires_grad = False


def select_new_modules(model, n, lr, wgt_decay):
    params = []
    for name, param in model.named_parameters():
        if (name.find('lhc1_list.{}'.format(str(n))) >= 0) \
                or (name.find('lhc2_list.{}'.format(str(n))) >= 0) \
                or (name.find('downsample') >= 0 and (name.find('gwc') >= 0 or name.find('pwc') >= 0)) \
                or name.find('fc') >= 0:
            param.requires_grad = True
            params += [{"params": param, 'lr': lr, 'weight_decay': wgt_decay}]
        else:
            param.requires_grad = False
    return params


def select_mixing_weights(model, lr, wgt_decay):
    params = []
    for name, param in model.named_parameters():
        if name.find('lhc1_weights') >= 0 \
                or name.find('lhc2_weights') >= 0 \
                or name.find('task_weights') >= 0:
            param.requires_grad = True
            params += [{"params": param, 'lr': lr, 'weight_decay': wgt_decay}]
        else:
            param.requires_grad = False
    return params


def train(train_loader, epoch, model, args, optimizer, criterion):
    model.train()
    train_loss = 0
    ce_loss = 0
    dc_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, learned_feature, ref_features, first_feats = model(inputs)
        loss, loss_ce, loss_dc, DC_results = criterion(outputs, targets, first_feats, ref_features)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        ce_loss += loss_ce.item()
        dc_loss += loss_dc
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total

    if epoch % 9 == 0:
        print('\nEpoch: %d' % epoch)
        print("[Train: ], [%d/%d: ], [Accuracy: %f], [Loss: %f] [CE Loss: %f] [DC Loss: %f] [Lr: %f]"
              % (epoch, args.total_epoch, acc, train_loss / (batch_idx + 1), ce_loss / (batch_idx + 1),
                 dc_loss / (batch_idx + 1),
                 optimizer.param_groups[0]['lr']), flush=True)

import torch.optim as optim
from torch.optim import lr_scheduler


def val_test(val_test_loader, epoch, cur_task_name, model, args, criterion, best_acc, best_model_, pre_tasks):
    model.eval()
    test_loss = 0
    ce_loss = 0
    dc_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if inputs.shape[0] != 0:
                outputs, learned_feature, ref_features, first_feats = model(inputs)
                loss, loss_ce, loss_dc, DC_results = criterion(outputs, targets, first_feats, ref_features)

                test_loss += loss.item()
                ce_loss += loss_ce.item()
                dc_loss += loss_dc
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total

    if epoch % 9 == 0:
        print("[Test Accuracy: %f], [Loss: %f], [CE Loss: %f], [DC Loss: %f] " % (acc, test_loss / (batch_idx + 1),
                                                                                  ce_loss / (batch_idx + 1),
                                                                                  dc_loss / (batch_idx + 1)), flush=True)

    if acc >= best_acc:
        best_acc = acc
        best_model_ = copy.deepcopy(model)
        loss = test_loss / (batch_idx + 1)
        dc_loss = dc_loss / (batch_idx + 1)
        print(acc, loss, dc_loss, flush=True)
        # save_model(cur_task_name, acc, model, args, pre_tasks, loss)

    return best_acc, best_model_


def test(test_loader, cur_task_name, model, args, criterion):
    model.eval()
    test_loss = 0
    ce_loss = 0
    dc_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if inputs.shape[0] != 0:
                outputs, learned_feature, ref_features, first_feats = model(inputs)
                loss, loss_ce, loss_dc, DC_results = criterion(outputs, targets, first_feats, ref_features)

                test_loss += loss.item()
                ce_loss += loss_ce.item()
                dc_loss += loss_dc
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total

    print("[Test Accuracy: %f], [Loss: %f], [CE Loss: %f], [DC Loss: %f] " % (acc, test_loss / (batch_idx + 1),
                                                                              ce_loss / (batch_idx + 1),
                                                                              dc_loss / (batch_idx + 1)), flush=True)

    return acc
