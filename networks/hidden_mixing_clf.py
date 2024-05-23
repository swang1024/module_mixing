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
        return nn.functional.linear(input, self.weight.exp()/torch.sum(self.weight.exp()), self.bias.exp()/torch.sum(self.bias.exp()))

    def __call__(self, input):
        return self.forward(input)


class ConvAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvAdapt, self).__init__()
        gp = 8
        pt = 16
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=int(p/gp), bias=True)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=int(p/pt),bias=True)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
            norm_layer: Optional[Callable[..., nn.Module]] = None
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
        self.lhc1 = ConvAdapt(planes, planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.lhc2 = ConvAdapt(planes, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = self._shortcut(x)

        out = self.conv1(x)
        out = self.lhc1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.lhc2(out)
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
            width_per_group: int = 64, pre_tasks=None, args=None, replace_stride_with_dilation: Optional[List[bool]] = None,
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
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model_paths = []
        prev_model_path = self.args.prev_model_path

        for m in range(len(self.pre_tasks)):
            path = prev_model_path + '/' + self.pre_tasks[m] + '.pth'
            self.model_paths.append(path)

        pre_model = resnet_EFT.Net(args)
        pre_num_classes, num_ftrs = 5, pre_model.fc.in_features
        pre_model.fc = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=pre_num_classes))
        self.pre_model = pre_model

        self.task_weights = ExpLinear(len(self.pre_tasks)+1, 1)
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for n, m in self.named_modules():
            if n.find("pre_model") >= 0:
                continue
            elif isinstance(m, ExpLinear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x_):
        x_list = []
        old_list = []
        for m in range(len(self.pre_tasks)):
            checkpoint = torch.load(self.model_paths[m])
            self.pre_model.load_state_dict(checkpoint['net'])
            self.pre_model.to('cuda')
            with torch.no_grad():
                _, e = self.pre_model(x_)
            x_list.append(e)
            old_list.append(e)

        x = self.conv1(x_)
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
        x_list = self.flatten(x_list)
        feats = self.relu1(x_list)

        out = self.fc(feats)

        return out, feats, old_list, probs


def Net(pre_tasks, args,  **kwargs):
    return hidden_mix_clf(BasicBlock, [2,2,2,2], pre_tasks=pre_tasks, args=args, **kwargs)


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


def load_model(prev_task_name, model, args):
    fname = args.model_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(fname + '/' + str(prev_task_name) + '.pth')
    checkpoint = torch.load(fname + '/' + str(prev_task_name) + '.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    return best_acc


def grad_false(model):
    for name, param in model.named_parameters():
        if name.find('gwc') >= 0 or name.find('pwc') >= 0 or name.find('task_weights') >= 0 or name.find('fc') >= 0:
            param.requires_grad = True
        else:
            param.requires_grad = False


def grad_false_just_mix_weights(model):
    for name, param in model.named_parameters():
        if name.find('task_weights') >= 0 or name.find('fc') >= 0:
            param.requires_grad = True
        else:
            param.requires_grad = False

