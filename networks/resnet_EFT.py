import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torch import autograd
from torch.autograd import Variable
from datasets.main import get_data_loader
import os
import torch.nn.functional as F
torch.set_printoptions(precision=5,sci_mode=False)
import copy
import collections


class ConvAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvAdapt, self).__init__()
        gp = 8
        pt = 16
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=int(p/gp), bias=True)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=int(p/pt), bias=True)

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


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
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
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # # new head
        # self.fc=torch.nn.ModuleList()
        # for t in range(100):
        #     self.fc.append(nn.Linear(512 * block.expansion, 5))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

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

    def forward(self, x):
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
        x = self.fc(probs)
        # probas = F.softmax(x, dim=1)

        return x, probs

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def Net(args, **kwargs):
    return _resnet(BasicBlock, [2,2,2,2], **kwargs)


def save_model(cur_task_name, acc, model, args):
    print('Saving..')
    statem = {
        'net': model.state_dict(),
        'acc': acc,
    }
    fname = args.model_path
    if not os.path.isdir(fname):
        os.makedirs(fname)
    torch.save(statem, fname + '/' + str(cur_task_name) + '.pth')


def save_model_grid_search(cur_task_name, acc, model, args, lr, wgt_decay, loss):
    print('Saving..', acc, loss)
    statem = {
        'net': model.state_dict(),
        'acc': acc,
        'lr': lr,
        'wgt_decay': wgt_decay
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


def train(train_loader, epoch, model, args, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # print("target", targets)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        # print(outputs)
        # print(targets.size())

        ce_loss = criterion(outputs, targets)
        loss = ce_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        # print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
    
    if epoch % 9 == 0:
        print('\nEpoch: %d' % epoch)
        print("[Train: ], [%d/%d: ], [Accuracy: %f], [Loss: %f] [Lr: %f]"
	  % (epoch, args.total_epoch, acc, train_loss / (batch_idx+1),
	     optimizer.param_groups[0]['lr']))

def val_test(val_test_loader, epoch, model, criterion, best_acc, best_model_):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if inputs.shape[0] != 0:
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total

    if epoch % 9 == 0:
        print("[Test Accuracy: %f], [Loss: %f]" % (acc, test_loss / (batch_idx + 1)))

    if acc >= best_acc:
        best_acc = acc
        best_model_ = copy.deepcopy(model)
        loss = test_loss / (batch_idx + 1)
        print(acc, loss)
        # save_model(cur_task_name, acc, model, args, pre_tasks, loss)

    return best_acc, best_model_


def test(test_loader, model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if inputs.shape[0] != 0:
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total

    print("[Test Accuracy: %f], [Loss: %f]" % (acc, test_loss / (batch_idx + 1)))

    return acc


def test_cl(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # targets = targets1 - task * args.class_per_task
            # if task > 0:
            #     correct_sample, Ncorrect, _ = check_task(task, inputs, model, task_model)
            #     tcorrect += Ncorrect
            #     inputs = inputs[correct_sample]
            #     targets = targets[correct_sample]
            if inputs.shape[0] != 0:
                outputs, _ = model(inputs)
                # loss = criterion(outputs, targets)
                # test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    # if task > 0:
    #     taskC = tcorrect.item() / total
    # else:
    #     taskC = 1.0
    acc = 100. * correct / total

    return acc


# this false the gradient of the global parameter after the 1st task
def grad_false(model):
    for name, param in model.named_parameters():
        if name.find('gwc') >= 0 or name.find('pwc') >= 0 or name.find('task_weights') >= 0 or name.find('fc') >= 0:
            param.requires_grad = True
        else:
            param.requires_grad = False



def grad_true_only_fc(model):
    for name, param in model.named_parameters():
        if name.find('fc') >= 0:
            param.requires_grad = True
        else:
            param.requires_grad = False
