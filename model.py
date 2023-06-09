import torch
from torch import nn
import torch.nn.functional as F

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x
    
class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 512

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x

class NicoEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 4, 3, stride=1, padding=0) 
        self.bn4 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 12, 3, stride=3, padding=0) 
        self.bn12 = nn.BatchNorm2d(12)
        self.conv4 = nn.Conv2d(12, 16, 3, stride=1, padding=0) 
        self.bn16 = nn.BatchNorm2d(16)
    
    def forward(self, x):
        x = F.gelu(self.bn3(self.conv1(x)))
        x = F.gelu(self.bn4(self.conv2(x)))
        x = F.gelu(self.bn12(self.conv3(x)))
        x = F.gelu(self.bn16(self.conv4(x)))
        return x
    
class NicoDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1T = nn.ConvTranspose2d(16, 12, 3, stride=1, padding=0)
        self.bn1T = nn.BatchNorm2d(12)
        self.conv2T = nn.ConvTranspose2d(12, 4, 3, stride=3, padding=0)
        self.bn2T = nn.BatchNorm2d(4)
        self.conv3T = nn.ConvTranspose2d(4, 3, 3, stride=1, padding=0)
        self.bn3T = nn.BatchNorm2d(3)
        self.conv4T = nn.ConvTranspose2d(3, 3, 5, stride=1, padding=0)
    
    def forward(self, x):
        x = F.gelu(self.bn1T(self.conv1T(x)))
        x = F.gelu(self.bn2T(self.conv2T(x)))
        x = F.gelu(self.bn3T(self.conv3T(x)))
        x = F.gelu(self.conv4T(x))
        return x

class AE(nn.Module):

    def __init__(self, model='resnet18'):
        super().__init__()
        if model == 'resnet18':
            self.encoder = ResNet18Enc()
            self.decoder = ResNet18Dec()
        elif model == 'nicoAE':
            self.encoder = NicoEnc()
            self.decoder = NicoDec()
        else:
            print('Non implemented')

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
