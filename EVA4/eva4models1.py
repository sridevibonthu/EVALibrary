from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4modeltrainer import ModelTrainer

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]

    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True):
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))
      if relu:
        l.append(nn.ReLU())

      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)

    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None

class Cfar10Net(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 32, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x32, RF = 3
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 32x32x32, OUT 32x32x32, RF = 5
        self.conv3 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 32x32x32, OUT 32x32x32, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 16x16x32, OUT 16x16x64, RF = 12
        self.conv5 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 16x16x64, OUT 16x16x64, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv6 = self.create_conv2d(64, 128, dropout=dropout_value) # IN 8x8x64, OUT 8x8x128, RF = 26
        self.conv7 = self.create_conv2d(128, 128, dropout=dropout_value) # IN 8x8x128, OUT 8x8x128, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        
        self.conv8 = self.create_depthwise_conv2d(128, 256, dropout=dropout_value) # IN 4x4x128, OUT 4x4x256, RF = 54
        self.conv9 = self.create_depthwise_conv2d(256, 256, dropout=dropout_value) # IN 4x4x256, OUT 4x4x256, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(256, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net2(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net2, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 5
        self.conv3 = self.create_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_conv2d(32, 64, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net3(Net):
    def __init__(self, name="Cfar10Net3", dropout_value=0):
        super(Cfar10Net3, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_depthwise_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_depthwise_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 5
        self.conv3 = self.create_depthwise_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_depthwise_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_depthwise_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_depthwise_conv2d(32, 64, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64
        self.conv6 = self.create_depthwise_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_depthwise_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



class Cfar10Net4(Net):
    def __init__(self, name="Cfar10Net4", dropout_value=0):
        super(Cfar10Net4, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value, dilation=2, padding=2) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net5(Net):
    def __init__(self, name="Cfar10Net5", dropout_value=0):
        super(Cfar10Net5, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value, dilation=2, padding=2) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value, dilation=2, padding=2) # IN 16x16x16, OUT 16x16x32, RF = 16
        #self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64, RF = 34
        #self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 70
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 86

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv4(x)
        #x = self.conv5(x)

        x = self.pool2(x)
        x = self.conv6(x)
        #x = self.conv7(x)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class MnistNet(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(MnistNet, self).__init__(name)

        # Input Convolution Block
        self.convblock1 = self.create_conv2d(3, 32, dropout=dropout_value, groups=3) # input_size = 28, output_size = 28, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = self.create_conv2d(10, 10, dropout=dropout_value) # output_size = 28, RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6

        self.convblock3 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 12, RF = 10

        # CONVOLUTION BLOCK 2
        self.convblock4 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 10, RF = 14

        self.convblock5 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 8, RF = 18

        self.convblock6 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 6, RF = 22

        self.convblock7 = self.create_conv2d(10, 16, dropout=dropout_value, padding=0, bn=False, relu=False) # output_size = 4, RF = 26

        # OUTPUT BLOCK
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 

        self.convblock8 = self.create_conv2d(16, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # output_size = 1, RF = 26
        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_planes, planes, stride=1 ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    






class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.trainer = None
        self.name = "Models"

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.gap=nn.AvgPool2d(kernel_size=(4,4))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
        
    def summary(self, input_size):
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
