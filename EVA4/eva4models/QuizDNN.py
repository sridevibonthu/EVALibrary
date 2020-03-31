import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, num_convs=3):
        super(BasicBlock, self).__init__()
        self.num_convs = num_convs
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.expand1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out1 = self.conv1(F.relu(self.bn1(x)))
        # ensure of x is same size as out1
        if self.num_convs == 3:
          x = self.expand1(x)

        x = x + out1

        out2 = self.conv2(F.relu(self.bn2(x)))
        x = x + out2

        out3 = self.conv3(F.relu(self.bn3(x)))
        
        return x, out1, out2, out3


class QuizDNN(Net):
    def __init__(self, num_classes=10, name="QuizDNN"):
        super(QuizDNN, self).__init__(name)
        self.in_planes = 64
        self.num_classes = num_classes
        # Input Convolution
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(self.in_planes, 2)

        # first pooling
        self.pool1 = nn.MaxPool2d((2,2))
        self.layer2 = self._make_layer(self.in_planes*2)

        # second pooling
        self.pool2 = nn.MaxPool2d((2,2))
        self.layer3 = self._make_layer(self.in_planes*2)
        
        # 1x1 before softmax
        self.conv10 = self.create_conv2d(self.in_planes, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 512 OUT:10

    def _make_layer(self, planes, num_convs=3):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, num_convs))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x123, _, __, ___ = self.layer1(x1) 
        x4 = self.pool1(x123) #maxpool of x1+x2+x3
        
        x456, x5, x6, x7 = self.layer2(x4)

        x8 = self.pool2(x5+x6+x7)

        x8910, x9, x10, x11 = self.layer3(x8)

        out = F.adaptive_avg_pool2d(x11, 1)
        out = self.conv10(out)

        out = out.view(-1, self.num_classes)
        return F.log_softmax(out, dim=-1)



def test():
    net = QuizDNN()
    y = net(torch.randn(1,3,32,32))
    print(y.size())