import torch
import torch.nn as nn
import torch.nn.functional as F
from ../eva4net import Net

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