#from eva4net import Net

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4modeltrainer import ModelTrainer

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, stride=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)]

    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, stride=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True,max_pooling=0):
      if(max_pooling>0):
        l.append(nn.MaxPool2d(2,2))
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))
      if relu:
        l.append(nn.ReLU())

      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, stride=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros",max_pooling=0):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu,max_pooling)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)

    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, criterion1=None, criterion2=None, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, criterion1=criterion1, criterion2=criterion2, L1lambda=L1lambda)
      self.trainer.run(epochs)

 
    
    # to initialize parameters
    def init_params(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.ConvTranspose2d):
          torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
          torch.nn.init.constant_(m.weight, 1)
          torch.nn.init.constant_(m.bias, 0)
      print("Model parameters initialized")



class DownSampling(Net):
  def __init__(self, inc, outc, dropout_value=0):
    super(DownSampling, self).__init__()
    self.conviden = self.create_conv2d(in_channels = inc, out_channels=outc, kernel_size=1, stride=2, padding=0 )
    self.conv1 = self.create_conv2d(in_channels = inc, out_channels=outc, stride=2, groups=inc)
    self.conv11 = self.create_conv2d(in_channels = outc, out_channels=outc, kernel_size=1, padding=0)
    self.conv2 = self.create_conv2d(in_channels = outc, out_channels=outc, groups = outc)
    self.conv21 = self.create_conv2d(in_channels = outc, out_channels=outc, kernel_size=1, padding=0)

  def forward(self,x):
    x1 = self.conv21(self.conv2(self.conv11(self.conv1(x))))
    x2 = self.conviden(x)
    return(x1+x2)


class UpSampling(Net):
  def __init__(self, inc, outc):
    super(UpSampling, self).__init__()
    self.conv1 = self.create_conv2d(in_channels=outc, out_channels=outc, groups=outc)
    self.conv2 = self.create_conv2d(in_channels=outc, out_channels=outc)
    self.conv21 = self.create_conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0)
    self.tconv = nn.ConvTranspose2d(inc, outc, kernel_size=3, stride=1, padding=1, bias=False)
    
  def forward(self,x):
    x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = self.tconv(x)
    x = self.conv21(self.conv2(self.conv1(x)))
    return x
	
class Encoder(Net):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = self.create_conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.down1= DownSampling(32, 64)
    self.down2 = DownSampling(64,128)
    self.down3 = DownSampling(128, 256)
    self.down4 = DownSampling(256, 512)
    
  def forward(self,x):
    out0 = self.conv1(x)
    out1 = self.down1(out0)
    out2 = self.down2(out1)
    out3 = self.down3(out2)
    out4 = self.down4(out3)
    return out0, out1, out2, out3, out4
    #return out0, out1, out2, out3
	
class Decoder(Net):
  def __init__(self):
    super(Decoder, self).__init__()
    self.up1 = UpSampling(512,256)
    self.up2 = UpSampling(256, 128)
    self.up3 = UpSampling(128,64)
    self.up4 = UpSampling(64, 32)
    self.convend1 = self.create_conv2d(32, 32, kernel_size=3, padding=1)
    self.convend2 = self.create_conv2d(32, 1, kernel_size=1, padding=0, bn=False, relu=False)
    
        
  def forward(self, x0, x1, x2, x3,x4):
    y = x3 + self.up1(x4)
    y = x2 + self.up2(x3)
    y = x1 + self.up3(y)
    y = x0 + self.up4(y)
    #y = self.up4(y)
    y = self.convend1(y)
    y = self.convend2(y)
    return(y)
	
class S15Model(Net):
  def __init__(self):
    super(S15Model, self).__init__()
    self.encoder = Encoder()
    self.decoder1 = Decoder()
    self.decoder2 = Decoder()
    self.init_params()
        
  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    e = self.encoder(x)
    mask = self.decoder1(*e)
    depth = self.decoder2(*e)
    return(mask, depth)
