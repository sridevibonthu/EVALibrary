from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


class AlbumentationTransforms:
  """
  Helper class to create test and train transforms using Albumentations
  """
  def __init__(self, transforms_list=[]):
    transforms_list.append(AP.ToTensor())
    
    self.transforms = A.Compose(transforms_list)


  def __call__(self, img):
    img = np.array(img)
    #print(img)
    return self.transforms(image=img)['image']



class Transforms:
  """
  Helper class to create test and train transforms
  """
  def __init__(self, normalize=False, mean=None, stdev=None):
    if normalize and (not mean or not stdev):
      raise ValueError('mean and stdev both are required for normalize transform')
  
    self.normalize=normalize
    self.mean = mean
    self.stdev = stdev

  def test_transforms(self):
    transforms_list = [transforms.ToTensor()]
    if(self.normalize):
      transforms_list.append(transforms.Normalize(self.mean, self.stdev))
    return transforms.Compose(transforms_list)

  def train_transforms(self, pre_transforms=None, post_transforms=None):
    if pre_transforms:
      transforms_list = pre_transforms
    else:
      transforms_list = []
    transforms_list.append(transforms.ToTensor())

    if(self.normalize):
      transforms_list.append(transforms.Normalize(self.mean, self.stdev))
    if post_transforms:
      transforms_list.extend(post_transforms)
    return transforms.Compose(transforms_list)    