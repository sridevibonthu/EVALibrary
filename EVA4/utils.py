import os
import random
from PIL import Image
import torchvision
import torch
from matplotlib import pyplot as plt

def prepareData(root):
  #path = root+"/bgimages/"
  #bgimages = []
  #bgimages.extend([path+f for f in os.listdir(path) for i in range(4000)])
  #bgimages.sort()

  path = root+"/out2/images/"
  fgbgimages = [ (path+f) for f in os.listdir(path)]
  fgbgimages.sort()

  path = root+"/out2/masks/"
  maskimages = [ (path+f) for f in os.listdir(path)]
  maskimages.sort()

  path = root+"/out2/depth/"
  depthimages = [ (path+f) for f in os.listdir(path)]
  depthimages.sort()
  #print(len(bgimages))
  #return([bgimages, fgbgimages, maskimages, depthimages])
  dataset = list(zip(fgbgimages, fgbgimages, maskimages, depthimages))
  #dataset = list(zip(fgbgimages[:25000], fgbgimages[:25000], maskimages[:25000], depthimages[:25000]))
  random.shuffle(dataset)
  return dataset

def displayData(data, i):
  im1 = Image.open(data[i][0])
  im2 = Image.open(data[i][1])
  im3 = Image.open(data[i][2])
  im4 = Image.open(data[i][3])
  print(im1.size, im2.size, im3.size, im4.size)
  images = [im1, im2, im3, im4]
  widths, heights = zip(*(i.size for i in images))
  
  total_width = sum(widths)+30
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  offset = 0
  for im in images:
    new_im.paste(im, (offset, 0))
    offset += im.size[0]+10
  display(new_im)

  
def show(tensors, normalize=False, figsize=(15, 15), *args, **kwargs):
  grid_tensor=torchvision.utils.make_grid(tensors, normalize=normalize, *args, **kwargs)
  grid_image = grid_tensor.permute(1,2, 0)
  plt.figure(figsize = figsize)
  plt.imshow(grid_image)
  plt.xticks([])
  plt.yticks([])
  plt.show()

def saveresults(tensors, filename, normalize=False, figsize=(20,20), *args, **kwargs):
  grid_tensor = torchvision.utils.make_grid(tensors, normalize=normalize, *args, **kwargs)
  grid_image = grid_tensor.permute(1, 2, 0)
  plt.figure(figsize = figsize)
  plt.imshow(grid_image)
  plt.xticks([])
  plt.yticks([])
  plt.savefig(filename)
  plt.close()
