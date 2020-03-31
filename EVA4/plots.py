
import matplotlib.pyplot as plt
import torch
import numpy as np

def denormalize(tensor, mean, std):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
	    raise TypeError('tensor should be 4D')

	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
	return ret[0] if single_img else ret

def plot_images(img_data, classes, img_name, mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784]):
  figure = plt.figure(figsize=(10, 10))
  
  num_of_images = len(img_data)
  for index in range(1, num_of_images + 1):
      img = denormalize(img_data[index-1]["img"], mean, std)  # unnormalize
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(((np.transpose(img.cpu().numpy(), (1, 2, 0))*255).astype('uint8')))
      plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))
  
  plt.tight_layout()
  plt.savefig(img_name)
