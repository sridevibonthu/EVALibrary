import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import torch
from tqdm.notebook import trange, tqdm


class LRRangeFinder():
  def __init__(self, model, epochs, start_lr, end_lr, dataloader, trainlen, batch_size):
    self.model = model
    self.epochs = epochs
    self.start_lr = start_lr
    self.end_lr = end_lr
    self.loss = []
    self.lr = []
    self.dataloader = dataloader
    self.trainlen = trainlen
    self.batch_size = batch_size
    
    
  def findLR(self):
    iter = 0
    smoothing = 0.05
    self.loss = []
    self.lr = []
    #print("Epochs - ", self.epochs)

    # Set up ptimizer and loss function for the experiment for our Resnet Model
    optimizer = torch.optim.SGD(self.model.parameters(), self.start_lr)
    criterion = nn.CrossEntropyLoss() 
    lr_lambda = lambda x: math.exp(x * math.log(self.end_lr / self.start_lr) / (self.epochs * self.trainlen/self.batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    for i in trange(self.epochs):
      print("epoch {}".format(i))
      for inputs, labels in tqdm(self.dataloader):
        
        # Send to device
        inputs = inputs.to(self.model.device)
        labels = labels.to(self.model.device)
        
        # Training mode and zero gradients
        self.model.train()
        optimizer.zero_grad()
        
        # Get outputs to calc loss
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update LR
        scheduler.step()
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        self.lr.append(lr_step)

        # smooth the loss
        if iter==0:
          self.loss.append(loss)
        else:
          loss = smoothing  * loss + (1 - smoothing) * self.loss[-1]
          self.loss.append(loss)
        
        iter += 1
        #print(iter, end="*")
      
    plt.ylabel("loss")
    plt.xlabel("Learning Rate")
    plt.xscale("log")
    plt.plot(self.lr, self.loss)
    plt.show()

    return(self.lr[self.loss.index(min(self.loss))])


