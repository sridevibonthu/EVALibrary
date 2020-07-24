from tqdm import tqdm_notebook, tnrange
from eva4modelstats import ModelStats
import torch.nn.functional as F
import torch

# https://github.com/tqdm/tqdm
class Train:
  def __init__(self, model, dataloader, optimizer, stats, scheduler=None, L1lambda = 0):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.stats = stats
    self.L1lambda = L1lambda

  def run(self):
    self.model.train()
    pbar = tqdm_notebook(self.dataloader)
    for data, target in pbar:
      # get samples
      data, target = data[0].to(self.model.device), target.to(self.model.device)

      # Init
      self.optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = self.model(data)
      target=target.view(-1)
      # Calculate loss
      loss = F.nll_loss(y_pred, target)

      #Implementing L1 regularization
      if self.L1lambda > 0:
        reg_loss = 0.
        for param in self.model.parameters():
          reg_loss += torch.sum(param.abs())
        loss += self.L1lambda * reg_loss


      # Backpropagation
      loss.backward()
      self.optimizer.step()

      # Update pbar-tqdm
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct = pred.eq(target.view_as(pred)).sum().item()
      lr = 0
      if self.scheduler:
        lr = self.scheduler.get_last_lr()[0]
      else:
        # not recalling why i used sekf.optimizer.lr_scheduler.get_last_lr[0]
        lr = self.optimizer.param_groups[0]['lr']
      
      #lr =  if self.scheduler else (self.optimizer.lr_scheduler.get_last_lr()[0] if self.optimizer.lr_scheduler else self.optimizer.param_groups[0]['lr'])
      
      self.stats.add_batch_train_stats(loss.item(), correct, len(data), lr)
      pbar.set_description(self.stats.get_latest_batch_desc())
      if self.scheduler:
        self.scheduler.step()

class Test:
  def __init__(self, model, dataloader, stats, scheduler=None):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.scheduler = scheduler
    self.loss=0.0
    self.average_loss=0.0

  def run(self):
    self.model.eval()
    with torch.no_grad():
        for data, target in self.dataloader:
            data, target = data[0].to(self.model.device), target.to(self.model.device)
            target=target.view(-1)
            output = self.model(data)
            self.loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            self.average_loss+=self.loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            self.stats.add_batch_test_stats(self.loss, correct, len(data))
        self.average_loss/=len(self.dataloader)
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
              #we are trying to step the scheduler using average loss
              self.scheduler.step(self.average_loss)

class Misclass:
  def __init__(self, model, dataloader, stats):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats

  def run(self):
    self.model.eval()
    with torch.no_grad():
        for data, target in self.dataloader:
          if len(self.stats.misclassified_images) == 25:
            return
          data, target = data[0].to(self.model.device), target.to(self.model.device)
          target=target.view(-1)
          output = self.model(data)
          loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          is_correct = pred.eq(target.view_as(pred))
          misclassified_inds = (is_correct==0).nonzero()[:,0]
          for mis_ind in misclassified_inds:
            if len(self.stats.misclassified_images) == 25:
               break
            self.stats.misclassified_images.append({"target": target[mis_ind].cpu().numpy(), "pred": pred[mis_ind][0].cpu().numpy(),"img": data[mis_ind]})
            
            
class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, statspath, scheduler=None, batch_scheduler=False, L1lambda = 0):
    self.model = model
    self.scheduler = scheduler
    self.batch_scheduler = batch_scheduler
    self.optimizer = optimizer
    self.stats = ModelStats(model, statspath)
    self.train = Train(model, train_loader, optimizer, self.stats, self.scheduler if self.batch_scheduler else None, L1lambda)
    self.test = Test(model, test_loader, self.stats,self.scheduler)
    self.misclass = Misclass(model, test_loader, self.stats)

  def run(self, epochs=10):
    pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
    for epoch in pbar:
      self.train.run()
      self.test.run()
      lr = self.optimizer.param_groups[0]['lr']
      self.stats.next_epoch(lr)
      pbar.write(self.stats.get_epoch_desc())
      # need to ake it more readable and allow for other schedulers
      if self.scheduler and not self.batch_scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.scheduler.step()
      pbar.write(f"Learning Rate = {lr:0.6f}")
    self.misclass.run()
    # save stats for later lookup
    self.stats.save()
