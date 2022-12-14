from tqdm import tqdm_notebook, tnrange
from eva4modelstats import ModelStats
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from loss import ssim, msssim
from utils import saveresults, show
import gc
import numpy as np

def compute_errors(gt, pred):
    print(np.min(gt),np.max(pred))
    theta = 0.00001
    gt = gt+theta
    pred = pred+theta
    print(gt.min(),gt.max(),pred.min(),pred.max())
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10



# https://github.com/tqdm/tqdm
class Train:
  def __init__(self, model, dataloader, optimizer, stats, scheduler=None, criterion1=None, criterion2=None, L1lambda = 0, tb=None):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.stats = stats
    self.L1lambda = L1lambda
    self.criterion1 = criterion1
    self.criterion2 = criterion2
    self.tb = tb
    # This is to store threshold accuracies(3),abs_rel,rmse,log_10
    self.metrics = {"mask": [list() for i in range(6)], "depth": [list() for i in range(6)]}
    self.size = 0

  def run(self):
    self.model.train()
    pbar = tqdm_notebook(enumerate(self.dataloader))
    for batch_idx, data in pbar:
      # get samples
      bg, fgbg, mask, depth = data['bg'].to(self.model.device), data['fgbg'].to(self.model.device), data['mask'].to(self.model.device), data['depth'].to(self.model.device)

      # Init
      self.optimizer.zero_grad()
      
      
      mask_pred, depth_pred = self.model(bg,fgbg)
      

      # Calculate loss
      if self.criterion1 is not None:
        loss1 = self.criterion1(mask_pred, mask)
        

        m_ssim = torch.clamp((1 - msssim(mask_pred, mask, normalize=True)) * 0.5, 0, 1)
    
        loss1 = (0.84 * m_ssim) + (0.16 * loss1)
      if self.criterion2 is not None:
        loss2 = self.criterion2(depth_pred, depth)
        

      d_ssim = torch.clamp( 1 - msssim(depth_pred, depth, normalize=True)*0.5, 0, 1)
      
    
      loss2 = (0.84 * d_ssim) + (0.16 * loss2)
      #print(loss1.item(), loss2.item(), d_ssim.item())
      loss = 2 * loss1 + loss2

      #Implementing L1 regularization
      if self.L1lambda > 0:
        reg_loss = 0.
        for param in self.model.parameters():
          reg_loss += torch.sum(param.abs())
        loss += self.L1lambda * reg_loss
      
      n = self.stats.get_batches()
      if n%500 == 0:
        self.tb.add_scalar('loss/train', loss.item(), n)
      
      #if (n+1) % 3000 == 0:
        #grid = torchvision.utils.make_grid(mask_pred.detach().cpu(), nrow=8, normalize=False)
        #self.tb.add_image('imagesmask', grid, n)
        #grid = torchvision.utils.make_grid(depth_pred.detach().cpu(), nrow=8, normalize=False)
        #self.tb.add_image('imagesdepth', grid, n)
      
      
        #saveresults(fgbg.detach().cpu(), "./plots/fgbg"+str(n+1)+".jpg", normalize=True)
        #saveresults(mask.detach().cpu(), "./plots/orimask"+str(n+1)+".jpg")
        #saveresults(depth.detach().cpu(), "./plots/oridepth"+str(n+1)+".jpg")
        #saveresults(mask_pred.detach().cpu(), "./plots/predmask"+str(n+1)+".jpg")
        #saveresults(depth_pred.detach().cpu(), "./plots/preddepth"+str(n+1)+".jpg")
	
		

      # Backpropagation
      loss.backward()
      self.optimizer.step()

      # Let's compute the metrics for every batch...
      mask1, mask_pred1, depth1, depth_pred1 = [], [], [], []
      for i in range(len(fgbg)):
        mask1.append(mask[i].detach().cpu())
        depth1.append(depth[i].detach().cpu())
        mask_pred1.append(mask_pred[i].detach().cpu())
        depth_pred1.append(depth_pred[i].detach().cpu())

      masks1 = np.stack(mask1, axis=0)
      mask_pred1 = np.stack(mask_pred1, axis=0)
      depth1 = np.stack(depth1, axis=0)
      depth_pred1 = np.stack(depth_pred1, axis=0)
      e1 = compute_errors(masks1, mask_pred1)
      e2 = compute_errors(depth1, depth_pred1)
      size = 1
      for i in mask.shape:
        size *= i
      self.size += size

      for i in range(6):
        self.metrics["mask"][i].append(e1[i])
        self.metrics["depth"][i].append(e2[i])
    

      # Update pbar-tqdm
      
      lr = 0.0
      if self.scheduler:
        lr = self.scheduler.get_last_lr()[0]
      else:
        # not recalling why i used sekf.optimizer.lr_scheduler.get_last_lr[0]
        lr = self.optimizer.param_groups[0]['lr']
      
      #lr =  if self.scheduler else (self.optimizer.lr_scheduler.get_last_lr()[0] if self.optimizer.lr_scheduler else self.optimizer.param_groups[0]['lr'])
      #print('lr for this batch:", lr)
      self.stats.add_batch_train_stats(loss.item(), 0, len(data), lr)
      pbar.set_description(self.stats.get_latest_batch_desc())
      if self.scheduler:
        self.scheduler.step()

    f = open(f"train.txt", "a")
    mask_metrics = [i/self.size for i in self.metrics["mask"]]
    depth_metrics = [i/self.size for i in self.metrics["depth"]]
    f.write(f"Mask {' '.join(mask_metrics)}")
    f.write(f"Depth {' '.join(depth_metrics)}")
    f.close()
    print("metrics for training set : ")
    print("\tMask")
    print(f"\t\t\t{' '.join(mask_metrics)}")
    print("\tDepth")
    print(f"\t\t\t{' '.join(depth_metrics)}")
      

class Test:
  def __init__(self, model, dataloader, stats, scheduler=None, criterion1=None, criterion2=None, tb=None):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.scheduler = scheduler
    self.loss=0.0
    self.criterion1 = criterion1
    self.criterion2 = criterion2
    self.tb = tb
	

  def run(self):
    self.model.eval()
    with torch.no_grad():
        
        for batch_idx, data in enumerate(self.dataloader):
            bg, fgbg, mask, depth = data['bg'].to(self.model.device), data['fgbg'].to(self.model.device), data['mask'].to(self.model.device), data['depth'].to(self.model.device)
            
            mask_pred, depth_pred = self.model(bg, fgbg)
            
            # Calculate loss
            if self.criterion1 is not None:
              loss1 = self.criterion1(mask_pred, mask)
            m_ssim = torch.clamp((1 - ssim(mask_pred, mask)) * 0.5, 0, 1)
    
            loss1 = (0.84 * m_ssim) + (0.16 * loss1)

            if self.criterion2 is not None:
                loss2 = self.criterion2(depth_pred, depth)

            d_ssim = torch.clamp((1 - ssim(depth_pred, depth)) * 0.5, 0, 1)
    
            loss2 = (0.84 * d_ssim) + (0.16 * loss2)

            self.loss = 2 * loss1 + loss2

            if batch_idx == 0:
              
              inp = fgbg.detach().cpu()
              orimp = mask.detach().cpu()
              mp = mask_pred.detach().cpu()
              oridp = depth.detach().cpu()
              dp = depth_pred.detach().cpu()
              print("First batch in testing fgbg, (mask, predicted mask), (depth, predicted depth)")
              show(inp[:8,:,:,:], normalize=True)
              mdinp = torch.cat([orimp[:8,:,:,:], mp[:8,:,:,:], oridp[:8,:,:,:], dp[:8,:,:,:]],dim=0)
              show(mdinp)
                       
            
            self.stats.add_batch_test_stats(self.loss.item(), 0, len(data))
        
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
              #print("hello yes i am ")
              self.scheduler.step(self.loss)

            
class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, statspath, scheduler=None, batch_scheduler=False, criterion1=None, criterion2=None, L1lambda = 0):
    self.tb = SummaryWriter()
    self.model = model
    
    #x = torch.rand(1,3,128,128)
    #self.tb.add_graph(self.model, x.to(self.model.device), x.to(self.model.device))
    self.scheduler = scheduler
    self.batch_scheduler = batch_scheduler
    self.optimizer = optimizer
    self.stats = ModelStats(model, statspath)
    self.criterion1 = criterion1
    self.criterion2 = criterion2
    self.train = Train(model, train_loader, optimizer, self.stats, self.scheduler if self.batch_scheduler else None, criterion1=criterion1, criterion2=criterion2, L1lambda=L1lambda, tb=self.tb)
    self.test = Test(model, test_loader, self.stats,self.scheduler, criterion1=criterion1, criterion2=criterion2, tb=self.tb)
	
  

  def run(self, epochs=10):
    pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
    for epoch in pbar:
      gc.collect()
      self.train.run()
      self.test.run()
      lr = self.optimizer.param_groups[0]['lr']
      self.stats.next_epochmaskdepth(lr)
      pbar.write(self.stats.get_epoch_desc())
      # need to ake it more readable and allow for other schedulers
      if self.scheduler and not self.batch_scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.scheduler.step()
        print(self.scheduler.get_last_lr())
      pbar.write(f"Learning Rate = {lr:0.6f}")
      self.tb.close()

    # save stats for later lookup
    #self.stats.save()
