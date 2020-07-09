import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, Subset
from utils.utils0 import logging, timeLog

#----------------------------------------------------------
def get_ds_attr(dataset):
   imgsz = 32; channels = 3
   if dataset == 'CIFAR10' or dataset == 'SVHN':
      nclass = 10
   elif dataset == 'CIFAR100':
      nclass = 100
   elif dataset == 'ImageNet': 
      nclass = 1000; imgsz = 224
   elif dataset == 'MNIST':
      nclass = 10; channels = 1
   elif dataset.endswith('64'):
      nclass = 1; imgsz = 64
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

   return { "nclass": nclass, "image_size": imgsz, "channels": channels }

#----------------------------------------------------------
def gen_lsun_balanced(dataroot, nms, tr, indexes):
   sub_dss = []
   for i,nm in enumerate(nms):
      sub_dss += [Subset(datasets.LSUN(dataroot, classes=[nm], transform=tr), indexes)]
   return ConcatUniClassDataset(sub_dss)

#----------------------------------------------------------
# Concatenate uni-class datasets into one dataset. 
class ConcatUniClassDataset:
   def __init__(self, dss):
      self.dss = dss
      self.top = [0]
      num = 0
      for ds in self.dss:
         num += len(ds)
         self.top += [num]
         
   def __len__(self):
      return self.top[len(self.top)-1]
         
   def __getitem__(self, index):
      cls = -1
      for i,top in enumerate(self.top):
         if index < top:
            cls = i-1
            break
      if cls < 0:
         raise IndexError
         
      return ((self.dss[cls])[index-top][0], cls)
         
#----------------------------------------------------------
def get_ds(dataset, dataroot, is_train, do_download, do_augment):
   tr = get_tr(dataset, is_train, do_augment)
   if dataset == 'SVHN':
      if is_train:
         train_ds = datasets.SVHN(dataroot, split='train', transform=tr, download=do_download)
         extra_ds = datasets.SVHN(dataroot, split='extra', transform=tr, download=do_download)
         return ConcatDataset([train_ds, extra_ds])
      else:
         return datasets.SVHN(dataroot, split='test', transform=tr, download=do_download) 
   elif dataset == 'MNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)    
   elif dataset.startswith('lsun_') and dataset.endswith('64'):
      nm = dataset[len('lsun_'):len(dataset)-len('64')] + ('_train' if is_train else '_val')
      if nm.startswith('brlr'):
         indexes = list(range(1300000)) if is_train else list(range(1300000,1315802))
         return gen_lsun_balanced(dataroot, ['bedroom_train', 'living_room_train'], tr, indexes)
      elif nm == 'twbg_train':
         indexes = list(range(700000)) if is_train else list(range(700000,708264))
         return gen_lsun_balanced(dataroot, ['tower_train', 'bridge_train'], tr, indexes)
      else:
         timeLog('Loading LSUN %s ...' % nm)
         return datasets.LSUN(dataroot, classes=[ nm ], transform=tr)            
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

#----------------------------------------------------------
def to_pm1(input):
   return input*2-1

#----------------------------------------------------------
def get_tr(dataset, is_train, do_augment):
   if dataset == 'ImageNet':  
      tr = T.Compose([ T.Resize(256), T.CenterCrop(224) ]) 
   elif dataset == 'MNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32
   elif dataset.endswith('64'):
      tr = T.Compose([ T.Resize(64), T.CenterCrop(64) ])
   else:
     tr = T.Compose([ ])
     if do_augment:
        tr = T.Compose([
              tr, 
              T.Pad(4, padding_mode='reflect'),
              T.RandomHorizontalFlip(),
              T.RandomCrop(32),
        ])

   return T.Compose([ tr, T.ToTensor(), to_pm1 ])    
