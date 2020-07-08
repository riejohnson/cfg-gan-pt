import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
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
   elif dataset == 'brlr64' or dataset == 'twbg64':
      nclass = 2; imgsz = 64
   elif dataset.endswith('64'):
      nclass = 1; imgsz = 64
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

   return { "nclass": nclass, "image_size": imgsz, "channels": channels }

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
      nm = dataset[len('lsun_'):len(dataset)-len('64')] + '_train'
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
