import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.init import normal_
import numpy as np

from data import get_ds_attr, get_ds
import netdef
from cfggan import cfggan
from utils.utils0 import raise_if_absent, add_if_absent_, logging, timeLog, raise_if_nonpositive_any

DCGANx = 'dcganx'
Resnet4 = 'resnet4'
FCn = 'fcn'

cudnn.benchmark = True

#----------------------------------------------------------
def proc(opt): 
   check_opt_(opt)

   torch.manual_seed(opt.seed)
   np.random.seed(opt.seed)

   ds_attr = get_ds_attr(opt.dataset)
   opt.image_size = ds_attr['image_size']
   opt.channels = ds_attr['channels']

   def d_config(requires_grad):  # D
      if opt.d_model == DCGANx:
         return netdef.dcganx_D(opt.d_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.d_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.d_model == Resnet4:
         if opt.d_depth != 4:
            logging('WARNING: d_depth is ignored as d_model is Resnet4.')
         return netdef.resnet4_D(opt.d_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, 
                                 do_bias=not opt.do_no_bias)      
      else:
         raise ValueError('d_model must be dcganx.')
   def g_config(requires_grad):  # G
      if opt.g_model == DCGANx:
         return netdef.dcganx_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.g_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.g_model == Resnet4:
         if opt.g_depth != 4:
            logging('WARNING: d_depth is ignored as d_model is Resnet4.')      
         return netdef.resnet4_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, do_bias=not opt.do_no_bias)                                 
      elif opt.g_model == FCn:
         return netdef.fcn_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, requires_grad, depth=opt.g_depth)
      else:
         raise ValueError('g_model must be dcganx or fcn.')
   def z_gen(num):
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std)

   ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
   timeLog('#train = %d' % len(ds))
   loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available())
      
   cfggan(opt, d_config, g_config, z_gen, loader)

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, ['d_model','d_dim','d_depth','z_dim','z_std','g_model','g_dim','g_depth','dataset','dataroot','num_workers','batch_size','norm_type'], 'cfggan_train')
   add_if_absent_(opt, ['do_no_bias','do_augment'], False)
   add_if_absent_(opt, ['do_download'], True)
   add_if_absent_(opt, ['seed'], 1)

   raise_if_nonpositive_any(opt, ['d_depth','g_depth','d_dim','z_dim','z_std','g_dim','batch_size'])  
