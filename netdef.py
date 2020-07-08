import torch
from torch.nn.init import normal_
import torch.nn.functional as F
import utils.utils as utils

#-------------------------------------------------------------
def conv2d_params(ni, no, k, do_bias, std=0.01):
   return {'w': normal_(torch.Tensor(no, ni, k, k), std=std), 
           'b': torch.zeros(no) if do_bias else None }

def conv2dT_params(ni, no, k, do_bias, std=0.01):
   return {'w': normal_(torch.Tensor(ni, no, k, k), std=std), 
           'b': torch.zeros(no) if do_bias else None }

#-------------------------------------------------------------
def dcganx_D(nn0, imgsz,
             channels,    # 1: gray-scale, 3: color
             norm_type,   # 'bn', 'none'
             requires_grad, 
             depth=3, leaky_slope=0.2, nodemul=2, do_bias=True):
              
   ker=5; padding=2
   
   def gen_block_params(ni, no, k):
      return {
         'conv0': conv2d_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, 1, do_bias), 
         'bn0': utils.bnparams(no) if norm_type == 'bn' else None, 
         'bn1': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no, count):
       return {'block%d' % i: gen_block_params(ni if i == 0 else no, no, ker) for i in range(count)}

   count = 1
   sz = imgsz // (2**depth)
   nn = nn0
   p = { 'conv0': conv2d_params(channels, nn0, ker, do_bias) }
   for d in range(depth-1):
      p['group%d'%d] = gen_group_params(nn, nn*nodemul, count)
      nn = nn*nodemul
   p['fc'] = utils.linear_params(sz*sz*nn, 1)
   flat_params = utils.cast(utils.flatten(p))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, stride):
      o = F.conv2d(x, params[base+'.conv0.w'], params.get(base+'conv0.b'), stride=stride, padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn0', mode)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'conv1.b'), stride=1, padding=0)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn1', mode)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      return o

   def group(o, params, base, mode, stride=2):
      n = 1
      for i in range(n):
         o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
      return o

   def f(input, params, mode):
      o = F.conv2d(input, params['conv0.w'], params.get('conv0.b'), stride=2, padding=padding)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      for d in range(depth-1):
         o = group(o, params, 'group%d'%d, mode)
      o = o.view(o.size(0), -1)
      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params

#-------------------------------------------------------------
def dcganx_G(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad, depth=3, 
             nodemul=2, do_bias=True):
              
   ker=5; padding=2; output_padding=1

   def gen_block_T_params(ni, no, k):
      return {
         'convT0': conv2dT_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, 1, do_bias), 
         'bn0': utils.bnparams(no) if norm_type == 'bn' else None, 
         'bn1': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_T_params(ni, no, count):
       return {'block%d' % i: gen_block_T_params(ni if i == 0 else no, no, ker) for i in range(count)}

   count = 1
   nn0 = n0g * (nodemul**(depth-1))
   sz = imgsz // (2**depth)  
   p = { 'proj': utils.linear_params(input_dim, nn0*sz*sz) }
   nn = nn0
   for d in range(depth-1):
      p['group%d'%d] = gen_group_T_params(nn, nn//nodemul, count)
      nn = nn//nodemul
   p['last_convT'] = conv2dT_params(nn, channels, ker, do_bias)
   flat_params = utils.cast(utils.flatten(p))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, stride):
      o = F.relu(x, inplace=True)
      o = F.conv_transpose2d(o, params[base+'.convT0.w'], params.get(base+'.convT0.b'),
                             stride=stride, padding=padding, output_padding=output_padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn0', mode)

      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'.conv1.b'),
                   stride=1, padding=0)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn1', mode)
      return o

   def group(o, params, base, mode, stride=2):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
      return o

   def f(input, params, mode):
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn0, sz, sz)
      for d in range(depth-1):
        o = group(o, params, 'group%d'%d, mode)
      o = F.relu(o, inplace=True)
      o = F.conv_transpose2d(o, params['last_convT.w'], params.get('last_convT.b'), stride=2,
                             padding=padding, output_padding=output_padding)
      o = torch.tanh(o)
      return o

   return f, flat_params

#-------------------------------------------------------------
def fcn_G(input_dim, nn, imgsz, channels, requires_grad, depth=2):
   def gen_block_params(ni, no):
      return {'fc': utils.linear_params(ni, no),}

   def gen_group_params(ni, no, count):
      return {'block%d' % i: gen_block_params(ni if i == 0 else no, no) for i in range(count)}

   flat_params = utils.cast(utils.flatten({
        'group0': gen_group_params(input_dim, nn, depth),
        'last_proj': utils.linear_params(nn, imgsz*imgsz*channels),
   }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode):
      return F.relu(F.linear(x, params[base+'.fc.weight'], params[base+'.fc.bias']), inplace=True)

   def group(o, params, base, mode):
      for i in range(depth):
         o = block(o, params, '%s.block%d' % (base,i), mode)
      return o

   def f(input, params, mode):
      o = group(input, params, 'group0', mode)
      o = F.linear(o, params['last_proj.weight'], params['last_proj.bias'])
      o = torch.tanh(o)
#      o = o.view(o.size(0), channels, imgsz, imgsz)
      o = o.reshape(o.size(0), channels, imgsz, imgsz)      
      return o

   return f, flat_params

#-------------------------------------------------------------
def resnet4_D(nn, imgsz,
              channels,    # 1: gray-scale, 3: color
              norm_type,  # 'bn', 'none'
              requires_grad,
              do_bias=True):             
   depth =4
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_group0_params(no):
      ni = channels
      return { 'block0' : {
         'conv0': conv2d_params(ni, no, ker, do_bias), 
         'conv1': conv2d_params(no, no, ker, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }}

   def gen_resnet_D_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, ni, k, do_bias), 
         'conv1': conv2d_params(ni, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_D_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}

   sz = imgsz // (2**depth)
   flat_params = utils.cast(utils.flatten({
        'group0': gen_group0_params(nn),
        'group1': gen_group_params(nn,   nn*2),
        'group2': gen_group_params(nn*2, nn*4),
        'group3': gen_group_params(nn*4, nn*8),        
        'fc': utils.linear_params(sz*sz*nn*8, 1),
   }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_downsample, is_first):
      o = x
      if not is_first:
         o = F.relu(o, inplace=True)   
      o = F.conv2d(x, params[base+'.conv0.w'], params.get(base+'conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)      
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
 
      if do_downsample:
         o = F.avg_pool2d(o,2)
         x = F.avg_pool2d(x,2)
      
      if base + '.convdim' in params:
         return o + F.conv2d(x, params[base + '.convdim'])
      else:
         return o + x


   def group(o, params, base, mode, do_downsample, is_first=False):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, 
                   do_downsample=(do_downsample and i == count-1), 
                   is_first=(is_first and i == 0))                   
      return o

   def f(input, params, mode):
      o = group(input, params, 'group0', mode, do_downsample=True, is_first=True)
      o = group(o, params, 'group1', mode, do_downsample=True)
      o = group(o, params, 'group2', mode, do_downsample=True)
      o = group(o, params, 'group3', mode, do_downsample=True)      
      o = F.relu(o, inplace=True)
      o = o.view(o.size(0), -1)
      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params   
   
#-------------------------------------------------------------
def resnet4_G(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad,
             do_bias=True):         
   depth = 4
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_resnet_G_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_G_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}

   nn = n0g * (2**(depth-1)); sz = imgsz // (2**depth)
   flat_params = utils.cast(utils.flatten({
        'proj': utils.linear_params(input_dim, nn*sz*sz),
        'group0': gen_group_params(nn,    nn//2),
        'group1': gen_group_params(nn//2, nn//4),
        'group2': gen_group_params(nn//4, nn//8),
        'group3': gen_group_params(nn//8, nn//8),        
        'last_conv': conv2d_params(nn//8, channels, ker, do_bias),
   }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_upsample):
      o = F.relu(x, inplace=True)
      if do_upsample:
        o = F.interpolate(o, scale_factor=2, mode='nearest')
            
      o = F.conv2d(o, params[base+'.conv0.w'], params.get(base+'.conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'.conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
         
      xo = F.conv2d(x, params[base + '.convdim']) 
      if do_upsample:
         return o + F.interpolate(xo, scale_factor=2, mode='nearest')
      else:
         return o + xo
 
   def group(o, params, base, mode, do_upsample):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, do_upsample if i == 0 else False)
      return o

   def show_shape(o, msg=''):
      print(o.size(), msg)

   def f(input, params, mode):
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn, sz, sz)
      o = group(o, params, 'group0', mode, do_upsample=True)
      o = group(o, params, 'group1', mode, do_upsample=True)
      o = group(o, params, 'group2', mode, do_upsample=True)
      o = group(o, params, 'group3', mode, do_upsample=True)
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params['last_conv.w'], params.get('last_conv.b'), padding=padding)
      o = torch.tanh(o)
      return o

   return f, flat_params   
   