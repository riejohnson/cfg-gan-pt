import sys
import os
import argparse
import torch
import time
import datetime
import pytz

#----------------------------------------------------------
def add_if_absent_(opt,names,val):
   for name in names:
      if not hasattr(opt,name):
         setattr(opt,name,val)

#----------------------------------------------------------
def raise_if_absent(opt,names,who):
   for name in names:
      if not hasattr(opt,name):
         raise Exception("%s requires %s, but it's missing." % (who,name))

#----------------------------------------------------------
def set_if_none(opt, name, val):
   if getattr(opt, name) is None:
      setattr(opt, name, val)

#----------------------------------------------------------
#!! This actually writes a file of length 0. 
#----------------------------------------------------------
def is_writable(pathname):
   try:
      fp = open(pathname, 'wb')
   except IOError:
      return False
   else:
      fp.close()
   return True

#----------------------------------------------------------
def divup(a,b):
   if a % b == 0:
      return a//b
   else:
      return a//b+1

#----------------------------------------------------------
def stem_name(fname, suffixes):
   if not isinstance(suffixes, list):
      suffixes = [ suffixes ]
   for suffix in suffixes:
      if fname.endswith(suffix) and len(fname) > len(suffix):
         return fname[0:len(fname)-len(suffix)]
   return fname

#----------------------------------------------------------
def raise_if_nonpositive_any(opt, arg_names):
   od = vars(opt)
   for name in arg_names:
      arg = od[name]
      if arg <= 0:
         raise ValueError('%s must be positive: %s.' % (name, str(arg)))

#----------------------------------------------------------
def raise_if_None_any(opt, arg_names):
   od = vars(opt)
   for name in arg_names:
      arg = od[name]
      if arg is None:
         raise ValueError('%s must not be None.' % name)

#----------------------------------------------------------
def show_args(opt, arg_names, header='', do_show_all=False):
   od = vars(opt)
   if header:
      logging(header)
   def show(name,arg):
      logging('  %s= %s' % (name,arg))
   def show_bool(name,arg):
      if arg:
         logging('  %s is turned on.' % name)
   def is_not_specified(arg):
      return arg is None or ((isinstance(arg,int) or isinstance(arg,float)) and arg < 0) or (isinstance(arg,str) and len(arg) == 0)

   if do_show_all:
      for name in arg_names:
         show(name,od[name])
   else:
      for name in arg_names:
         arg = od[name]
         if is_not_specified(arg):
            continue
         if isinstance(arg,bool):
            show_bool(name,arg)
         else:
            show(name,arg)

#----------------------------------------------------------
def raise_if_negative(value, kw):
   if value is None or value < 0:
      raise ValueError(kw + ' must be nonnegative.')

#----------------------------------------------------------
def raise_if_nonpositive(value, kw):
   if value is None or value <= 0:
      raise ValueError(kw + ' must be positive: ' + str(value))

#----------------------------------------------------------
def raise_if_None(value, kw):
   if value is None:
      raise ValueError(kw + ' must not be None.')

#----------------------------------------------------------
def raise_if_nan(value):
   if value != value:
      raise Exception("nan was detected.")

#----------------------------------------------------------
class Clock(object):
   def __init__(self):
      self.data = {}
      self.data['clk'] = time.clock()
      self.data['tim'] = time.time()
      self.data['accum_clk'] = 0
      self.data['accum_tim'] = 0

   def tick(self):
      clk = time.clock()
      tim = time.time()
      self.data['accum_clk'] += clk - self.data['clk']
      self.data['accum_tim'] += tim - self.data['tim']
      self.data['clk'] = clk
      self.data['tim'] = tim
      return ( 'clk,' + '%.5f' % (self.data['accum_clk']) +
              ',tim,' + '%.5f' % (self.data['accum_tim']) )

   def suspend(self):
      return self.tick()

   def resume(self):
      self.data['clk'] = time.clock()
      self.data['tim'] = time.time()

#----------------------------------------------------------
def logging(str, filename=None):
   if filename:
      with open(filename, 'a') as flog:
         flog.write(str + '\n')
   print(str)
   sys.stdout.flush()

#----------------------------------------------------------
def reset_logging(filename):
   if filename:
      logfile = open(filename, 'w+')
      logfile.close()

#----------------------------------------------------------
def timeLog(msg, filename=None):
   s = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z') + ': ' + msg
   logging(s, filename)

#----------------------------------------------------------
class Local_state: # this is for convenience in saving/restoring a snapshot
   def __init__(self, epo=0, upd=0, lr_coeff=1, inplist=None):
      self.reset(epo, upd, lr_coeff)
      if inplist is not None:
         self.from_list(inplist)
   def to_list(self):
      return [ self._epo, self._upd, self._lr_coeff ]
   def from_list(self, list):
      self.reset(list[0], list[1], list[2])
   def reset(self, epo=0, upd=0, lr_coeff=1):
      self._epo = epo
      self._upd = upd
      self._lr_coeff = lr_coeff
   def get(self):
      return self._epo, self._upd, self._lr_coeff
   def __str__(self):
      return ( 'epo:'      + str(self._epo)
            + ',upd:'      + str(self._upd)
            + ',lr_coeff:' + str(self._lr_coeff) )

#----------------------------------------------------------
class Global_state:
   def __init__(self, inplist=None):
      self._g_epo = self._g_upd = self._g_lc = 0 # 'g' for global
      if inplist is not None:
         self.from_list(inplist)
   def to_list(self):
      return [ self._g_epo, self._g_upd, self._g_lc ]
   def from_list(self, inp):
      self._g_epo = inp[0]; self._g_upd = inp[1]; self._g_lc = inp[2]
   def update(self, inc_epo, inc_upd): # call this at the end of base_update or gulf_update
      self._g_epo += inc_epo
      self._g_upd += inc_upd
      self._g_lc += 1
   def epo(self, local_epo):
      return self._g_epo + local_epo
   def upd(self, local_upd):
      return self._g_upd + local_upd
   def lc(self):
      return self._g_lc
   def __str__(self):
      return ( 'g_epo:%d,g_upd:%d,g_lc:%d' % (self._g_epo, self._g_upd, self._g_lc))

#----------------------------------------------------------
class ArgParser_HelpWithDefaults(argparse.ArgumentParser):
   def add_argument(self, *args, help=None, default=None, **kwargs):
      if help is not None:
         kwargs['help'] = help
      else:
         kwargs['help'] = ''
      if default is not None and args[0] != '-h':
         kwargs['default'] = default
#         if help is not None:
         kwargs['help'] += ' Default: {}'.format(default)
      super().add_argument(*args, **kwargs)

#----------------------------------------------------------
def are_these_same(o0, o1, names):
   for name in names:
      if getattr(o0,name) != getattr(o1,name):
         return False
   return True

#----------------------------------------------------------
def copy_params(src, dst):
   for key, value in dst.items():
      value.data.copy_(src[key])  
#----------------------------------------------------------
def clone_params(src, do_copy_requires_grad=False):
   p = { key: torch.zeros_like(value).data.copy_(value) for key, value in src.items() }
   if do_copy_requires_grad:
      for k,v in p.items():
         v.requires_grad = src[k].requires_grad
   return p
   
#----------------------------------------------------------
def print_params(params):
   if len(params) <= 0:
      return
   kmax = max(len(key) for key in params.keys())
   for (key, v) in sorted(params.items()):
      print(key.ljust(kmax+3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad, v.is_leaf)  
      
def print_num_params(params):      
   n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
   logging('#parameters:' + str(n_parameters))       