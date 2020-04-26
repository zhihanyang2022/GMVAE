"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

Edited by Zhihan to use as a VAE.

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *

# Inference Network
class InferenceNet(nn.Module):
  
  def __init__(self, x_dim, z_dim):
    
    super(InferenceNet, self).__init__()

    # q(z|x)
    self.inference_qzx = torch.nn.ModuleList([
        
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        
        nn.Linear(512, 512),
        nn.ReLU(),
        
        nn.Linear(512, 512),
        nn.ReLU(),
        
        Gaussian(512, z_dim)
        
    ])

  # q(z|x)
  def qzx(self, x):
    for layer in self.inference_qzx:
      x = layer(x)
    return x
  
  def forward(self, x):

    # q(z|x)
    mu, var, z = self.qzx(x)

    output = {'mean': mu, 'var': var, 'gaussian': z}
    return output


# Generative Network
class GenerativeNet(nn.Module):
  
  def __init__(self, x_dim, z_dim):
    
    super(GenerativeNet, self).__init__()

    # p(x|z)
    self.generative_pxz = torch.nn.ModuleList([
        
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        
        nn.Linear(512, 512),
        nn.ReLU(),
        
        nn.Linear(512, 512),
        nn.ReLU(),
        
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid()
        
    ])
 
  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z):
    
    # p(x|z)
    x_rec = self.pxz(z)

    output = {'x_rec': x_rec}
    return output


# VAE Network
class VAENet(nn.Module):
  
  def __init__(self, x_dim, z_dim):
    
    super().__init__()

    self.inference = InferenceNet(x_dim, z_dim)
    self.generative = GenerativeNet(x_dim, z_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x)
    z = out_inf['gaussian']
    out_gen = self.generative(z)
    
    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output
