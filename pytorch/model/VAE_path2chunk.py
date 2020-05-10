"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder for Unsupervised Clustering

=============================
NOTE

TITLE: Adapting code for training GM-VAE to training VAE
AUTHOR: Zhihan
DATE: 2020/04/24

This file was originally the master script for GM-VAE. However, after 
the GM-VAE with one component trained for 10000 epochs significantly 
out-performed the VAE trained for 1000 in terms of SMBA metrics, I 
decided to modify this file and use it to train a VAE. 

In terms of architectural differences, the previous VAE implementation 
used CNN layers while this VAE implementation uses fully-connected layers.

In terms of loss functions, VAE uses KL divergence loss between the
means and variances of latent vectors and the means and variances of 
a unit Gaussian; GM-VAE uses Gaussian loss, which encourages latent
vectors to be more centered at its generating Gaussian but also centered
at the Gaussian of its assigned cluster. This edited script trains a
VAE using Gaussian loss. 
=============================

"""
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from networks.networks_for_vae import *
from losses.LossFunctions import *
from metrics.Metrics import *
import matplotlib.pyplot as plt

class VAE:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.verbose = args.verbose

    self.batch_size = args.batch_size
    self.batch_size_val = args.batch_size_val
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.w_cat = args.w_categ
    self.w_gauss = args.w_gauss    
    self.w_rec = args.w_rec
    self.rec_type = args.rec_type 

    self.gaussian_size = args.gaussian_size
    self.input_size = args.input_size

    self.network = VAENet(self.input_size, self.gaussian_size)
    self.losses = LossFunctions()
    self.metrics = Metrics()

    if self.cuda:
      self.network = self.network.cuda() 
  
  # ===============================================================================
  # This is where all the losses are computed; helper method for train_epoch.

  def unlabeled_loss(self, chunk, out_net):
    
    # obtain network variables
    z, data_recon = out_net['gaussian'], out_net['x_rec'] 
    mu, var = out_net['mean'], out_net['var']
    
    # reconstruction loss
    loss_rec = self.losses.reconstruction_loss(chunk, data_recon, self.rec_type)

    # gaussian loss
    loss_gauss = self.losses.kl_divergence_v2(mu, var)

    # total loss
    loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss

    loss_dic = {
        'total': loss_total,
        'reconstruction': loss_rec,
        'gaussian': loss_gauss,
    }
    return loss_dic

  # ===============================================================================
    
  # ===============================================================================
  # Train the model for one epoch; helper method for train.
 
  def train_epoch(self, optimizer, data_loader):
        
    self.network.train()
    total_loss = 0.
    recon_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    num_batches = 0.

    for (path, chunk) in data_loader:
      
      if self.cuda == 1:
        path = path.cuda()
        chunk = chunk.cuda()

      optimizer.zero_grad()

      # flatten data
      path = path.view(path.size(0), -1)
      chunk = chunk.view(chunk.size(0), -1)
      
      # forward call
      out_net = self.network(path) 
      unlab_loss_dic = self.unlabeled_loss(chunk, out_net) 
      total = unlab_loss_dic['total']

      # accumulate values
      total_loss += total.item()
      recon_loss += unlab_loss_dic['reconstruction'].item()
      gauss_loss += unlab_loss_dic['gaussian'].item()

      # perform backpropagation
      total.backward()
      optimizer.step()
   
      num_batches += 1. 

    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches

    return total_loss, recon_loss, gauss_loss

  # ==============================================================================
    
  # ==============================================================================
  # Test the model. This method is essentially the same as train_epoch except that
  # self.network.eval is called at the beginning of this method.

  def test(self, data_loader):

    self.network.eval()
    total_loss = 0.
    recon_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    num_batches = 0.

    with torch.no_grad():
      for (data, ) in data_loader:
        
        if self.cuda == 1:
            path = path.cuda()
            chunk = chunk.cuda()
      
        # flatten data
        path = path.view(path.size(0), -1)
        chunk = chunk.view(chunk.size(0), -1)

        # forward call
        out_net = self.network(path) 
        unlab_loss_dic = self.unlabeled_loss(chunk, out_net)  

        # accumulate values
        total_loss += unlab_loss_dic['total'].item()
        recon_loss += unlab_loss_dic['reconstruction'].item()
        gauss_loss += unlab_loss_dic['gaussian'].item()
   
        num_batches += 1. 

    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches
  
    return total_loss, recon_loss, gauss_loss

  # ==============================================================================

  def train(self, train_loader, val_loader):
    """Train the model

    Args:
        train_loader: (DataLoader) corresponding loader containing the training data
        val_loader: (DataLoader) corresponding loader containing the validation data

    Returns:
        output: (dict) contains the history of train/val loss
    """
    optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
    
    train_total_losses = []
    train_recon_losses = []
    train_gauss_losses = []
    
    valid_total_losses = []
    valid_recon_losses = []
    valid_gauss_losses = []

    for epoch in range(1, self.num_epochs + 1):
      train_total_loss, train_recon_loss, train_gauss_loss = self.train_epoch(optimizer, train_loader)
      valid_total_loss, valid_recon_loss, valid_gauss_loss = self.test(val_loader)

      print(f"Epoch {epoch:5} / {self.num_epochs} | Total loss: {round(train_total_loss, 2):7} | Recon loss: {round(train_recon_loss, 2):7} | Gauss loss: {round(train_gauss_loss, 2):7}")

      train_total_losses.append(train_total_loss)
      train_recon_losses.append(train_recon_loss)
      train_gauss_losses.append(train_gauss_loss)
        
      valid_total_losses.append(valid_total_loss)
      valid_recon_losses.append(valid_recon_loss)
      valid_gauss_losses.append(valid_gauss_loss)
        
    return {
        'train_total_losses' : train_total_losses,
        'train_recon_losses' : train_recon_losses,
        'train_gauss_losses' : train_gauss_losses,
        'valid_total_losses' : valid_total_losses,
        'valid_recon_losses' : valid_recon_losses,
        'valid_gauss_losses' : valid_gauss_losses,
    }
  

  def latent_features(self, data_loader):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    self.network.eval()
   
    N = len(data_loader.dataset)
    
    features = np.zeros((N, self.gaussian_size))

    start_ind = 0
    
    with torch.no_grad():
      for (data, ) in data_loader:

        if self.cuda == 1:
          data = data.cuda()

        data = data.view(data.size(0), -1)  
        
        out = self.network(data)
        latent_feat = out['mean']
        
        end_ind = min(start_ind + data.size(0), N+1)
        features[start_ind:end_ind] = latent_feat.cpu().numpy()

        start_ind += data.size(0)
    
    return features


  def reconstruct_data(self, data_loader, sample_size=-1):

    self.network.eval()

    # sample random data from loader
    indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
    test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))
  
    # obtain values
    it = iter(test_random_loader)
    test_batch_data, _ = it.next()
    original = test_batch_data.data.numpy()
    if self.cuda:
        test_batch_data = test_batch_data.cuda()  

    # obtain reconstructed data  
    out = self.network(test_batch_data) 
    reconstructed = out['x_rec']
    return original, reconstructed.data.cpu().numpy()


  def plot_latent_space(self, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    # obtain the latent features
    features = self.latent_features(data_loader)
    
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.colorbar()
    if(save):
        fig.savefig('latent_space.png')
    return fig
  
  
  def random_generation(self, num_chunks):
    
    latent_vecs = torch.randn(num_chunks, 64).cuda()
    chunks = self.network.generative.pxz(latent_vecs)

    return chunks.cpu().detach().numpy()

