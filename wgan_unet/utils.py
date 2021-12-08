import random
import torch
import decimal
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

import pdb

def normalize(data,mincount,maxcount):
    data = (data - mincount)/(maxcount-mincount)
    return data

def normalize(data,mincount,maxcount):
    data = (data - mincount)/(maxcount-mincount)
    return data


def randomnoise(images,data_min,data_max,start_pix,end_pix,rand_thres,intensity):
    if data_max==torch.tensor(0.):
        return images
    for image in range(images.shape[0]):
        for channel in range(images.shape[1]):
            for row in range(start_pix,end_pix):
                for col in range(start_pix,end_pix):
                    if random.random()>rand_thres:
                        images[image,channel,row,col] = intensity*torch.tensor(random.uniform(data_min, data_max))
    return images

def randomnoise_1pix(images,data_min,data_max):
    if data_max==torch.tensor(0.):
        return images
    for image in range(images.shape[0]):
        for channel in range(images.shape[1]):
            images[image,channel,random.randrange(30,79),random.randrange(30,79)] = torch.tensor(random.uniform(data_min.clone().detach(), data_max.clone().detach()))
    return images


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
        
def gradient_penalty(critic,real,fake,device="cpu"):
    BATCH_SIZE,C,H,W = real.shape
    e = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real * e + fake * (1-e)
    
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
    inputs = interpolated_images,
    outputs = mixed_scores,
    grad_outputs = torch.ones_like(mixed_scores),
    create_graph = True,
    retain_graph = True)[0]
    
    gradient = gradient.view((gradient.shape[0]),-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm -1)**2)
    
    return gradient_penalty
    
def gradient_penalty_(critic,real,fake,device="cpu"):
    BATCH_SIZE,C,H,W = real.shape
    e = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real * e + fake * (1-e)
    
    mixed_scores, _ = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
    inputs = interpolated_images,
    outputs = mixed_scores,
    grad_outputs = torch.ones_like(mixed_scores),
    create_graph = True,
    retain_graph = True)[0]
    
    gradient = gradient.view((gradient.shape[0]),-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm -1)**2)
    
    return gradient_penalty

def gradient_penalty_conditional(critic,real,fake,labels,device="cpu"):
    BATCH_SIZE,C,H,W = real.shape
    e = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real * e + fake * (1-e)
    
    mixed_scores = critic(interpolated_images,labels)
    
    gradient = torch.autograd.grad(
    inputs = interpolated_images,
    outputs = mixed_scores,
    grad_outputs = torch.ones_like(mixed_scores),
    create_graph = True,
    retain_graph = True)[0]
    
    gradient = gradient.view((gradient.shape[0]),-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm -1)**2)
    
    return gradient_penalty


def plot_grad_flow(named_parameters,outpath):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(outpath)
    
    
def gradient_balancing(fake,ggan,gmse,device="cpu"):
    
    gradient_gan = torch.autograd.grad(
    inputs = fake,
    outputs = ggan,
    grad_outputs = torch.ones_like(ggan))[0]
    
    gradient_gan = gradient_gan.view((gradient_gan.shape[0]),-1)
    gradient_gan_std = torch.std(gradient_gan)
    
    gradient_mse = torch.autograd.grad(
    inputs = fake,
    outputs = gmse,
    grad_outputs = torch.ones_like(gmse))[0]
    
    gradient_mse = gradient_mse.view((gradient_mse.shape[0]),-1)
    gradient_mse_std = torch.std(gradient_mse)
    
    return gradient_gan_std, gradient_mse_std

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid