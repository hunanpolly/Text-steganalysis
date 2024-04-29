import torch
import numpy as np
import matplotlib.pyplot as plt
import torch

import sinkhorn as spc


# Inspired from Numerical tours : Point cloud OT
from numpy import random
def sinkhornL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target

    epsilon = 0.01
    niter = 100
    loss = spc.sinkhorn_loss(xm,xmt,epsilon,nt,niter)


    return loss
