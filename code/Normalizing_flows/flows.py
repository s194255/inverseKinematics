# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:12:38 2021

@author: malth
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Normalizing_flows.nets import LeafParam, MLP, ARMLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1, requires_grad=True, device=device))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.u = nn.Parameter(torch.randn(dim, 1, requires_grad=True, device=device))
        
    def forward(self, x):
        #Tried doing elementwise u * ... as GitHub line 57 https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
        z = x + self.u*(torch.matmul(self.w.T, x)+self.b)
        psi = self.w
        log_det = torch.log(torch.abs(1 + torch.matmul(self.u.T, psi)))
        return z, log_det

    def backward(self, z):
        pass


class planar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1, requires_grad=True, device=device))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.u = nn.Parameter(torch.randn(dim, 1, requires_grad=True, device=device))
            
    def forward(self, x): #x should be an OBSERVATION x ATTRIBUTES matrix
        # Activation function: h = tanh
        x = x.T
        h = torch.tanh((torch.matmul(self.w.T, x)+self.b))
        z = x + self.u*h
        psi = (1 - h**2) * self.w
        log_det = torch.log(torch.abs(1 + torch.matmul(self.u.T, psi))+10**(-4))
        return z.T, log_det

    def backward(self, z):
        pass


class normalizeModel(nn.Module):
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x): #x should be an OBSERVATION x ATTRIBUTES matrix
        log_det_total = 0
        log_det = torch.zeros(len(x), device=device)
        for flow in self.flows:
            x, log_det_flow = flow.forward(x)
            log_det += log_det_flow.squeeze()
        prior_log = self.prior.log_prob(x)
        # log_density = prior_log + log_det
        #now we are done so
        z_K = x
        return z_K, prior_log, log_det






class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity

    def forward(self, x):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        # we have to decode the x one at a time, sequentially
        z = z.float()
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0), device=device)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det
    
    
# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        log_det = log_det.to(device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m, device=device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det  
    
    
class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs

    def sampleDirection(self, vectors):
        xs, _ = self.flow.backward(vectors)
        return xs