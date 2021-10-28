"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fast implementation of kernel in GMMIL
Imitation Learning via Kernel Mean Embedding, AAAI 2018. 
http://ailab.kaist.ac.kr/papers/pdfs/KP2018.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

"""
Here, data is the concatenated raw encodings of state and action vectors i.e.,
data = torch.cat([states, actions], 1)
"""

import torch
import math

def estimate_sigmas(data_1, data_2):
    """
    data_1 : data sampled from initial agent distribution/policy
    data_2 : data sampled from expert distribution/policy
    """
    # estimate sigma_1
    """
    median of the pairwise squared L2 distances of the data points
    from expert and agent policies
    """
    n = data_1.size(0)
    m = data_2.size(0)
    d = data_1.size(1)

    x = data_1.cpu().unsqueeze(1).expand(n, m, d)
    y = data_2.cpu().unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    sigma_1 = dist.median()
    
    # estimate sigma_2
    """
    median of the pairwise squared L2 distances of the data points
    from expert policy only 
    """
    n = data_2[:5000].size(0) # sizing down to avoid memory issues
    m = data_2[:5000].size(0)
    d = data_2[:5000].size(1)

    x = data_2[:5000].cpu().unsqueeze(1).expand(n, m, d)
    y = data_2[:5000].cpu().unsqueeze(0).expand(n, m, d)
    dist_2 = torch.pow(x - y, 2).sum(2)
    sigma_2 = dist_2.median()

def GMMIL_kernel(data_11, data_22, sigma):
    """
    data_11 : data sampled from agent distribution/policy
    data_22 : data sampled from expert distribution/policy
    sigma : bandwidth parameter for the Gaussian kernel
    """

    r = data_11.unsqueeze(0).permute(dims=[1,0,2])
    return torch.exp( -(1/sigma) * ((r - data_22)**2).sum(dim=-1))
    