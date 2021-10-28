"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fast implementation of regularizers in WAIL
Wasserstein Adversarial Imitation Learning
https://arxiv.org/abs/1906.08113#
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def l2_reg(g_sa, e_sa, g_o, e_o, wail_epsilon):
    """
    g_sa : concatenated agent/generator data i.e. g_sa = torch.cat([g_states, g_actions], 1)
    e_sa : concatenated expert data i.e. e_sa = torch.cat([e_states, e_actions], 1)
    g_o : g_sa passed through Discriminator network i.e. g_o = D(g_sa)
    e_o : e_sa passed through Discriminator network i.e. e_o = D(e_sa)
    wail_epsilon : hyper-parameter epsilon used in WAIL algorithm  
    """
    a = e_o.unsqueeze(0).permute(dims=[1,0,2])
    b = a - g_o
    diff = b.reshape(g_sa.shape[0]*e_sa.shape[0])
    
    r = g_sa.unsqueeze(0).permute(dims=[1,0,2])
    s = ((r - e_sa)**2).sum(dim=-1)
    dxy = s.reshape(diff.shape)
    dxy = torch.sqrt(dxy)
    
    reg = diff - dxy
    reg[reg < 0] = 0
    reg1 = reg**2
    
    return -reg1.mean()/(4*wail_epsilon)

def entropy_reg(g_sa, e_sa, g_o, e_o, wail_epsilon):
    """
    g_sa : concatenated agent/generator data i.e. g_sa = torch.cat([g_states, g_actions], 1)
    e_sa : concatenated expert data i.e. e_sa = torch.cat([e_states, e_actions], 1)
    g_o : g_sa passed through Discriminator network i.e. g_o = D(g_sa)
    e_o : e_sa passed through Discriminator network i.e. e_o = D(e_sa)
    wail_epsilon : hyper-parameter epsilon used in WAIL algorithm  
    """
    a = e_o.unsqueeze(0).permute(dims=[1,0,2])
    b = a - g_o
    diff = b.reshape(g_sa.shape[0]*e_sa.shape[0])
    
    r = g_sa.unsqueeze(0).permute(dims=[1,0,2])
    s = ((r - e_sa)**2).sum(dim=-1)
    dxy = s.reshape(diff.shape)
    dxy = torch.sqrt(dxy)
    
    reg = (diff - dxy)/args.wail_epsilon
    torch.exp(reg).sum()
    
    reg[reg < 0] = 0
    reg1 = reg**2
    
    return -reg1.sum()/(4*wail_epsilon)
