from ossaudiodev import SNDCTL_DSP_BIND_CHANNEL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
class Factor(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Factor, self).__init__()
        self.nfeat=nfeat
        self.nhid=nhid
        self.mlp = nn.Linear(nfeat, nhid)
    def forward(self,x):
        x=self.mlp(x)
        return x
class Factor2(nn.Module):
    def __init__(self, nfeat,nmid, nhid):
        super(Factor2, self).__init__()
        self.nfeat=nfeat
        self.nhid=nhid
        self.nmid = nmid
        self.mlp1 = nn.Linear(nfeat, nmid)
        self.mlp2 = nn.Linear(nmid, nhid)
    def forward(self,x):
        x=F.relu(self.mlp1(x))
        x=self.mlp2(x)
        return x
class Dec2(nn.Module):
    def __init__(self, nembed,nhid, nfeat):
        super(Dec2, self).__init__()
        self.nfeat=nfeat
        self.nhid=nhid
        self.nembed = nembed
        self.mlp1 = nn.Linear(nembed, nhid)
        self.mlp2 = nn.Linear(nhid, nfeat)
    def forward(self,x):
        x=F.relu(self.mlp1(x))
        x=self.mlp2(x)
        return x
class Dec(nn.Module):
    def __init__(self,nhid,nfeat):
        super(Dec, self).__init__()
        self.nfeat=nfeat
        self.nhid=nhid
        self.mlp = nn.Linear(nhid,nfeat)
    def forward(self,x):
        x=self.mlp(x)
        return x
class Disentangle_layer(nn.Module):
    def __init__(self, nfactor,beta,t=1):
        super(Disentangle_layer, self).__init__()
        self.temperature = t
        self.nfactor = nfactor
        self.beta =beta
    def forward(self, Z, adj):
        temp = [torch.exp(torch.mm(z, z.t())/self.temperature) for z in Z]
        alpha0 = torch.stack(temp, dim=0)
        alpha_fea=torch.diagonal(alpha0, 0)
        t=torch.sum(alpha0, dim=0)
        alpha = alpha0/t
        p = torch.argmax(alpha, dim=0) + 1
        p_adj = p*adj
        edge_factor_list = []
        for i in range(self.nfactor):
            mask = (p_adj == i + 1).float()
            edge_factor_list.append(mask)
        h_all_factor = []
        att=[]
        for i in range(self.nfactor):
            alpha1=edge_factor_list[i] * alpha[i]
            sum=torch.sum(alpha1,dim=1)
            sum[sum==0]=1
            alpha1 = alpha1 / sum
            att.append(alpha1)
            temp = self.beta * Z[i] + (1 - self.beta) * torch.mm(alpha1, Z[i])
            h_all_factor.append(temp)
        return h_all_factor,alpha0,att

class Disentangle_out_layer(nn.Module):
    def __init__(self,beta,t=1):
        super(Disentangle_out_layer, self).__init__()
        self.temperature = t
        self.beta = beta
    def forward(self, Z, adj):
        temp = torch.exp(torch.mm(Z, Z.t())/self.temperature)
        alpha = temp / torch.sum(temp,dim=1)
        p_adj = alpha*adj
        h_all_factor = self.beta * Z + (1 - self.beta) * torch.mm(p_adj, Z)
        return h_all_factor,alpha

class Disentangle(nn.Module):
    def __init__(self, nfeat, nhid,nebed,nfactor,beta,t=1):
        super(Disentangle, self).__init__()
        if(nhid==1):
            self.factors= [Factor(nfeat,nebed) for _ in range(nfactor)]
        else:
            self.factors= [Factor2(nfeat,nhid,nebed) for _ in range(nfactor)]
        for i, factor in enumerate(self.factors):
            self.add_module('factor_{}'.format(i), factor)
        self.disentangle_layer1=Disentangle_layer(nfactor,beta,t)
        self.disentangle_layer2 = Disentangle_layer(nfactor, beta,t)
        self.temperature=t
        self.nfactor=nfactor
        self.beta=beta
    def forward(self,x,adj):
        Z=[f(x) for f in self.factors]
        h_all_factor,alpha,att=self.disentangle_layer1(Z,adj)
        #link prediction
        link_pred=[]
        for i in range(self.nfactor):
            link_pred.append(h_all_factor[i]@h_all_factor[i].t())
        link_pred=torch.stack(link_pred,dim=0)
        link_pred=torch.sigmoid(torch.sum(link_pred*alpha,dim=0))
        return torch.cat(h_all_factor,dim=1),link_pred