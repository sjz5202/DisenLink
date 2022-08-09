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
        
class Disentangle_fea(nn.Module):
    def __init__(self, nfeat, nhid,nebed,nfactor,beta,nlabel,t=1):
        super(Disentangle_fea, self).__init__()
        if(nhid==0):
            self.factors= [Factor(nfeat,nebed) for _ in range(nfactor)]
        else:
            self.factors= [Factor2(nfeat,nhid,nebed) for _ in range(nfactor)]
        for i, factor in enumerate(self.factors):
            self.add_module('factor_{}'.format(i), factor)
        
        self.x_decoders = Dec(5*nebed, nfeat)
        self.x_decoders1 = [Dec(nebed, nfeat) for _ in range(nfactor)]
        for i, dec in enumerate(self.x_decoders1):
            self.add_module('x_dec1_{}'.format(i), dec)
        #self.x_decoders = Dec2(nebed,nhid,nfeat)
        self.disentangle_layer1=Disentangle_layer(nfactor,beta,t)
        self.temperature=t
        self.nfactor=nfactor
        self.beta=beta
        self.nlabel=nlabel
        self.nebed=nebed
        self.cluster_layer = Parameter(torch.Tensor(nlabel,5*nebed))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.factor_weight = Parameter(torch.Tensor(nfactor,1))
        torch.nn.init.xavier_normal_(self.factor_weight.data)
        self.factor_attention = Parameter(torch.Tensor(nebed,nfactor))
        torch.nn.init.xavier_normal_(self.factor_attention.data)
    def get_Q(self, z):
        v=1
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / v)
        q = q.pow((v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
    def forward(self,x,adj):
        Z=[f(x) for f in self.factors]
        h_all_factor,alpha,_=self.disentangle_layer1(Z,adj)
        #h_all_factor1=torch.concat(h_all_factor,dim=1)
        #link_pred=torch.sigmoid(h_all_factor1@h_all_factor1.t())
        #link prediction
        link_pred=[]
        for i in range(self.nfactor):
            #link_pred.append(torch.sigmoid(h_all_factor[i]@h_all_factor[i].t()))
            link_pred.append(h_all_factor[i]@h_all_factor[i].t())
        link_pred=torch.stack(link_pred,dim=0)
        #link_pred=torch.max(link_pred,dim=0)
        link_pred=torch.sigmoid(torch.sum(link_pred*alpha,dim=0))
        #link_pred=torch.sigmoid(torch.sum(link_pred,dim=0))

        # feature reconstruction
        #h_pooling=torch.stack(h_all_factor,dim=0)
        if(1):
            #k*1
            #h_pooling=torch.stack([self.factor_weight[i]*h_all_factor[i] for i in range(self.nfactor)],dim=0)
            #h_pooling=torch.sum(h_pooling,dim=0)
            h_pooling=torch.cat([self.factor_weight[i]*h_all_factor[i] for i in range(self.nfactor)],dim=1)
            #n*k
            #h_pooling=torch.stack([self.factor_weight[i].unsqueeze (1)*h_all_factor[i] for i in range(self.nfactor)],dim=0)
            #h_pooling=torch.sum(h_pooling,dim=0)
            #h_pooling=torch.cat([self.factor_weight[i].unsqueeze (1)*h_all_factor[i] for i in range(self.nfactor)],dim=1)
            #embed*k
            #attention=[torch.mm(h_all_factor[i],self.factor_attention[:,i].unsqueeze (1)) for i in range(self.nfactor)]
            #h_pooling=torch.stack([attention[i]*h_all_factor[i] for i in range(self.nfactor)],dim=0)
            #h_pooling=torch.sum(h_pooling,dim=0)
            #h_pooling=torch.cat([attention[i]*h_all_factor[i] for i in range(self.nfactor)],dim=1)
        #h_pooling=torch.cat([self.factor_weight[i]*h_all_factor[i] for i in range(self.nfactor)],dim=1)

        #xp=[self.x_decoders1[i](h_all_factor[i]*self.factor_weight[i]) for i in range(self.nfactor)]
        #xp=torch.stack(xp,dim=0)
        #x_pred=torch.sum(xp,dim=0)
        x_pred=self.x_decoders(h_pooling)
        return torch.cat(h_all_factor,dim=1),link_pred,x_pred,self.get_Q(h_pooling),h_pooling
class Disentangle_2layer(nn.Module):
    def __init__(self, nfeat, nhid,nebed,nfactor,beta,t=1):
        super(Disentangle_2layer, self).__init__()
        self.factors1= [nn.Linear(nfeat, nhid) for _ in range(nfactor)]
        self.factors2= [nn.Linear(nhid*nfactor,nebed) for _ in range(nfactor)]
        self.out_layer=nn.Linear(nhid*nfactor,nebed)
        for i, factor in enumerate(self.factors1):
            self.add_module('factor1_{}'.format(i), factor)
        for i, factor in enumerate(self.factors2):
            self.add_module('factor2_{}'.format(i), factor)
        self.x_decoders1 = [Dec(nebed, nfeat) for _ in range(nfactor)]
        for i, dec in enumerate(self.x_decoders1):
            self.add_module('x_dec1_{}'.format(i), dec)
        self.disentangle_layer1=Disentangle_layer(nfactor,beta,t)
        self.disentangle_layer2 =Disentangle_out_layer(beta,t)
        self.temperature=t
        self.nfactor=nfactor
        self.beta=beta
    def forward(self,x,adj):
        Z1=[f(x) for f in self.factors1]
        h_all_factor1,alpha1=self.disentangle_layer1(Z1,adj)
        z2=torch.cat(h_all_factor1,dim=1)
        Z2=self.out_layer(F.relu(z2))
        h_all_factor2,alpha2=self.disentangle_layer2(Z2,adj)
        link_pred=torch.sigmoid(h_all_factor2@h_all_factor2.t())
        #h_all_factor1=torch.concat(h_all_factor,dim=1)
        #link_pred=torch.sigmoid(h_all_factor1@h_all_factor1.t())

        #link prediction
        '''
        link_pred=[]
        for i in range(self.nfactor):
            #link_pred.append(torch.sigmoid(h_all_factor[i]@h_all_factor[i].t()))
            link_pred.append(h_all_factor2[i]@h_all_factor2[i].t())
        link_pred=torch.stack(link_pred,dim=0)
        #link_pred=torch.max(link_pred,dim=0)
        link_pred=torch.sigmoid(torch.sum(link_pred*alpha2,dim=0))
        print(link_pred)
        print(sddd)
        '''
        #feature reconstruction
        x=[self.x_decoders1[i](h_all_factor[i]) for i in range(self.nfactor)]
        X=torch.stack(X,dim=0)
        x_pred=torch.sum(X,dim=0)
        x_pred=1
        return h_all_factor2,link_pred,x_pred
class Disentangle2(nn.Module):
    def __init__(self, nfeat, nhid,nfactor,beta,t=1):
        super(Disentangle2, self).__init__()
        self.factors= [Factor2(nfeat,64, nhid) for _ in range(nfactor)]
        for i, factor in enumerate(self.factors):
            self.add_module('factor_{}'.format(i), factor)
        self.mlp1 = nn.Linear(4*nhid, nhid)
        self.x_decoders1 = [Dec2(nfeat, 64,nhid) for _ in range(nfactor)]
        for i, dec in enumerate(self.x_decoders1):
            self.add_module('x_dec1_{}'.format(i), dec)
        #self.x_decoders2 = [Dec(4*nhid, nfeat) for _ in range(nfactor)]
        #for i, dec in enumerate(self.x_decoders2):
        #    self.add_module('x_dec2_{}'.format(i), dec)
        self.disentangle_layer1=Disentangle_layer(nfactor,beta,t=1)
        self.disentangle_layer2 = Disentangle_layer(nfactor, beta, t=1)
        self.temperature=t
        self.nfactor=nfactor
        self.beta=beta
    def forward(self,x,adj):
        Z=[f(x) for f in self.factors]
        h_all_factor=self.disentangle_layer1(Z,adj)
        #h_all_factor=[self.mlp1(F.relu(z)) for z in h_all_factor]
        h_all_factor=self.disentangle_layer1(h_all_factor,adj)

        #link prediction
        link_pred=[]
        for i in range(self.nfactor):
            link_pred.append(torch.sigmoid(h_all_factor[i]@h_all_factor[i].t()))
        link_pred=torch.stack(link_pred,dim=0)
        link_pred=torch.max(link_pred,dim=0)
        #feature reconstruction
        X=[self.x_decoders1[i](h_all_factor[i]) for i in range(self.nfactor)]
        #X = [self.x_decoders2[i](X[i]) for i in range(self.nfactor)]
        X=torch.stack(X,dim=0)
        x_pred=torch.sum(X,dim=0)
        return torch.cat(h_all_factor,dim=1),link_pred[0],x_pred

class Disentangle_fea1(nn.Module):
    def __init__(self, nfeat, nhid,nebed,nlabel):
        super(Disentangle_fea1, self).__init__()
        self.factors1= Factor(nfeat,nfeat)
        self.factors2= Factor(nfeat,nebed)
        self.factors= Factor(nfeat,nebed)
        self.x_decoders = Dec(nebed, nfeat)
        self.x_decoders1 = Dec(nebed, nfeat)
        self.x_decoders2 = Dec(nfeat, nfeat)
    def forward(self,x,adj):
        Z=torch.relu(self.factors1(x))
        Z=torch.relu(self.factors2(Z))
        #Z=torch.relu(self.factors(x))
        Z=torch.relu(self.x_decoders1(Z))
        x_pred=torch.relu(self.x_decoders2(Z))
        #x_pred=torch.relu(self.x_decoders(Z))
        return Z,x_pred





