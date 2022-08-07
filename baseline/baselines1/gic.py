"Implementation based on https://github.com/PetarV-/DGI"
import torch
import torch.nn as nn
from baselines.GIC_layers import GCN, AvgReadout, Discriminator, Discriminator_cluster, Clusterator
import torch.nn.functional as F
import numpy as np



class GIC(nn.Module):
    def __init__(self,n_nb, n_in, n_h, activation, num_clusters, beta,device):
        super(GIC, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.device=device
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.disc_c = Discriminator_cluster(n_h,n_h,n_nb,num_clusters)
        
        
        self.beta = beta
        
        self.cluster = Clusterator(n_h,num_clusters)
        
        

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, cluster_temp, device):
        h_1 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)

        self.beta = cluster_temp
        
        Z, S = self.cluster(h_1[-1,:,:], cluster_temp)
        Z_t = S @ Z
        c2 = Z_t.to(device)
        
        c2 = self.sigm(c2)
        
        c = self.read(h_1, msk)
        c = self.sigm(c) 
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1)
        
        ret = self.disc(c_x, h_1, h_2, samp_bias1, samp_bias2)
        
        
        ret2 = self.disc_c(c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , self.device,samp_bias1,samp_bias2)
        
        #################################### OLD ##############################################
        # Z=[f(x) for f in self.factors]
        # h_all_factor=self.disentangle_layer1(Z,adj)
        # #h_all_factor=[self.mlp1(F.relu(z)) for z in h_all_factor]
        # h_all_factor=self.disentangle_layer1(h_all_factor,adj)

        # #link prediction
        # link_pred=[]
        # for i in range(self.nfactor):
        #     link_pred.append(torch.sigmoid(h_all_factor[i]@h_all_factor[i].t()))
        # link_pred=torch.stack(link_pred,dim=0)
        # link_pred=torch.max(link_pred,dim=0)
        # #feature reconstruction
        # X=[self.x_decoders1[i](h_all_factor[i]) for i in range(self.nfactor)]
        # #X = [self.x_decoders2[i](X[i]) for i in range(self.nfactor)]
        # X=torch.stack(X,dim=0)
        # x_pred=torch.sum(X,dim=0)
        # return torch.cat(h_all_factor,dim=1),link_pred[0],x_pred
        #################################### NEW ##############################################
        link_pred=[]
        emb = h_1[0]
        link_pred = torch.sigmoid(emb@emb.t())
        # link_pred=torch.max(link_pred,dim=0)
        return ret, ret2, link_pred, emb
        #################################### NEW ##############################################


        

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk, cluster_temp):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        
        
        Z, S = self.cluster(h_1[-1,:,:], self.beta)
        H = S@Z
        
        
        return h_1.detach(), H.detach(), c.detach(), Z.detach()

