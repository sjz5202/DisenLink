from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import argparse
import numpy as np
import math
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix,dense_to_sparse,is_undirected
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import sys
import os.path
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append('%s/software/' % par_dir)
from drnl import drnl_node_labeling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_edges(data,args):
    set_random_seed(args.seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v= floor(args.val_ratio * row.size(0)).int() #number of validation positive edges
    n_t=floor(args.test_ratio * row.size(0)).int() #number of test positive edges
    #split positive edges   
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    #sample negative edges
    if args.practical_neg_sample == False:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:row.size(0)]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
        data.train_neg = torch.stack([row, col], dim=0)

    else:

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        data.test_neg = torch.stack([neg_row, neg_col], dim=0)

        # Sample the train and val negative edges with only knowing 
        # the train positive edges
        row, col = data.train_pos
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

        n_tot = n_v + data.train_pos.size(1)
        perm = torch.randperm(neg_row.size(0))[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:], neg_col[n_v:]
        data.train_neg = torch.stack([row, col], dim=0)

    return data




def plus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_m, mapping, _ = k_hop_subgraph(node_idx= p_edge, num_hops=args.num_hops,\
 edge_index = data_observed.edge_index, max_nodes_per_hop=args.max_nodes_per_hop ,num_nodes=data_observed.num_nodes)
    x_sub = data_observed.x[nodes,:]
    edge_index_p = edge_index_m
    edge_index_p = torch.cat((edge_index_p, mapping.view(-1,1)),dim=1)
    edge_index_p = torch.cat((edge_index_p, mapping[[1,0]].view(-1,1)),dim=1)

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1),dtype=torch.bool)
    edge_mask[-1] = False
    edge_mask[-2] = False

    if args.drnl == True:
        num_nodes = torch.max(edge_index_p)+1
        z = drnl_node_labeling(edge_index_m, mapping[0],mapping[1],num_nodes)
        data = Data(edge_index = edge_index_p, x = x_sub, z = z)
    else:
        data = Data(edge_index = edge_index_p, x = x_sub, z = 0)
    data.edge_mask = edge_mask

    #label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    data.label = float(label)

    return data

def minus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_p, mapping,_ = k_hop_subgraph(node_idx= p_edge, num_hops=args.num_hops,\
 edge_index = data_observed.edge_index,max_nodes_per_hop=args.max_nodes_per_hop, num_nodes = data_observed.num_nodes)
    x_sub = data_observed.x[nodes,:]

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1), dtype = torch.bool)
    ind = torch.where((edge_index_p == mapping.view(-1,1)).all(dim=0))
    edge_mask[ind[0]] = False
    ind = torch.where((edge_index_p == mapping[[1,0]].view(-1,1)).all(dim=0))
    edge_mask[ind[0]] = False
    if args.drnl == True:
        num_nodes = torch.max(edge_index_p)+1
        z = drnl_node_labeling(edge_index_p[:,edge_mask], mapping[0],mapping[1],num_nodes)
        data = Data(edge_index = edge_index_p, x= x_sub,z = z)
    else:
        data = Data(edge_index = edge_index_p, x= x_sub,z = 0)
    data.edge_mask = edge_mask

    #label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    data.label = float(label)
    return data


def load_splitted_data(args):
    par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    data_name=args.data_name+'_split_'+args.data_split_num
    if args.test_ratio==0.5:
        data_dir = os.path.join(par_dir, 'data/splitted_0_5/{}.mat'.format(data_name))
    else:
        data_dir = os.path.join(par_dir, 'data/splitted/{}.mat'.format(data_name))
    import scipy.io as sio
    print('Load data from: '+data_dir)
    net = sio.loadmat(data_dir)
    data = Data()

    data.train_pos = torch.from_numpy(np.int64(net['train_pos']))
    data.train_neg = torch.from_numpy(np.int64(net['train_neg']))
    data.test_pos = torch.from_numpy(np.int64(net['test_pos']))
    data.test_neg = torch.from_numpy(np.int64(net['test_neg']))

    n_pos= floor(args.val_ratio * len(data.train_pos)).int()
    nlist=np.arange(len(data.train_pos))
    np.random.shuffle(nlist)
    val_pos_list=nlist[:n_pos]
    train_pos_list=nlist[n_pos:]
    data.val_pos=data.train_pos[val_pos_list]
    data.train_pos=data.train_pos[train_pos_list]

    n_neg = floor(args.val_ratio * len(data.train_neg)).int()
    nlist=np.arange(len(data.train_neg))
    np.random.shuffle(nlist)
    val_neg_list=nlist[:n_neg]
    train_neg_list=nlist[n_neg:]
    data.val_neg=data.train_neg[val_neg_list]
    data.train_neg=data.train_neg[train_neg_list]

    data.val_pos = torch.transpose(data.val_pos,0,1)
    data.val_neg = torch.transpose(data.val_neg,0,1)
    data.train_pos = torch.transpose(data.train_pos,0,1)
    data.train_neg = torch.transpose(data.train_neg,0,1)
    data.test_pos = torch.transpose(data.test_pos,0,1)
    data.test_neg = torch.transpose(data.test_neg,0,1)
    num_nodes = max(torch.max(data.train_pos), torch.max(data.test_pos))+1
    num_nodes=max(num_nodes,torch.max(data.val_pos)+1)
    data.num_nodes = num_nodes

    return data

def load_unsplitted_data(args):
    # read .mat format files
    data_dir = os.path.join(par_dir, 'data_no/{}.mat'.format(args.data_name))
    print('Load data from: '+ data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = torch.max(data.edge_index)+1
    return data
data_dir = os.path.join( 'data_no/{}.mat'.format('USAir'))
print('Load data from: '+ data_dir)
import scipy.io as sio
net = sio.loadmat(data_dir)
print(net)
edge_index,a = from_scipy_sparse_matrix(net['net'])
print(edge_index.shape)
print(torch.unique(edge_index))
print(a)
print(a.shape)
print(torch.unique(a))
