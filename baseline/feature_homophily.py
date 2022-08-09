from xmlrpc.server import DocCGIXMLRPCRequestHandler
from torch_geometric.nn import GCNConv, ARGVA, ARGA
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from re import A
from dataset import WikipediaNetwork
import argparse
import torch
from torch_geometric.nn import VGAE, GCNConv,GAE
from torch_geometric.datasets import Planetoid, WebKB,Amazon
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from sklearn.decomposition import PCA
from torch_geometric.utils import to_dense_adj,structured_negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from other_hetero_datasets import load_nc_dataset
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from sklearn.model_selection import train_test_split
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--beta', type=float, default=0.85)
parser.add_argument('--nfactor', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=128,
                    help='Number of feature units.')
parser.add_argument('--nhidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--nembed', type=int, default=32,
                    help='Number of embedding units.')
parser.add_argument('--epochs', type=int,  default=2000, help='Number of epochs to train.')
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument('--dataset', type=str, default='twitch-e', help='Random seed.')
parser.add_argument('--sub_dataset', type=str, default='ENGB', help='Random seed.')
parser.add_argument('--loss_weight', type=int, default=20)
parser.add_argument('--run', type = int, default = 1)
parser.add_argument('--gpu', type = int, default = 5)
parser.add_argument('--m', type = int, default = 5)
parser.add_argument("--miniid", type=int, default=0)
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)
transform = T.Compose([T.NormalizeFeatures()])
if args.dataset in ['deezer-europe','ogbn-proteins','arxiv-year','yelp-chi']:
    data=load_nc_dataset(args.dataset).graph
# "DE", "ENGB", "ES", "FR", "PTBR", "RU", "TW"
# 'Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'
if args.dataset in ['twitch-e','fb100']:
    y=load_nc_dataset(args.dataset,args.sub_dataset).label
    data=load_nc_dataset(args.dataset,args.sub_dataset).graph
if args.dataset in ['photo']:
    dataset=Amazon(root='data/',name=args.dataset)
    data = dataset[0].to(device)
if args.dataset in ["texas", "wisconsin","cornell"]:
    dataset = WebKB(root='data/',name=args.dataset)
    data = dataset[0].to(device)
if args.dataset in ['photo']:
    dataset=Amazon(root='data/',name=args.dataset)
    data = dataset[0].to(device)
if args.dataset in ["crocodile", "squirrel",'chameleon']:
    if args.dataset=="crocodile":
        dataset = WikipediaNetwork('data/',name=args.dataset,geom_gcn_preprocess=False)
    else:
        dataset = WikipediaNetwork('data/',name=args.dataset)
        dataset1 = WikipediaNetwork('data_pre_false/', name=args.dataset, geom_gcn_preprocess=False)
        data1 = dataset1[0].to(device)
    data = dataset[0].to(device)
#nfeat=data.x.shape[1]
if args.dataset in ['photo']:
    nfeat=data.x.shape[1]
    x = data.x
    y=data.y
    #x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ["texas", "wisconsin","cornell"]:
    nfeat=data.x.shape[1]
    x = data.x
    #x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
    y=data.y
if args.dataset in ["squirrel",'chameleon']:
    nfeat=data.x.shape[1]
    x = data.x
    edge_index=data.edge_index
    y=data.y
if args.dataset in ['crocodile']:
    nfeat=data.x.shape[1]
    x=data.x
    edge_index = data.edge_index
    y=data.y
if args.dataset in ['twitch-e','fb100','deezer-europe','ogbn-proteins','arxiv-year']:
    nfeat = data['node_feat'].shape[1]
    x=data['node_feat'].to(device)
    #x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data['edge_index'].to(device)
    if args.dataset in ['twitch-e']:
        e1=torch.stack((edge_index[1],edge_index[0])).to(device)
        edge_index=torch.cat((edge_index,e1),dim=1).to(device)
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root='./data/',name=args.dataset,transform=None)
    data = dataset[0].to(device)
    x=data.x
    nfeat=x.shape[1]
    edge_index = data.edge_index
    y=data.y
if args.dataset in ['year']:
    data=torch.load('mini/year{}.pt'.format(args.miniid)).to(device)
    x=data.x
    #x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    nfeat=x.shape[1]
    edge_index = data.edge_index
    y=data.y
cos = torch.nn.CosineSimilarity(eps=1e-6)
sim=cos(x[edge_index[0]],x[edge_index[1]])
std=torch.std(sim/torch.mean(sim))
ne_sim=[]
for i in range(x.shape[0]):
    ne_num=torch.sum(edge_index[0]==i)
    if(ne_num!=0):
        ne_sim.append(torch.mean(sim[edge_index[1][edge_index[0]==i]]))
ne_sim=torch.tensor(ne_sim)
fea_homo2=torch.mean(ne_sim)

fea_homo=torch.mean(sim)
fea_homo1=torch.mean(sim[y[edge_index[0]]!=y[edge_index[1]]])/torch.mean(sim[y[edge_index[0]]==y[edge_index[1]]])
filename = f'performance/feature_similarity.csv'
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"dataset " + f"{args.dataset}," +
                        f"sub_dataset " + f"{args.sub_dataset}," +
                        f"fea_homo " + f"{fea_homo}\n")
