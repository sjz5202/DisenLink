from torch_geometric.nn import GCNConv, ARGVA, ARGA,GATConv
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from re import A
from dataset import WikipediaNetwork
import argparse
import torch
from torch_geometric.nn import VGAE, GCNConv
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
import dgl
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from dgl import function as fn
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--beta', type=float, default=0.85)
parser.add_argument('--alpha', type=float, default=0.85)
parser.add_argument('--nfactor', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=128,
                    help='Number of feature units.')
parser.add_argument('--nhidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nembed', type=int, default=32,
                    help='Number of embedding units.')
parser.add_argument('--epochs', type=int,  default=2000, help='Number of epochs to train.')
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument('--dataset', type=str, default='chameleon', help='Random seed.')
parser.add_argument('--sub_dataset', type=str, default='Reed98', help='Random seed.')
parser.add_argument('--loss_weight', type=int, default=20)
parser.add_argument('--run', type = int, default = 1)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--m', type = int, default = 5)
parser.add_argument('--head', type = int, default = 5)
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
    data=load_nc_dataset(args.dataset,args.sub_dataset).graph

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
if args.dataset in ["squirrel",'chameleon']:
    nfeat = args.nfeat
    x = data1.x
    x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index=data1.edge_index
if args.dataset in ['photo']:
    nfeat=data.x.shape[1]
    x = data.x
    y=data.y
    #x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ['crocodile']:
    nfeat=data.x.shape[1]
    x=data.x
    edge_index = data.edge_index
if args.dataset in ['twitch-e','fb100','deezer-europe','ogbn-proteins','arxiv-year']:
    nfeat = data['node_feat'].shape[1]
    x=data['node_feat'].to(device)
    edge_index = data['edge_index'].to(device)
    if args.dataset in ['twitch-e']:
        e1=torch.stack((edge_index[1],edge_index[0])).to(device)
        edge_index=torch.cat((edge_index,e1),dim=1).to(device)
if args.dataset in ["texas", "wisconsin","cornell"]:
    nfeat=data.x.shape[1]
    x = data.x
    #x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root='./data/',name=args.dataset,transform=None)
    data = dataset[0].to(device)
    x=data.x
    nfeat=x.shape[1]
    edge_index = data.edge_index
if args.dataset in ['year']:
    data=torch.load('mini/year{}.pt'.format(args.miniid)).to(device)
    x=data.x
    #x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    nfeat=x.shape[1]
    edge_index = data.edge_index 
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def get_g(edge_index,x):
        U = edge_index[0]
        V = edge_index[1]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        feat = normalize_features(x)
        feat = torch.FloatTensor(feat)

        return g,feat
class FALayer(torch.nn.Module):
    def __init__(self, g, in_dim):
        super(FALayer, self).__init__()
        self.g = g
        self.gate = torch.nn.Linear(2 * in_dim, 1)
        torch.nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(torch.nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = 0.3
        self.layer_num = layer_num

        self.layers2 = torch.nn.ModuleList()
        self.layers3 = torch.nn.ModuleList()
        for i in range(self.layer_num):
            self.layers2.append(FALayer(self.g, hidden_dim))
        for i in range(self.layer_num):
            self.layers3.append(FALayer(self.g, hidden_dim))

        self.t1 = torch.nn.Linear(in_dim, hidden_dim)
        self.t_mu = torch.nn.Linear(hidden_dim, out_dim)
        self.t_logstd = torch.nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t_mu.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t_logstd.weight, gain=1.414)

    def forward(self, h):
        h = torch.relu(self.t1(h))
        raw = h
        for i in range(self.layer_num):
            h_mu = self.layers2[i](h)
            h_mu = self.eps * raw + h_mu
        for i in range(self.layer_num):
            h_logstd = self.layers3[i](h)
            h_logstd = self.eps * raw + h_logstd
        return self.t_mu(h_mu),self.t_logstd(h_logstd)
def compute_scores(z, test_pos, test_neg):
    test = torch.cat((test_pos, test_neg), dim=1)
    labels = torch.zeros(test.size(1), 1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt, dim=1))
    auc = roc_auc_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    ap = average_precision_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return auc, ap

result = []
for run in range(args.run):
    print("run:",run)

    #85/5/10 split training data
    link_train,link_test_val=train_test_split(range(0, edge_index.shape[1]),train_size=0.85)
    link_test,link_val=train_test_split(link_test_val,train_size=2/3)
    train_edge_list=torch.stack((edge_index[0][link_train],edge_index[1][link_train]))

    ori_adj=torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device), torch.Size([x.shape[0], x.shape[0]])).to_dense()
    adj=torch.sparse_coo_tensor(train_edge_list, torch.ones(train_edge_list.shape[1]).to(device), torch.Size([x.shape[0], x.shape[0]])).to_dense()
    ori_adj[ori_adj!=0]=1
    adj[adj!=0]=1
    adj_sym=adj+adj.t()
    adj_sym[adj_sym!=0]=1
    negative_adj=((adj+adj.t())==0)
    a_up=torch.triu(torch.ones(x.shape[0],x.shape[0]),diagonal=0).to(device)
    aup=adj_sym
    g,x1=get_g(train_edge_list.to('cpu'),x.to('cpu'))
    g=g.to(device)
    deg = g.in_degrees().cuda().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm
    x1=x1.to(device)
    model = VGAE(FAGCN(g.to(device),nfeat, 256,args.nembed,2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_auc=0
    #negative sampling for each connected node
    pos_val =torch.stack((edge_index[0][link_val], edge_index[1][link_val]))
    pos_test = torch.stack((edge_index[0][link_test], edge_index[1][link_test]))
    neg_test=[]
    neg_tra=[]
    neg_val=[]
    for m_index in range(args.m):
        n = structured_negative_sampling(edge_index)
        neg_test.append(torch.stack((n[0][link_test], n[2][link_test])))
        neg_tra.append(torch.stack((n[0][link_train], n[2][link_train])))
        neg_val.append(torch.stack((n[0][link_val], n[2][link_val])))
    neg_test=torch.cat(neg_test, dim=1)
    neg_train=torch.cat(neg_tra, dim=1)
    neg_val=torch.cat(neg_val, dim=1)
    pos_test_adj = torch.sparse_coo_tensor(pos_test, torch.ones(pos_test.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    neg_test_adj = torch.sparse_coo_tensor(neg_test, torch.ones(neg_test.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    pos_train_adj = torch.sparse_coo_tensor(train_edge_list, torch.ones(train_edge_list.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    neg_train_adj = torch.sparse_coo_tensor(neg_train, torch.ones(neg_train.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_train = torch.cat((train_edge_list, neg_train), dim=1)
    all_test = torch.cat((pos_test, neg_test), dim=1)
    all_test_adj = torch.sparse_coo_tensor(all_test, torch.ones(all_test.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_train_adj = torch.sparse_coo_tensor(all_train, torch.ones(all_train.shape[1]).to(device),
                                            torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_train_adj[all_train_adj != 0] = 1
    all_test_adj[all_test_adj != 0] = 1
    m=0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x1)
        loss=0
        for m_index in range(args.m):
            loss += model.recon_loss(z, train_edge_list,neg_tra[m_index])
        loss = loss/args.m + (1 / x.shape[0]) * model.kl_loss()
        loss.backward()
        optimizer.step()

        model.eval()
        auc,_=compute_scores(z,pos_val,neg_val)
        if(auc>best_auc):
            m=0
            best_auc=auc
            weights = deepcopy(model.state_dict())
        else:
            m+=1
        if(m>300):
            break
        print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc)
    model.load_state_dict(weights)
    z = model.encode(x1)
    test_auc,_=compute_scores(z,pos_test,neg_test)
    result.append(test_auc)
import numpy as np
result = np.array(result)
if args.dataset in ['twitch-e','fb100']:
    #filename = f'performance/{args.dataset}_{args.sub_dataset}_gat.csv'
    filename = f'performance/{args.dataset}_fagcn.csv'
else:
    filename = f'performance/{args.dataset}_fagcn.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{result.mean():.3f} ± {result.std():.3f}   "+f"{result},"+
                    f"nhidden " + f"{args.nhidden}," +
                    f"nembed " + f"{args.nembed}," +
                    f"layer " + f"{args.layer}," +
                    f"dataset " + f"{args.dataset}," +
                    f"sub_dataset " + f"{args.sub_dataset}," +
                    f"run " + f"{args.run}," +
                    f"epochs " + f"{args.epochs}," +
                    f"heads " + f"{args.head}," +
                    f"m " + f"{args.m}," +
                    f"lr " + f"{args.lr}\n")