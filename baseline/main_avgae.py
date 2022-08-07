from torch_geometric.nn import GCNConv, ARGVA, ARGA
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from re import A, M
from dataset import WikipediaNetwork
import argparse
import torch
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
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--beta', type=float, default=0.85)
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
parser.add_argument('--m', type = int, default = 1)
parser.add_argument("--miniid", type=int, default=0)

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)
transform = T.Compose([T.NormalizeFeatures()])
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x
def compute_scores(z,test_pos,test_neg):
    test = torch.cat((test_pos, test_neg),dim=1)
    labels = torch.zeros(test.size(1),1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt,dim=1))
    auc = roc_auc_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    ap = average_precision_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return auc,ap
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
if args.dataset in ['crocodile']:
    nfeat=data.x.shape[1]
    x=data.x
    edge_index = data.edge_index
if args.dataset in ["texas", "wisconsin","cornell"]:
    nfeat=data.x.shape[1]
    x = data.x
    #x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ['photo']:
    nfeat=data.x.shape[1]
    x = data.x
    y=data.y
    #x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ['twitch-e','fb100','deezer-europe','ogbn-proteins','arxiv-year']:
    nfeat = data['node_feat'].shape[1]
    x=data['node_feat'].to(device)
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
if args.dataset in ['year']:
    data=torch.load('mini/year{}.pt'.format(args.miniid)).to(device)
    x=data.x
    #x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    nfeat=x.shape[1]
    edge_index = data.edge_index 
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
    encoder = Encoder(nfeat, hidden_channels=args.nhidden, out_channels=args.nembed)
    discriminator = Discriminator(in_channels=args.nembed, hidden_channels1=128, hidden_channels2=128,
                                  out_channels=32)
    model = ARGVA(encoder, discriminator).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=0.001)
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
        encoder_optimizer.zero_grad()
        z = model.encode(x, train_edge_list)
        for i in range(3):
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()
        loss=0
        for m_index in range(args.m):
            loss += model.recon_loss(z, train_edge_list,neg_tra[m_index])
        loss = loss/args.m + model.reg_loss(z)
        loss = loss + (1 / x.shape[0]) * model.kl_loss()
        loss.backward()
        encoder_optimizer.step()
        model.eval()
        auc,_=compute_scores(z,pos_val,neg_val)
        if(auc>best_auc):
            m=0
            best_auc=auc
            weights = deepcopy(model.state_dict())
        else:
            m+=1
        if(m>500):
            break
        print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc)
    model.load_state_dict(weights)
    z = model.encode(x, train_edge_list)
    test_auc,_=compute_scores(z,pos_test,neg_test)
    result.append(test_auc)
import numpy as np
result = np.array(result)
if args.dataset in ['twitch-e','fb100']:
    #filename = f'performance/{args.dataset}_{args.sub_dataset}_avgae.csv'
    filename = f'performance/{args.dataset}_avgae.csv'
else:
    filename = f'performance/{args.dataset}_avgae.csv'
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
                    f"lr " + f"{args.lr}\n")