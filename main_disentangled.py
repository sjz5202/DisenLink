import torch
from torch_geometric.datasets import Planetoid, WebKB,Amazon
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from sklearn.decomposition import PCA
from torch_geometric.utils import to_dense_adj,structured_negative_sampling,homophily,degree
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
import argparse
from model import Disentangle
from re import A
from dataset import WikipediaNetwork
from other_hetero_datasets import load_nc_dataset
from copy import deepcopy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--nfactor', type=int, default=15)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=128,
                    help='Number of feature units.')
parser.add_argument('--nhidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--nembed', type=int, default=32,
                    help='Number of embedding units.')
parser.add_argument('--epochs', type=int,  default=2000, help='Number of epochs to train.')
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument('--dataset', type=str, default='chameleon', help='Random seed.')
parser.add_argument('--sub_dataset', type=str, default='Amherst41', help='Random seed.')
parser.add_argument('--run', type = int, default = 1)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--m', type = int, default = 5,help='number of factors')
parser.add_argument('--save', type = int, default = 0)
parser.add_argument('--loss_weight', type=int, default=20)
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--miniid", type=int, default=9)
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

if args.dataset in ["crocodile", "squirrel",'chameleon']:
    if args.dataset=="crocodile":
        dataset = WikipediaNetwork('data/',name=args.dataset,geom_gcn_preprocess=False)
    else:
        dataset = WikipediaNetwork('data/',name=args.dataset)
        dataset1 = WikipediaNetwork('data_pre_false/', name=args.dataset, geom_gcn_preprocess=False)
        data1 = dataset1[0].to(device)
    data = dataset[0].to(device)
    y=data.y
    one_hot_y=F.one_hot(y, num_classes=5).to(torch.double)
    homo=torch.mm(one_hot_y,one_hot_y.t())
#nfeat=data.x.shape[1]
if args.dataset in ['photo']:
    nfeat=data.x.shape[1]
    x = data.x
    y=data.y
    x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ["texas", "wisconsin","cornell"]:
    nfeat=data.x.shape[1]
    x = data.x
    y=data.y
    x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ["squirrel",'chameleon']:
    x = data1.x
    nfeat=x.shape[1]
    x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index=data1.edge_index
    y=data.y
if args.dataset in ['crocodile']:
    nfeat=data.x.shape[1]
    y=data.y
    x=data.x
    x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data.edge_index
if args.dataset in ['twitch-e','fb100','deezer-europe','ogbn-proteins','arxiv-year']:
    nfeat = data['node_feat'].shape[1]
    x=data['node_feat'].to(device)
    x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data['edge_index'].to(device)
    if args.dataset in ['twitch-e']:
        e1=torch.stack((edge_index[1],edge_index[0])).to(device)
        edge_index=torch.cat((edge_index,e1),dim=1).to(device)
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root='./data/',name=args.dataset,transform=None)
    data = dataset[0].to(device)
    x=data.x
    y=data.y
    nfeat=x.shape[1]
    edge_index = data.edge_index
if args.dataset in ['year']:
    data=torch.load('mini/year{}.pt'.format(args.miniid)).to(device)
    x=data.x
    x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
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
    adj_up=a_up*adj_sym

    if(args.layer==1):
        model=Disentangle(nfeat,args.nhidden,args.nembed,nfactor=args.nfactor,beta=args.beta,t=args.temperature).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
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
    pos_val_adj = torch.sparse_coo_tensor(pos_val, torch.ones(pos_val.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    neg_val_adj = torch.sparse_coo_tensor(neg_val, torch.ones(neg_val.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    pos_train_adj = torch.sparse_coo_tensor(train_edge_list, torch.ones(train_edge_list.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    neg_train_adj = torch.sparse_coo_tensor(neg_train, torch.ones(neg_train.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_train = torch.cat((train_edge_list, neg_train), dim=1)
    all_test = torch.cat((pos_test, neg_test), dim=1)
    all_val=torch.cat((pos_val, neg_val), dim=1)
    all_test_adj = torch.sparse_coo_tensor(all_test, torch.ones(all_test.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_val_adj = torch.sparse_coo_tensor(all_val, torch.ones(all_val.shape[1]).to(device),
                                           torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_train_adj = torch.sparse_coo_tensor(all_train, torch.ones(all_train.shape[1]).to(device),
                                            torch.Size([x.shape[0], x.shape[0]])).to_dense()
    all_train_adj[all_train_adj != 0] = 1
    all_test_adj[all_test_adj != 0] = 1
    all_val_adj[all_val_adj != 0] = 1
    m=0
    for epoch in range(args.epochs):
        model.train()
        h_all_factor,a_pred =model(x,adj_sym)
        loss0=F.binary_cross_entropy(a_pred[pos_train_adj==1].unsqueeze(0),ori_adj[pos_train_adj==1].unsqueeze(0))+F.binary_cross_entropy(a_pred[neg_train_adj==1].unsqueeze(0),ori_adj[neg_train_adj==1].unsqueeze(0))/args.m
        loss=loss0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        pred_score=a_pred[all_val_adj==1]
        link_label=ori_adj[all_val_adj==1]
        auc=roc_auc_score(link_label.cpu().detach().numpy(),pred_score.cpu().detach().numpy())

        if(auc>best_auc):
            m=0
            best_auc=auc
            weights = deepcopy(model.state_dict())
        else:
            m+=1
        if(m>200):
            break
        print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc)
    model.load_state_dict(weights)
    h_all_factor,a_pred =model(x,adj_sym)
    pred_score=a_pred[all_test_adj==1]
    link_label=ori_adj[all_test_adj==1]
    auc=roc_auc_score(link_label.cpu().detach().numpy(),pred_score.cpu().detach().numpy(),average='weighted')
    print("test auc:",auc)
    result.append(auc)

result = np.array(result)
print('final',result.mean(),result.std())
if(args.save==1):
    if args.dataset in ['twitch-e','fb100']:
        filename = f'performance/{args.dataset}_{args.sub_dataset}disentangle_nfactor.csv'
    else:
        filename = f'performance/{args.dataset}_disentangle_nfactor.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else 'none'
        write_obj.write(f"{result.mean():.3f} ± {result.std():.3f},"+f"{result},"
                        f"beta " + f"{args.beta}," +
                        f"temperature " + f"{args.temperature}," + 
                        f"nfactor " + f"{args.nfactor}," +
                        f"nhidden " + f"{args.nhidden}," +
                        f"nembed " + f"{args.nembed}," +
                        f"layer " + f"{args.layer}," +
                        f"dataset " + f"{args.dataset}," +
                        f"sub_dataset " + f"{args.sub_dataset}," +
                        f"run " + f"{args.run}," +
                        f"epochs " + f"{args.epochs}," +
                        f"lr " + f"{args.lr}," +
                        f"miniid " + f"{args.miniid}," +
                        f"m " + f"{args.m}\n")
