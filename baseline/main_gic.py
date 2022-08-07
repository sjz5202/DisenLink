import random
from re import A
from dataset import WikipediaNetwork
import argparse
import torch
from torch_geometric.datasets import Planetoid, WebKB
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from sklearn.decomposition import PCA
from torch_geometric.utils import to_dense_adj,structured_negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from other_hetero_datasets import load_nc_dataset
from baselines.gic import GIC
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
parser.add_argument('--m', type = int, default = 5)
parser.add_argument('--alpha', type = float, default = 0.5)
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

if args.dataset in ["Texas", "Wisconsin","Cornell"]:
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
if args.dataset in ['twitch-e','fb100','deezer-europe','ogbn-proteins','arxiv-year']:
    nfeat = data['node_feat'].shape[1]
    x=data['node_feat'].to(device)
    edge_index = data['edge_index'].to(device)
    if args.dataset in ['twitch-e']:
        e1=torch.stack((edge_index[1],edge_index[0])).to(device)
        edge_index=torch.cat((edge_index,e1),dim=1).to(device)
print(data)
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)
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

    pos_weight = float(aup.shape[0] * aup.shape[0] - aup.sum()) / aup.sum()
    norm = aup.shape[0] * aup.shape[0] / float((aup.shape[0] * aup.shape[0] - aup.sum()) * 2)
    adj_label = aup


    adj_up=preprocess_graph(aup.cpu().numpy())
    adj_up = torch.sparse.FloatTensor(torch.LongTensor(adj_up[0].T),
                                        torch.FloatTensor(adj_up[1]),
                                        torch.Size(adj_up[2])).to(device)
    

    #################################### NEW ##############################################
    # model=VGAE(nfeat,args.nhidden,args.nembed,adj_up).to('cuda')
    nb_nodes = x.shape[0]
    ft_size = nfeat
    nonlinearity = 'prelu'
    num_clusters = int(10) # we may need to change for different datasets
    beta_GIC = 100
    model = GIC(nb_nodes,ft_size, args.nhidden, nonlinearity, num_clusters, beta_GIC,device).to(device)

    #################################### NEW ##############################################
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
    for epoch in range(args.epochs):
        model.train()
        #################################### OLD ##############################################
        # h_all_factor,a_pred,x_pred=model(x,adj_up)
        # neg = []
        # negloss = 0
        # loss0=F.binary_cross_entropy(a_pred[all_train_adj==1].unsqueeze(0),ori_adj[all_train_adj==1].unsqueeze(0))
        # #loss1 = loss_function(x_pred, x)
        # #loss = loss1 + args.loss_weight*loss0
        # loss=loss0
        #################################### OLD ##############################################
        # logits, logits2  = model(features, shuf_fts, sp_adj, sparse, None, None, None, beta) 
        # loss = alpha* b_xent(logits, lbl)  + (1-alpha)*b_xent(logits2, lbl) 
        #################################### NEW ##############################################
        idx = np.random.permutation(nb_nodes)
        shuf_fts = x[idx, :]
        lbl_1 = torch.ones((1, nb_nodes))
        lbl_2 = torch.zeros((1, nb_nodes))
        lbl = torch.cat((lbl_1, lbl_2), 1).to(device)
        sparse = True
        b_xent = torch.nn.BCEWithLogitsLoss()
        logits, logits2, a_pred, emb = model(x, shuf_fts, adj_up, sparse, None, None, None, beta_GIC,device)
        #loss0=F.binary_cross_entropy(a_pred[all_train_adj==1].unsqueeze(0),ori_adj[all_train_adj==1].unsqueeze(0))
        loss0=F.binary_cross_entropy(a_pred[pos_train_adj==1].unsqueeze(0),ori_adj[pos_train_adj==1].unsqueeze(0))+F.binary_cross_entropy(a_pred[neg_train_adj==1].unsqueeze(0),ori_adj[neg_train_adj==1].unsqueeze(0))/args.m
        loss1 = args.alpha* b_xent(logits, lbl)  + (1-args.alpha)*b_xent(logits2, lbl) 
        loss = loss1
        #################################### END ##############################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        pred_score=a_pred[all_val_adj==1]
        link_label=ori_adj[all_val_adj==1]
        auc=roc_auc_score(link_label.cpu().detach().numpy(),pred_score.cpu().detach().numpy())
        if(auc>best_auc):
            best_auc=auc
            weights = deepcopy(model.state_dict())
        print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc)
    
    model.load_state_dict(weights)
    logits, logits2, a_pred, emb = model(x, shuf_fts, adj_up, sparse, None, None, None, beta_GIC,device)
    pred_score=a_pred[all_test_adj==1]
    link_label=ori_adj[all_test_adj==1]
    auc=roc_auc_score(link_label.cpu().detach().numpy(),pred_score.cpu().detach().numpy())
    result.append(auc)
import numpy as np
result = np.array(result)
if args.dataset in ['twitch-e','fb100']:
    filename = f'performance/{args.dataset}_{args.sub_dataset}_gic.csv'
else:
    filename = f'performance/{args.dataset}_gic.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{result.mean():.3f} ± {result.std():.3f}   "
                    f"beta_GIC " + f"{beta_GIC}," +
                    f"nhidden " + f"{args.nhidden}," +
                    f"nembed " + f"{args.nembed}," +
                    f"layer " + f"{args.layer}," +
                    f"dataset " + f"{args.dataset}," +
                    f"sub_dataset " + f"{args.sub_dataset}," +
                    f"run " + f"{args.run}," +
                    f"epochs " + f"{args.epochs}," +
                    f"lr " + f"{args.lr}," +
                    f"alpha " + f"{args.alpha}," +
                    f"loss_weight " + f"{args.loss_weight} \n")