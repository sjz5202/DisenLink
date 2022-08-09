from ossaudiodev import SNDCTL_DSP_BIND_CHANNEL
import random
from re import A
from dataset import WikipediaNetwork
import argparse
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid, WebKB
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam,AdamW
from sklearn.decomposition import PCA
from model_fea import Disentangle,Factor,Disentangle_2layer,Disentangle_fea1
from torch_geometric.utils import to_dense_adj,structured_negative_sampling
from sklearn.cluster import KMeans
from cluster_evaluation import eva
from sklearn.metrics import roc_auc_score, average_precision_score
from other_hetero_datasets import load_nc_dataset
from copy import deepcopy
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(numpy.float32)
    indices = torch.from_numpy(
        numpy.vstack((sparse_mx.row, sparse_mx.col)).astype(numpy.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--nfactor', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=128,
                    help='Number of feature units.')
parser.add_argument('--nhidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nembed', type=int, default=16,
                    help='Number of embedding units.')
parser.add_argument('--epochs', type=int,  default=2000, help='Number of epochs to train.')
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument('--dataset', type=str, default='squirrel', help='Random seed.')
parser.add_argument('--sub_dataset', type=str, default='squirrel', help='Random seed.')
parser.add_argument('--run', type = int, default = 1)
parser.add_argument('--gpu', type = int, default = 5)
parser.add_argument('--m', type = int, default = 5)
parser.add_argument('--save', type = int, default = 0)

parser.add_argument('--loss_weight', type=int, default=20)
parser.add_argument("--layer", type=int, default=1)
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
    data_label=load_nc_dataset(args.dataset,args.sub_dataset).label
    nlabel=torch.max(data_label)+1
if args.dataset in ["Texas", "Wisconsin","Cornell"]:
    dataset = WebKB(root='data/',name=args.dataset)
    data = dataset[0].to(device)
    x=data.x
    nfeat=x.shape[1]
    edge_index=data.edge_index
    data_label=data.y
    nlabel=int(torch.max(data_label))+1

if args.dataset in ["crocodile", "squirrel",'chameleon']:
    if args.dataset=="crocodile":
        dataset = WikipediaNetwork('data/',name=args.dataset,geom_gcn_preprocess=False)
    else:
        dataset = WikipediaNetwork('data/',name=args.dataset)
        dataset1 = WikipediaNetwork('data_pre_false/', name=args.dataset, geom_gcn_preprocess=False)
        data1 = dataset1[0].to(device)
    data = dataset[0].to(device)
    twohop=torch_geometric.transforms.two_hop.TwoHop()
    twodata=twohop(data1)
#nfeat=data.x.shape[1]
if args.dataset in ["squirrel",'chameleon']:
    x = data.x
    data_label=data.y
    nlabel=int(torch.max(data_label))+1
    nfeat = x.shape[1]
    #x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index=data1.edge_index
    edge_index2hop=twodata.edge_index
if args.dataset in ['crocodile']:
    nfeat=data.x.shape[1]
    data_label=data.y
    nlabel=int(torch.max(data_label))+1
    x=data.x
    edge_index = data.edge_index
if args.dataset in ['twitch-e','fb100','deezer-europe','ogbn-proteins','arxiv-year']:
    nfeat = data['node_feat'].shape[1]
    x=data['node_feat'].to(device)
    x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
    edge_index = data['edge_index'].to(device)
    twohop=torch_geometric.transforms.two_hop.TwoHop()
    datahop = torch_geometric.data.Data(x=x, edge_index=edge_index)
    twodata=twohop(datahop)
    edge_index2hop=twodata.edge_index
    if args.dataset in ['twitch-e']:
        e1=torch.stack((edge_index[1],edge_index[0])).to(device)
        edge_index=torch.cat((edge_index,e1),dim=1).to(device)
print(x.shape)
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
result = []
for run in range(args.run):
    print("run:",run)
    #85/5/10 split training data
    ori_adj1=torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device), torch.Size([x.shape[0], x.shape[0]])).to_dense()
    ori_adj2=torch.sparse_coo_tensor(edge_index2hop, torch.ones(edge_index2hop.shape[1]).to(device), torch.Size([x.shape[0], x.shape[0]])).to_dense()
    ori_adj1[ori_adj1!=0]=1
    ori_adj2[ori_adj2!=0]=1
    ori_adj=ori_adj1
    if(args.layer==1):
        model=Disentangle_fea1(nfeat,args.nhidden,args.nembed,nlabel=nlabel).to(device)
    else:
        model = Disentangle_fea1(nfeat, args.hidden, nfactor=args.nfactor, beta=args.beta, t=args.temperature).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0)
    best_auc=0
    #negative sampling for each connected node
    best_acc=0
    m=0
    if(1):
        for epoch in range(1000):
            print(epoch)
            model.train()
            z,x_pred=model(x,ori_adj)
            loss_fea=loss_function(x_pred, x)
            loss=loss_fea
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if(epoch==999):
                model.eval()
                kmeans = KMeans(n_clusters=nlabel, n_init=20)
                y_pred = kmeans.fit_predict(z.data.cpu().numpy())
                acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), y_pred,epoch)
                weights = deepcopy(model.state_dict())
                torch.save(model.state_dict(),'saved_model/embed32.pkl')
            
    if(1):
        model.load_state_dict(torch.load('saved_model/embed32.pkl'))
    with torch.no_grad():
        z,x_pred=model(x,ori_adj)
    kmeans1= KMeans(n_clusters=nlabel, n_init=20)
    y_pred1 = kmeans1.fit_predict(z.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), y_pred1,1)
    edge_label_wise=[]
    for label in range(int(data.y.max()) + 1):
        mask = (y_pred1 == label)
        mask=torch.tensor(mask)
        sub_adj = data.edge_index[:, mask[data.edge_index[1]]]
        if sub_adj.shape[1] <= 0:
            continue
        dense_adj = to_dense_adj(sub_adj, max_num_nodes=len(data.x))[0]
        dense_adj = sparse_mx_to_torch_sparse_tensor(csr_matrix(dense_adj.cpu().numpy())).to(device)
        edge_label_wise.append(dense_adj)
    h=[]
    for edge_index in edge_label_wise:
        if edge_index.is_sparse:
            h.append(torch.spmm(edge_index, z))
        else:
            h.append(edge_index @ z)
    h.append(z)
    h = torch.cat(h, dim=1)
    kmeans_final= KMeans(n_clusters=nlabel, n_init=20)
    y_pred_final = kmeans_final.fit_predict(h.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), y_pred_final,1)
    model1=Disentangle_fea1(args.nembed*6,args.nhidden,args.nembed,nlabel=nlabel).to(device)
    for epoch in range(400):
        #print(epoch)
        model1.train()
        z1,x_pred=model1(h,ori_adj)
        loss_fea=loss_function(x_pred, h)
        loss=loss_fea
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch==399):
            model.eval()
            kmeans = KMeans(n_clusters=nlabel, n_init=20)
            y_pred = kmeans.fit_predict(z1.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), y_pred,epoch)