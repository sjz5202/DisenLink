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
from model_fea import Disentangle,Factor,Disentangle_2layer,Disentangle_fea
from torch_geometric.utils import to_dense_adj,structured_negative_sampling
from sklearn.cluster import KMeans
from cluster_evaluation import eva
from sklearn.metrics import roc_auc_score, average_precision_score
from other_hetero_datasets import load_nc_dataset
from copy import deepcopy
from sklearn.model_selection import train_test_split
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
parser.add_argument('--nembed', type=int, default=6,
                    help='Number of embedding units.')
parser.add_argument('--epochs', type=int,  default=2000, help='Number of epochs to train.')
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument('--dataset', type=str, default='chameleon', help='Random seed.')
parser.add_argument('--sub_dataset', type=str, default='Amherst41', help='Random seed.')
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
    x = data1.x
    data_label=data1.y
    nlabel=int(torch.max(data_label))+1
    nfeat = x.shape[1]
    x=(x-torch.mul(torch.ones(x.shape).to(device),torch.mean(x,dim=1).unsqueeze(dim=1)))/torch.std(x,dim=1).unsqueeze(dim=1)
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
        model=Disentangle_fea(nfeat,args.nhidden,args.nembed,nfactor=args.nfactor,beta=args.beta,nlabel=nlabel,t=args.temperature).to(device)
    else:
        model = Disentangle_fea(nfeat, args.hidden, nfactor=args.nfactor, beta=args.beta, t=args.temperature).to(device)
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
            h_all_factor,a_pred,x_pred,q,h_pooling=model(x,ori_adj)
            loss_a=F.binary_cross_entropy(a_pred.unsqueeze(0),ori_adj.unsqueeze(0))
            loss_fea=loss_function(x_pred, x)
            #loss=loss_a+loss_fea
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if(epoch>=7000):
                model.eval()
                kmeans = KMeans(n_clusters=nlabel, n_init=20)
                y_pred = kmeans.fit_predict(h_pooling.data.cpu().numpy())
                acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), y_pred,epoch)
                if((acc>best_acc)):
                    best_acc=acc
                    #weights = deepcopy(model.state_dict())
                    #torch.save(model.state_dict(), 'saved_model/nembed32_3.pkl')
            if(epoch==(args.epochs-1)):
                weights = deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'saved_model/nembed32_cat.pkl')


            
    if(0):
        model.load_state_dict(torch.load('saved_model/nembed32_cat.pkl'))
    with torch.no_grad():
        h_all_factor,a_pred,x_pred,Q,h_pooling=model(x,ori_adj)
    kmeans= KMeans(n_clusters=nlabel, n_init=20)
    y_pred = kmeans.fit_predict(h_pooling.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), y_pred,1)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    best_acc=0
    for epoch in range(args.epochs):
        model.train()
        if epoch % 3 == 0:
            # update_interval
            h_all_factor,a_pred,x_pred,Q,h_pooling=model(x,ori_adj)
            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(data_label.cpu().numpy(), q,epoch)
            if(acc>best_acc):
                m=0
                best_acc=acc
                weights = deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'saved_model/nembed32_selft_cat.pkl')
            #print("epoch:",epoch,"val_auc:",best_auc,f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
        h_all_factor,a_pred,x_pred,q,h_pooling=model(x,ori_adj)
        p = target_distribution(Q.detach())
        loss1 = F.kl_div(q.log(), p, reduction='batchmean')
        #loss0=F.binary_cross_entropy(a_pred[pos_train_adj==1].unsqueeze(0),ori_adj[pos_train_adj==1].unsqueeze(0))+F.binary_cross_entropy(a_pred[neg_train_adj==1].unsqueeze(0),ori_adj[neg_train_adj==1].unsqueeze(0))/args.m
        loss0=F.binary_cross_entropy(a_pred.unsqueeze(0),ori_adj.unsqueeze(0))
        #loss2 = loss_function(x_pred, x)
        loss = loss1 + 20*loss0
        #print(loss0,loss1)
        #loss=loss0+20*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #kmeans = KMeans(n_clusters=5).fit(h_pooling.data.cpu().numpy())
        #acc, nmi, ari, f1 = eva(data.y.cpu().numpy(), kmeans.labels_,epoch)
        #print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc,f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
        '''
        if(auc>best_auc):
            m=0
            best_auc=auc
            weights = deepcopy(model.state_dict())
        else:
            m+=1
        if(m>100):
            break
        '''
        #print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc,f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
        #print("epoch:",epoch,"loss:",loss.item(),"val_auc:",best_auc)
    '''
    model.load_state_dict(weights)
    _,a_pred,_,_=model(x,adj_sym)
    pred_score=a_pred[all_test_adj==1]
    link_label=ori_adj[all_test_adj==1]
    auc=roc_auc_score(link_label.cpu().detach().numpy(),pred_score.cpu().detach().numpy())
    print("test auc:",auc)
    result.append(auc)
    '''

import numpy as np
result = np.array(result)
if(args.save==1):
    if args.dataset in ['twitch-e','fb100']:
        filename = f'performance/{args.dataset}_disentangle1.csv'
    else:
        filename = f'performance/{args.dataset}_disentangle.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
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
                        f"m " + f"{args.m}\n")