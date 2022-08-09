#1
from dataset import WikipediaNetwork
import argparse
import torch
from torch_geometric.datasets import Planetoid, WebKB
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from sklearn.decomposition import PCA
from model import Disentangle,Factor,Disentangle2
from torch_geometric.utils import to_dense_adj,structured_negative_sampling
from sklearn.cluster import KMeans
from cluster_code_res.cluster_evaluation import eva
# Training settings

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=128,
                    help='Number of feature units.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument('--dataset', type=str, default='crocodile', help='Random seed.')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)
transform = T.Compose([T.NormalizeFeatures()])

if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(root='./data/',name=args.dataset)
    data = dataset[0].to(device)
    data.train_mask = torch.unsqueeze(data.train_mask, dim=1)
    data.val_mask = torch.unsqueeze(data.val_mask, dim=1)
    data.test_mask = torch.unsqueeze(data.test_mask, dim=1)

if args.dataset in ["Texas", "Wisconsin","Cornell"]:
    dataset = WebKB(root='./data/',name=args.dataset)
    data = dataset[0].to(device)

if args.dataset in ["crocodile", "squirrel",'chameleon']:
    if args.dataset=="crocodile":
        dataset = WikipediaNetwork('./data/',name=args.dataset,geom_gcn_preprocess=False)
    else:
        dataset = WikipediaNetwork('./data/',name=args.dataset)
        dataset1 = WikipediaNetwork('./data_pre_false/', name=args.dataset, geom_gcn_preprocess=False)
        data1 = dataset1[0].to(device)
    data = dataset[0].to(device)
print(data)
if args.dataset in ["chameleon", "squirrel"]:
    nfeat = args.nfeat
    #pca = PCA(n_components=nfeat)
    #x=torch.tensor(pca.fit_transform(data.x.cpu().numpy())).to(device)
    x = data1.x
    x = (x - torch.mul(torch.ones(x.shape).to(device), torch.mean(x, dim=1).unsqueeze(dim=1))) / torch.std(x,dim=1).unsqueeze(dim=1)
else:
    nfeat=data.x.shape[1]
    x=data.x

posi=data.edge_index[0]
adj=torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]).to(device), torch.Size([data.x.shape[0], data.x.shape[0]])).to_dense()
adj_sym=adj+adj.t()
adj_sym[adj_sym!=0]=1
negative_adj=((adj+adj.t())==0)
#nfeat=data.x.shape[1]
print(adj)

if(args.layer==1):
    model=Disentangle(nfeat,args.hidden,nfactor=5,beta=0.85,t=args.temperature).to(device)
else:
    model = Disentangle2(nfeat, args.hidden, nfactor=5, beta=0.85, t=args.temperature).to(device)
loss_function = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
for epoch in range(args.epochs):
    model.train()
    h_all_factor,a_pred,x_pred=model(x,adj)

    #loss 0
    neg = []
    if(0):
        negloss = 0
        for i in range(20):
            n = structured_negative_sampling(data.edge_index)[2]
            neg_edge_list=torch.stack((posi,n))
            neg_adj = torch.sparse_coo_tensor(neg_edge_list, torch.ones(data.edge_index.shape[1]).to(device),
                                          torch.Size([data.x.shape[0], data.x.shape[0]])).to_dense()
            negloss+=loss_function(neg_adj*a_pred,neg_adj*adj)
        negloss=negloss/20
        loss0=loss_function(adj_sym*a_pred,adj_sym)+negloss
    # * data.x.shape[0] / torch.sum(negative_adj)
    loss0 = loss_function(adj_sym * a_pred, adj_sym) + loss_function(a_pred[negative_adj], adj_sym[negative_adj])* data.x.shape[0] / torch.sum(negative_adj)
    #loss0=loss_function(a_pred.view(-1),adj_sym.view(-1))

    # feature reconstruction loss
    loss1=loss_function(x_pred,x)
    loss = loss1+20*loss0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    kmeans = KMeans(n_clusters=5, n_init=20).fit(h_all_factor.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(data.y.cpu().numpy(), kmeans.labels_,loss0,loss1,loss,epoch)
    #print(epoch,':',loss)
