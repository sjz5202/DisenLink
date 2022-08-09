from random import seed
from torch_geometric.loader import NeighborSampler,RandomNodeSampler,ClusterData, ClusterLoader ,GraphSAINTNodeSampler,GraphSAINTEdgeSampler,GraphSAINTRandomWalkSampler
#, ClusterData, ClusterLoader, Data, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, RandomNodeSampler
from other_hetero_datasets import load_nc_dataset
from torch_geometric.data import Data
import torch
data=load_nc_dataset(dataname='arxiv-year').graph
x=data['node_feat']
edge_index = data['edge_index']
'''
num=1000
for i in range(1):
    nodeid=torch.randint(low=0,high=x.shape[0],size=(num,))
    sub=NeighborSampler(edge_index,sizes=[100],node_idx=nodeid,batch_size=num)
    for batch_size, n_id, adjs in sub:
        print(batch_size)
        print(x[n_id].shape)
        print(adjs.edge_index.shape)
'''
data = Data(x=x, edge_index=edge_index)
'''
sub1=RandomNodeSampler(data, num_parts=40, shuffle=False, num_workers=8)
for batch in sub1:
        print(batch)
'''
'''
cluster_data = ClusterData(data, num_parts=40)
loader = ClusterLoader(cluster_data, batch_size=40, shuffle=True, num_workers=0)
for batch in loader:
        print(batch)
'''

loader=GraphSAINTRandomWalkSampler(data, batch_size=850, walk_length=20, shuffle=True, num_workers=0, num_steps=1)

for batch in loader:
        print(batch)
        #torch.save(batch,'mini/year{}.pt'.format(i))

'''
loader=GraphSAINTNodeSampler(data, batch_size=4000, shuffle=True, num_workers=0, num_steps=2)
for batch in loader:
        print(batch)
'''
'''
loader=GraphSAINTEdgeSampler(data, batch_size=4000, shuffle=True, num_workers=0, num_steps=1)
for batch in loader:
        print(batch)
        print(batch.edge_index)
'''
    