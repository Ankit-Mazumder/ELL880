import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
#from torch_geometric.loader import DataLoader 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv
import pandas as pd
import ast
import torch
from torch.nn import Module, Linear
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Module, Linear
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import pickle
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib
import time
from torch_geometric.nn import GraphSAGE
import gc
torch.cuda.empty_cache()
gc.collect()
matplotlib.use('Agg')

df = pd.read_csv('/home/scai/msr/aiy227513/scratch/research/superSetAnalysis.csv')
node_ids =[]
node_features = []
req_A=['citationVelocity', 'influentialCitationCount', 'isOpenAccess',
       'isPublisherLicence', 'numCitebBy', 'numCiting','year']
for index, row in df.iterrows():
    paperId = row['paperId']
    node_ids.append(paperId)
    f=[]
    for attribute in req_A:
            if attribute=="isOpenAccess":
             if row[attribute]==True:attribute_value=1
             else:attribute_value=0
            elif attribute=="isPublisherLicence":
              if row[attribute]==True:attribute_value=1
              else:attribute_value=0
            else:
               attribute_value = row[attribute]
            f.append(attribute_value)
    node_features.append(np.array(f).astype(np.float32))

df_test = pd.read_csv('/home/scai/msr/aiy227513/scratch/research/data/superSetAnalysis.csv').sample(frac=1)
node_ids_test =[]
node_features_test = []
for index, row in df_test.iterrows():
    paperId = row['paperId']
    if paperId in node_ids:continue
    node_ids_test.append(paperId)
    f=[]
    for attribute in req_A:
            if attribute=="isOpenAccess":
             if row[attribute]==True:attribute_value=1
             else:attribute_value=0
            elif attribute=="isPublisherLicence":
              if row[attribute]==True:attribute_value=1
              else:attribute_value=0
            else:
               attribute_value = row[attribute]
            f.append(attribute_value)
    node_features_test.append(np.array(f).astype(np.float32))


with open("/home/scai/msr/aiy227513/scratch/research/edges_same_ratio.pickle", "rb") as file:
    edges = pickle.load(file)
    file.close()
count=len(edges)
train_size=math.floor((len(edges)*0.8))
train_src=[]
train_dest=[]
train_label=[]
for i in range(train_size):
 if edges[i][2]==1:
    train_src.append(edges[i][0])
    train_dest.append(edges[i][1])
train_size=math.floor((len(edges)*0.8))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("/home/scai/msr/aiy227513/scratch/research/data/bert_encode.pkl", "rb") as file:
    embedding = pickle.load(file)
    file.close()
node_features = torch.tensor(node_features, dtype=torch.float)
embedding=torch.tensor(embedding, dtype=torch.float)
feature=torch.cat([node_features,embedding], dim=1).float()
feature =F.normalize(feature,p=2, dim=-1)
class LinkPredictionDataset(Dataset):
    def __init__(self, nodes, node_attributes, edges):
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        #self.edge_index=edge_index

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        src, dst,y = self.edges[index]
        # src_node = self.node_attributes[src]
        # dst_node = self.node_attributes[dst]
        #print(src, dst,self.node_attributes,self.y[index])
        return src, dst,self.node_attributes,y
class GCN(nn.Module):
    def __init__(self, in_channels=feature.shape[1], hidden_channels=512, out_channels=128, num_layers=4):
        super().__init__()
        self.conv1 = GraphSAGE(in_channels,out_channels,num_layers)
        #self.conv2 = GraphSAGE(hidden_channels, out_channels,num_layers)
        self.fc1 = Linear(out_channels*2,128)
        #self.fc2=Linear(256,128)
        self.fc3=Linear(128,64)
        self.fc4=Linear(64,16)
        self.fc5=Linear(16,1)
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        #x = self.conv2(x, edge_index).relu()
        #print(x.shape)
        return x

    def decode(self, z,src,dst):
        #return (z[0][src] * z[0][dst]).sum(dim=-1)
        x=torch.cat([z[0][src],z[0][dst]], dim=-1)
        x=F.leaky_relu(self.fc1(x))
        #x=self.fc2(x).relu()
        x=F.leaky_relu(self.fc3(x))
        x=F.leaky_relu(self.fc4(x))
        x=self.fc5(x)
        return x
edge_index = torch.tensor([train_src,train_dest], dtype=torch.long).to(device)        
model = GCN()
#if torch.cuda.device_count() > 1:model = nn.DataParallel(model)
model.load_state_dict(torch.load('/home/scai/msr/aiy227513/scratch/research/model/model_bert.pth', map_location=device),
                             strict=False)
g_truth=[]
prediction=[]
count=0

SAMPLE_SIZE = 1000

with torch.no_grad():
    #y = torch.tensor(test_label, dtype=torch.float)
    #edge_index = torch.tensor([test_src,test_dest], dtype=torch.long).to(device)
    test_dataset = LinkPredictionDataset(node_ids,feature,edges)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    recommendations = {}
    for batch_idx, batch in enumerate(test_dataloader):
        #src_nodes, dst_nodes,node_attributes,y = batch
        
#         src_nodes=batch[0].to(device)
        np.random.seed(batch_idx)
        src_nodes = np.random.randint(0, len(feature), size=SAMPLE_SIZE)
        src_nodes = torch.from_numpy(src_nodes).to(device)
#         print(dst_nodes.item())
        dst_node = batch[1].item()
        dst_nodes = torch.ones(SAMPLE_SIZE, device=device, dtype=torch.int32) * dst_node
#         dst_nodes=batch[1].to(device)
        node_attributes=batch[2].to(device)
#         y = batch[3].to(device)
        z = model.encode(node_attributes,edge_index)
        predictions=torch.round(torch.sigmoid(model.decode(z,src_nodes,dst_nodes))).view(-1)
        recommendations[dst_node] = {'src_nodes': src_nodes, 'confidence': predictions}
torch.save(recommendations, 'research/citation_recommendation/random_recommendations_ankit.pt')