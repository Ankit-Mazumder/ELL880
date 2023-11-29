# %%
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
matplotlib.use('Agg')


#from .DiGCNConv import DiGCNConv

# %%
df = pd.read_csv('/home/scai/msr/aiy227513/scratch/research/superSetAnalysis.csv')
#df.head()

# %%
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

# %% [markdown]
# Skip if edges.pickle file already exits

# %%
# edges=[]
# for i in range(df.shape[0]):
#     a = ast.literal_eval(df["citations"][i])
#     for j in range(len(a)):
#         if a[j]['paperId'] in node_ids:edges.append([node_ids.index(df["paperId"][i]),node_ids.index(a[j]['paperId']),1])


# %%
# edges2=np.array(edges)

# # %%
# #negative sampling
# count=len(edges)
# while(count!=0):
#     arr=np.random.randint(0,len(node_ids),size=2)
#     if arr[0]!=arr[1]:
#         a = ast.literal_eval(df["citations"][arr[0]])
#         flag=1
#         for j in range(len(a)):
#          if node_ids[arr[1]] in a[j]['paperId']:
#                flag=0
#                break
#         if flag==1:
#                f=[arr[0],arr[1],0]
#                edges.append(f)
#                count-=1
           

# # %%
# # import pickle
# # with open("edge", "wb") as fp:   #Pickling
# #   pickle.dump(edges, fp)
# # with open("edge", "rb") as fp:   # Unpickling
# #    edges2 = pickle.load(fp)

# # %%
# random.shuffle(edges)

# # %%
# #import pickle

# #my_list = [1, 2, 3, 4, 5]

# with open("/home/scai/msr/aiy227513/scratch/research/edges_same_ratio.pickle", "wb") as file:
#     pickle.dump(edges, file)
#     file.close()


# %% [markdown]
# Start here if edges.pickle file already exits

# %%
with open("/home/scai/msr/aiy227513/scratch/research/edges_same_ratio.pickle", "rb") as file:
    edges = pickle.load(file)
    file.close()

#print(loaded_list)

# %%
#preparing train set
count=len(edges)
train_size=math.floor((len(edges)*0.8))
train_src=[]
train_dest=[]
train_label=[]
for i in range(train_size):
 if edges[i][2]==1:
    train_src.append(edges[i][0])
    train_dest.append(edges[i][1])
 #train_label.append([edges[i][2]])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("/home/scai/msr/aiy227513/scratch/research/data/bert_encode.pkl", "rb") as file:
    embedding = pickle.load(file)
    file.close()
node_features = torch.tensor(node_features, dtype=torch.float)
embedding=torch.tensor(embedding, dtype=torch.float)
feature=torch.cat([node_features,embedding], dim=1).float()
feature =F.normalize(feature,p=2, dim=-1)

# %%

#device='cpu'

# %% [markdown]
# If error occurs while training, run from below.

# %%
import gc
torch.cuda.empty_cache()
gc.collect()

# %%
#Train
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



# Forward pass through the model
# output = model(x, edge_index)

# # Extract node embedding of a particular node (e.g., node 2)
# node_index = 2
# node_embedding = output[node_index]

# print("Node embedding for node", node_index, ":", node_embedding)


# %%
# Define the GCN model
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
    
    # def __init__(self, num_features, hidden_size):
    #     super(GCN, self).__init__()
    #     self.conv1 = GCNConv(num_features, hidden_size)
    #     self.conv2 = GCNConv(hidden_size, hidden_size)
    #     self.fc1 = Linear(hidden_size*2,1)

    # def forward(self,src_nodes, dst_nodes,x, edge_index):
    #     x = self.conv1(x, edge_index)
    #     x = torch.relu(x)
    #     x = self.conv2(x, edge_index)
    #     x=torch.relu(x)
    #     #print(torch.cat([x[1][src_nodes],x[1][dst_nodes]], dim=1))
    #     #z=torch.cat([x[0][src_nodes],x[0][dst_nodes]], dim=1)
    #     #print(x[src_nodes])
    #     #print(x.shape)
    #     x=self.fc1(torch.cat((x[0][src_nodes],x[0][dst_nodes]), dim=-1))
    #     #print(x.shape)
    #     #print(x.shape)
    #     #print(torch.sigmoid(x))
    #     return torch.sigmoid(x)

# Create an example graph
#num_nodes = 5


# %%
# num_features =len(node_features[0])
# hidden_size = 16

#node_ids=[0,1,2,3,4]
#edge=[(0,1),(1,0),(1,2),(2,1),(3,4)]
#feature = torch.randn(num_nodes, num_features)

#y = torch.tensor(train_label, dtype=torch.float)

edge_index = torch.tensor([train_src,train_dest], dtype=torch.long).to(device)

dataset = LinkPredictionDataset(node_ids,feature,edges[:train_size])
dataloader = DataLoader(dataset, batch_size=2**7, shuffle=True)


# Create the GCN model
model = GCN()
if torch.cuda.device_count() > 1:model = nn.DataParallel(model)
#model = torch.load('scratch/research/model.pth') #if already created model
learning_rate = 0.001
#optimizer = optim.Adam(params=model.parameters(), lr=learning_rate,weight_decay=0.001)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adadelta(model.parameters(),learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
model = model.to(device)
criteria = torch.nn.BCEWithLogitsLoss() #_with_logits
scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0, last_epoch=- 1, verbose=True)
# %%
losses=[]

# %%
#Train model

start_time = time.time()
epoch=0
while(1):
    model.train()
    optimizer.zero_grad()
    #count = 0
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        #src_nodes, dst_nodes,node_attributes,y = batch
        src_nodes=batch[0].to(device)
        dst_nodes=batch[1].to(device)
        node_attributes=batch[2].to(device)
        y = batch[3].to(device)
        
        z = model.module.encode(node_attributes, edge_index)
        predictions = model.module.decode(z, src_nodes, dst_nodes).view(-1)

        # z =  model.modulemodel.encode(node_attributes,edge_index)
        # predictions=model.decode(z,src_nodes,dst_nodes).view(-1)
        #print(predictions.shape)
        #print(predictions)
        #predictions = predictions.view(-1)
        #print(predictions.shape)
        #print(predictions)
        #print(y)
        #torch.reshape(y,(1,1024))  
        # y= torch.unsqueeze(y.float(), 1)     
        loss = criteria(predictions,y.float())
        loss.backward()
        max_norm = 1.0  # Define your desired maximum norm value
        torch.nn.utils.clip_grad_value_(model.parameters(), max_norm)
        optimizer.step()
        #print(loss.item())
        total_loss += loss.item()

        #count+=1
    #print(loss.item())
    scheduler.step()
    print("Epoch:",epoch+1)
    average_loss = total_loss / (batch_idx+1)
    losses.append(average_loss)
    print("Avg loss:",average_loss)
    torch.save(model, '/home/scai/msr/aiy227513/scratch/research/model/model_pratik.pt')
#     torch.save(model.state_dict(), '/home/scai/msr/aiy227513/scratch/research/model/model_bert.pth')
    epoch+=1    
end_time = time.time()

# Calculate the elapsed time
execution_time = end_time - start_time

# Print the execution time
print("Execution time:", execution_time/60, "mins")
# %%
plt.plot([i for i in range(len(losses[1:]))], losses[1:])
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('scratch/research/plots/adadelta_nlp_sage_loss.jpeg')
# torch.save(model.state_dict(), '/home/scai/msr/aiy227513/scratch/research/model/model.pth')
torch.save(model, '/home/scai/msr/aiy227513/scratch/research/model/model_pratik.pt')

# %%
print(model)

# %%
# #preparing test set
# count=len(edges)
# test_size=count-train_size
# test_src=[]
# test_dest=[]
# test_label=[]
# for i in range(train_size,count):
#     test_src.append(edges[i][0])
#     test_dest.append(edges[i][1])
#     #test_label.append([edges[i][2]])

# %%
#Saving the model
#torch.save(model, 'scratch/research/model.pth')

# %%
#Test model
model.eval()
g_truth=[]
prediction=[]
count=0
with torch.no_grad():
    #y = torch.tensor(test_label, dtype=torch.float)
    #edge_index = torch.tensor([test_src,test_dest], dtype=torch.long).to(device)
    test_dataset = LinkPredictionDataset(node_ids,feature,edges[train_size:])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch_idx, batch in enumerate(test_dataloader):
        #src_nodes, dst_nodes,node_attributes,y = batch
        src_nodes=batch[0].to(device)
        dst_nodes=batch[1].to(device)
        node_attributes=batch[2].to(device)
        y = batch[3].to(device)
        z = model.encode(node_attributes,edge_index)
        predictions=torch.round(torch.sigmoid(model.decode(z,src_nodes,dst_nodes))).view(-1)
        #print(predictions)
        if predictions==y.float():count+=1
        g_truth.append(y.item())
        prediction.append(predictions.item())
        #print('Predictions:', predictions)
print("Accuracy:",(count/(batch_idx+1))*100,"%") 
        

# %%
# ground_truth_np = np.array(g_truth)
# predicted_np = np.array(prediction)

# Calculate the confusion matrix
cm = confusion_matrix(np.array(g_truth),np.array(prediction))

#print(cm)
labels = ['Class 0', 'Class 1']
cmap = plt.cm.Blues

# Create figure and axes
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Add text inside each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

# Set plot labels and title
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")

# Add a color bar
plt.colorbar(im)

# Show the plot
#plt.show()
plt.savefig('scratch/research/plots/adadelta_nlp_sage_cnf_matrix.jpeg')
f1 = f1_score(np.array(g_truth),np.array(prediction))

print("F1 score:", f1)




