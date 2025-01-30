import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import KFold

# data = pd.read_csv('../dataset/msa_nas.csv')
data = pd.read_csv('../dataset/msa_nas_ext.csv')
data = data.drop(columns=['Unnamed: 0'])

G = nx.DiGraph()
unique_values = data['nas-eps_nas_msg_emm_type_value'].unique()
G.add_nodes_from(unique_values)

for i in range(len(data) - 1):
    source_node = data.loc[i, 'nas-eps_nas_msg_emm_type_value']
    target_node = data.loc[i + 1, 'nas-eps_nas_msg_emm_type_value']
    edge_label = data.loc[i + 1, 'label']
    G.add_edge(source_node, target_node, label=edge_label)
    
adj_matrix = nx.adjacency_matrix(G)
adj_matrix = torch.FloatTensor(adj_matrix.toarray())

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.fillna(0)

features = torch.FloatTensor(data.drop('label', axis=1).values)
labels = torch.LongTensor(data['label'].values)

class GraphSAGEModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define the number of features and classes
num_features = features.shape[1]
num_classes = 9  # Number of label classes

# Define the KFold cross-validator
kf = KFold(n_splits=num_classes)

# Initialize lists to store accuracy for each fold
accuracies = []

# Perform LOLO-CV
for train_labels, test_labels in kf.split(unique_values):
    # Filter data for training and testing
    train_mask = torch.BoolTensor(data['label'].isin(train_labels).values)
    test_mask = torch.BoolTensor(data['label'].isin(test_labels).values)
    
    # Initialize the model
    model = GraphSAGEModel(num_features, num_classes)
    
    # Create a PyTorch Geometric data object
    data_obj = Data(x=features, edge_index=adj_matrix.nonzero().T, y=labels)
    data_obj.train_mask = train_mask
    data_obj.test_mask = test_mask
    
    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 100  # Set this value appropriately
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj.x, data_obj.edge_index)
        loss = criterion(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data_obj.x, data_obj.edge_index).argmax(dim=1)
        correct = pred[data_obj.test_mask].eq(data_obj.y[data_obj.test_mask]).sum().item()
        total = data_obj.test_mask.sum().item()
#         accuracy = correct / total
#         accuracies.append(accuracy)
        
        # Report predicted classes in tabular format
        test_indices = torch.arange(len(data))[data_obj.test_mask]
        test_data = data.iloc[test_indices].copy()
        test_data['predicted_label'] = pred[data_obj.test_mask].numpy()
        print(f"Test Fold: {test_labels}")
        print(test_data[['label', 'predicted_label']])
        print("="*30)
        
        # Summarize counts of predicted classes
        counts = test_data.groupby(['label', 'predicted_label']).size().unstack(fill_value=0)
        print("Class Counts:")
        print(counts)
        print("="*30)

# Calculate and print average accuracy
# avg_accuracy = np.mean(accuracies)
# print("Average Accuracy:", avg_accuracy)