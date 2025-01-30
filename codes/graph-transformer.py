#!/usr/bin/env python
# coding: utf-8

# Data Preparation
import pandas as pd
# Load and parse network traffic data into a dataframe
data = pd.read_csv('../dataset/msa_nas.csv')
# data = pd.read_csv('../dataset/msa_rrc.csv')

# data = data.drop(columns=['Unnamed: 0'])
# Ensure  dataframe contains necessary features
# 'nas_eps_nas_msg_emm_type_value' and 'label'
# Print the first few rows of  dataframe
data.head()
# Graph Construction
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Identify unique values in the 'nas_eps_nas_msg_emm_type_value' feature
unique_values = data['nas_eps_nas_msg_emm_type_value'].unique()
# unique_values = data['lte-rrc_c1_showname'].unique()

# Add nodes for each unique value
G.add_nodes_from(unique_values)

# Add edges between nodes based on dataframe rows
for i in range(len(data) - 1):
    source_node = data.loc[i, 'nas_eps_nas_msg_emm_type_value']
    target_node = data.loc[i + 1, 'nas_eps_nas_msg_emm_type_value']
    # source_node = data.loc[i, 'lte-rrc_c1_showname']
    # target_node = data.loc[i + 1, 'lte-rrc_c1_showname']
    edge_label = data.loc[i + 1, 'label']
    G.add_edge(source_node, target_node, label=edge_label)

# Data Preprocessing
import torch
import numpy as np

# Convert the graph to PyTorch tensors
adj_matrix = nx.adjacency_matrix(G)
adj_matrix = torch.FloatTensor(adj_matrix.toarray())

# Preprocess  data for training, validation, and testing
# Split  data into train, validation, and test sets

# Convert  features and labels into PyTorch tensors
features = torch.FloatTensor(data.drop('label', axis=1).values)
labels = torch.LongTensor(data['label'].values)


# Model Implementation (Graph Transformer)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphTransformerModel(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_heads):
        super(GraphTransformerModel, self).__init__()
        self.conv1 = TransformerConv(num_features, num_classes, heads=num_heads)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create the model
num_nodes = len(unique_values)
num_features = features.shape[1]
num_classes = 5  # Number of label classes
num_heads = 2    # Number of attention heads
model = GraphTransformerModel(num_nodes, num_features, num_classes, num_heads)


# Training and Evaluation

import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Split  data into train, validation, and test sets
train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
train_mask = torch.BoolTensor(np.isin(np.arange(len(data)), train_idx))
test_mask = ~train_mask

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
    accuracy = correct / total
    print("Accuracy:", accuracy)


# Performance Metrics Calculation
from sklearn.metrics import multilabel_confusion_matrix

# Get the multi-label confusion matrix
mcm = multilabel_confusion_matrix(data_obj.y[data_obj.test_mask].numpy(), pred[data_obj.test_mask].numpy())

# Calculate False Positive Rate (FPR) and False Negative Rate (FNR) for each class
fprs = []
fnrs = []

for i in range(num_classes):
    tn, fp, fn, tp = mcm[i].ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    fprs.append(fpr)
    fnrs.append(fnr)

print("False Positive Rates:", fprs)
print("False Negative Rates:", fnrs)