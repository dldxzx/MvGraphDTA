import numpy as np
import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_scatter import scatter
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

device = torch.device('cuda:0')

def L2Norm(feature):
    feature = F.normalize(feature, p=2, dim=1)
    return feature

class SageConv(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, out_channels, activation=F.leaky_relu, normalize=True, bias=False):
        super(SageConv, self).__init__()
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.activation = activation

        self.lin_center_node = nn.Linear(node_in_channels, out_channels)
        self.lin_neighbor_node = nn.Linear(node_in_channels, out_channels)
        self.lin_neighbor_edge = nn.Linear(edge_in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_center_node.reset_parameters()
        self.lin_neighbor_node.reset_parameters()
        self.lin_neighbor_edge.reset_parameters()
    
    def forward(self, node_x, edge_index, edge_x, node_edge_index, node_edge_scatter_index):
        if edge_index.shape == torch.Size([0]):
            out =  self.lin_center_node(node_x)
        else:
            aggr, aggr_1 = self.propagate(node_x, edge_index, edge_x, node_edge_index, node_edge_scatter_index)
            out = self.lin_center_node(node_x) + self.lin_neighbor_node(aggr) + self.lin_neighbor_edge(aggr_1)
            if self.normalize:
                out = F.normalize(out, p=2)
            if self.activation:
                out = self.activation(out)
        return out
    
    def propagate(self, node_x, edge_index, edge_x, node_edge_index, node_edge_scatter_index):
        out, out_1 = self.message(node_x, edge_index, edge_x, node_edge_index)
        out, out_1 = self.aggregate(out, edge_index, out_1, node_edge_scatter_index)
        out = self.update(out, out_1)
        return out

    def message(self, node_x, edge_index, edge_x, node_edge_index):
        row, col = edge_index
        x_j = node_x[row]
        x_j_1 = edge_x[node_edge_index]
        return x_j, x_j_1
    
    def aggregate(self, x_j, edge_index, x_j_1, scatter_index):
        row, col = edge_index
        node_aggr_node = scatter(x_j, col, dim=-2, reduce='sum')
        node_aggr_edge = scatter(x_j_1, scatter_index, dim=-2, reduce='sum')
        return node_aggr_node, node_aggr_edge
        
    def update(self, aggr_out, aggr_out_1):
        return aggr_out, aggr_out_1
        
class GraphSage(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim, num_layers):
        super(GraphSage, self).__init__()
        # self.hideen_dim = hidden_dim
        self.num_layers = num_layers
        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        self.node_linear = nn.Linear(hidden_dim[-1], output_dim)
        self.edge_linear = nn.Linear(hidden_dim[-1], output_dim)
        for i in range(num_layers):
            if i == 0:
                self.node_layers.append(SageConv(node_input_dim, edge_input_dim, hidden_dim[i]))
                self.edge_layers.append(SageConv(edge_input_dim, node_input_dim, hidden_dim[i]))
            elif i == num_layers - 1:
                self.node_layers.append(SageConv(hidden_dim[i - 1], hidden_dim[i - 1], hidden_dim[i], activation=None))
                self.edge_layers.append(SageConv(hidden_dim[i - 1], hidden_dim[i - 1], hidden_dim[i], activation=None))
            else:
                self.node_layers.append(SageConv(hidden_dim[i - 1], hidden_dim[i - 1], hidden_dim[i]))
                self.edge_layers.append(SageConv(hidden_dim[i - 1], hidden_dim[i - 1], hidden_dim[i]))

    def forward(self, node_feature, edge_index, edge_feature, line_edge_index, node_edge_idnex, 
                edge_node_index, node_edge_scatter_index, edge_node_scatter_index):
        node_batch = torch.from_numpy(np.array([0] * node_feature.shape[0])).to(device)
        edge_batch = torch.from_numpy(np.array([0] * edge_feature.shape[0])).to(device)
        src_node_feature = node_feature
        src_edge_feature = edge_feature
        for i in range(self.num_layers):
            temp_node_feature = node_feature
            if edge_index.shape == torch.Size([0]):
                node_feature = self.node_layers[i](node_feature, edge_index, edge_feature, node_edge_idnex, node_edge_scatter_index)
            else:
                node_feature = self.node_layers[i](node_feature, edge_index, edge_feature, node_edge_idnex, node_edge_scatter_index)
                edge_feature = self.edge_layers[i](edge_feature, line_edge_index, temp_node_feature, edge_node_index, edge_node_scatter_index)

        # residue
        # node_feature = node_feature + 0.3 * src_node_feature
        # edge_feature = edge_feature + 0.3 * src_edge_feature

        node_feature = self.node_linear(node_feature)
        edge_feature = self.edge_linear(edge_feature)
        temp_node_feature = node_feature
        temp_edge_feature = edge_feature
        # pooling
        node_feature = gmp(node_feature, node_batch)
        edge_feature = gmp(edge_feature, edge_batch)
        
        node_feature = node_feature + edge_feature

        return node_feature, temp_node_feature, temp_edge_feature

class PredicterDTA(nn.Module):
    def __init__(self, drug_dim_node, drug_dim_edge, target_dim_node, target_dim_edge, hidden_dim, out_dim, num_layers, n_class):
        super(PredicterDTA, self).__init__()
        self.embed_dim = 128
        self.drug_node_embed = nn.Linear(drug_dim_node, self.embed_dim)
        self.drug_edge_embed = nn.Linear(drug_dim_edge, self.embed_dim)
        self.target_node_embed = nn.Linear(target_dim_node, self.embed_dim)
        self.target_edge_embed = nn.Linear(target_dim_edge, self.embed_dim)

        self.drug_feature_extraction = GraphSage(self.embed_dim, self.embed_dim, hidden_dim, out_dim, num_layers)
        self.target_feature_extraction = GraphSage(self.embed_dim, self.embed_dim, hidden_dim, out_dim, num_layers)

        self.fully_connected = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
             nn.Linear(128, n_class)
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        self.drug_node_embed.reset_parameters()
        self.drug_edge_embed.reset_parameters()
        self.target_node_embed.reset_parameters()
        self.target_edge_embed.reset_parameters()

        for m in self.fully_connected:
            if isinstance(m, nn.Linear):
                m.reset_parameters()
    
    def forward(self, drug_data, target_data):
        drug_node_feature, drug_edge_index, drug_edge_feature, drug_line_edge_index, drug_node_edge_idnex, drug_edge_node_index, drug_node_edge_scatter_index, drug_edge_node_scatter_index = drug_data
        target_node_feature, target_edge_index, target_edge_feature, target_line_edge_index, target_node_edge_idnex, target_edge_node_index, target_node_edge_scatter_index, target_edge_node_scatter_index = target_data
        
        # Feature Normalization
        target_edge_feature = L2Norm(target_edge_feature)

        drug_node_feature = self.drug_node_embed(drug_node_feature)
        drug_edge_feature = self.drug_edge_embed(drug_edge_feature)
        target_node_feature = self.target_node_embed(target_node_feature)
        target_edge_feature = self.target_edge_embed(target_edge_feature)

        drug_feature, drug_node, drug_edge = self.drug_feature_extraction(drug_node_feature, drug_edge_index, drug_edge_feature, drug_line_edge_index, drug_node_edge_idnex, drug_edge_node_index, drug_node_edge_scatter_index, drug_edge_node_scatter_index)
        target_feature, target_node, target_edge = self.target_feature_extraction(target_node_feature, target_edge_index, target_edge_feature, target_line_edge_index, target_node_edge_idnex, target_edge_node_index, target_node_edge_scatter_index, target_edge_node_scatter_index)

        fusion_x = torch.cat((drug_feature, target_feature), dim=1)

        x = self.fully_connected(fusion_x)

        return x