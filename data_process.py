import random, math, os, sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from biopandas.pdb import PandasPdb
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from models.MvGraphDTA import SageConv

device = torch.device('cuda:0')

def remove_subgraph(Graph, center, percent):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes())*percent))
    removed = []
    temp = [center]
    while len(removed) < num and temp:
        neighbors = []
        try:
            for n in temp:
                neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        except Exception as e:
            print(e)
            return None, None
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed

def target_graph_augmentation(id, target_data, times, percent):
    center_node_list = random.sample(list(range(len(target_data[0]))), times)
    origin_features = target_data[0]
    origin_edge_features = target_data[1]
    origin_edges = target_data[2]
    targetGraph = nx.Graph(origin_edges)

    removed_list = []
    subGraph_list = []
    center_node_list = random.sample(list(range(len(target_data[0]))), times)
    while True:
        for i in range(times):
            subGraph, removed = remove_subgraph(targetGraph, center_node_list[i], percent)
            subGraph_list.append(subGraph)
            removed_list.append(removed)
        if nx.is_connected(subGraph_list[0]) and nx.is_connected(subGraph_list[1]):
            break
        else:
            removed_list = []
            subGraph_list = []
            center_node_list = random.sample(list(range(len(target_data[0]))), times)

    for removed_i in removed_list:
        if not removed_i:
            return None, None, None
    
    features_list = []
    for removed_i in removed_list:
        features = []
        for i in range(len(origin_features)):
            if i not in removed_i:
                features.append(origin_features[i])
        features_list.append(features)
    
    edges_feature_list = []
    for i, removed_i in enumerate(removed_list):
        edge_features = {}
        for key, value in origin_edge_features.items():
            if (eval(key)[0] not in removed_i) and (eval(key)[1] not in removed_i):
                edge_features[key] = value
        edges_feature_list.append(edge_features)

    edges_list = []
    new_edges_feature_list = []
    for i in range(times):
        edges_list.append([])
        new_edges_feature_list.append({})

    for i, removed_i in enumerate(removed_list):
        for e_1, e_2 in subGraph_list[i].edges():
            if e_1 not in removed_i and e_2 not in removed_i:
                if e_1 < e_2:
                    value = edges_feature_list[i][str([e_1, e_2])]
                else:
                    value = edges_feature_list[i][str([e_2, e_1])]
                e_start = e_1 - sum(num < e_1 for num in removed_i)
                e_end = e_2 - sum(num < e_2 for num in removed_i)
                if e_start < e_end:
                    new_edges_feature_list[i][str([e_start, e_end])] = value
                else:
                    new_edges_feature_list[i][str([e_end, e_start])] = value
                edges_list[i].append([e_start, e_end])
    
    edge_index_list = []
    for i in range(times):
        edge_index_list.append([])
    for i, edges_i in enumerate(edges_list):
        g_i = nx.Graph(edges_i).to_directed()
        for e_1, e_2 in g_i.edges():
            edge_index_list[i].append([e_1, e_2])

    for i in range(times):
        g = nx.Graph()
        node_list = [i for i in range(len(features_list[i]))]
        g.add_nodes_from(node_list)
        g.add_edges_from(edge_index_list[i])
        adj = nx.adjacency_matrix(g)
        edge_name, transfer_g = create_edge_adj(adj)
        edge_feature = edge_feature_transfer(new_edges_feature_list[i], edge_name)
        node_edge_index, node_edge_scatter_index, edge_node_index, edge_node_scatter_index = get_node_to_edge_index(node_list, edge_name)
        node_feature = torch.from_numpy(np.array(features_list[i])).to(torch.float)
        edge_feature = torch.from_numpy(np.array(list(new_edges_feature_list[i].values()))).to(torch.float)
        edge_index = torch.LongTensor(edge_index_list[i]).transpose(-1, 0)
        line_edge_index = torch.LongTensor(list(transfer_g.edges)).transpose(-1, 0)
        node_edge_index = torch.from_numpy(np.array(node_edge_index))
        edge_node_index = torch.from_numpy(np.array(edge_node_index))
        node_edge_scatter_index = torch.from_numpy(np.array(node_edge_scatter_index))
        edge_node_scatter_index = torch.from_numpy(np.array(edge_node_scatter_index))
        torch.save(node_feature, '../data_augment/target_node_feature/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(edge_feature, '../data_augment/target_edge_feature/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(edge_index, '../data_augment/target_edge_index/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(line_edge_index, '../data_augment/target_line_edge_index/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(node_edge_index, '../data_augment/target_node_edge_index/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(edge_node_index, '../data_augment/target_edge_node_index/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(node_edge_scatter_index, '../data_augment/target_node_edge_scatter_index/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')
        torch.save(edge_node_scatter_index, '../data_augment/target_edge_node_scatter_index/' + id.split('.')[0] + '_' + str(i + 1) + '.pt')

        # test whether the sample is suitable for a neural network
        model_node = SageConv(20, 2, 64)
        model_edge = SageConv(2, 20, 64)
        try:
            node_out = model_node(node_feature, edge_index, edge_feature, node_edge_index, node_edge_scatter_index)
            edge_out = model_edge(edge_feature, line_edge_index, node_feature, edge_node_index, edge_node_scatter_index)
        except:
            with open('../data_augment/augment_error.txt', 'a') as write:
                write.writelines(id + '\n')


# 定义氨基酸
Amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
# 定义缩写氨基酸
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
Amino_acids_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Amino_acids_dict = {}
for i in range(len(Amino_acids)):
    Amino_acids_dict[Amino_acids[i]] = Amino_acids_num[i]

def create_edge_adj(vertex_adj):
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [list(x) for x in zip(edge_index[0], edge_index[1])]
    edge_adj = np.zeros((num_edge, num_edge))
    for i in range(num_edge):
        for j in range(i, num_edge):
            if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                edge_adj[i, j] = 0
            else:
                edge_adj[i, j] = 1
    adj = edge_adj + edge_adj.T
    np.fill_diagonal(adj, 0)
    g = nx.from_numpy_array(adj).to_directed()
    return list(edge_name), g

def edge_feature_transfer(edge_feature_dict, edge_name):
    edge_feature = {}
    key_list, value_list = list(edge_feature_dict.keys()), list(edge_feature_dict.values())
    for i in range(len(edge_name)):
        edge_feature[str(edge_name[i])] = value_list[key_list.index(str(edge_name[i]))]

    return edge_feature

def get_node_to_edge_index(node_list, edge_name):
    node_edge_scatter_index = []
    edge_node_scatter_index = []
    node_edge_index = []
    edge_node_index = []
    for node in node_list:
        for edge in edge_name:
            if node == edge[0]:
                node_edge_index.append(edge_name.index(edge))
                node_edge_scatter_index.append(node)
            if node == edge[1]:
                node_edge_index.append(edge_name.index(edge))
                node_edge_scatter_index.append(node)
    for i in range(len(edge_name)):
        edge_node_index.append(edge_name[i][0])
        edge_node_index.append(edge_name[i][1])
        edge_node_scatter_index += [i] * 2
    return node_edge_index, node_edge_scatter_index, edge_node_index, edge_node_scatter_index

def getProteinProperties(path):
    edge_list = []
    coordinate_list = []
    pdb_pandas = PandasPdb().read_pdb(path)
    atom_name_list = pdb_pandas.df['ATOM'].atom_name.tolist()
    atom_coordinate_list = np.array(pdb_pandas.df['ATOM'].loc[:, ('x_coord', 'y_coord', 'z_coord')]).tolist()
    residue_name_list = pdb_pandas.df['ATOM'].residue_name.tolist()
    node_features = []
    edge_features = {}
    protein_sequence = ''
    num_residue = 0
    for i in range(len(atom_name_list)):
        if atom_name_list[i] == 'CA':
            num_residue += 1
            protein_sequence += amino_acids[Amino_acids_dict[residue_name_list[i]]]
            coordinate_list.append(atom_coordinate_list[i])
    for i in protein_sequence:
        feature = [0 for i in range(len(Amino_acids))]
        feature[amino_acids.index(i)] = 1
        node_features.append(feature)
    reference_point = coordinate_list[0]
    for i in range(len(coordinate_list)):
        for j in range(len(coordinate_list)):
            dist = math.sqrt((coordinate_list[i][0] - coordinate_list[j][0]) ** 2 + (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + (coordinate_list[i][2] - coordinate_list[j][2]) ** 2)
            if i != j and dist <= 8:
                edge_list.append([i, j])
                x1, x2 = np.array([coordinate_list[i][k] - reference_point[k] for k in range(len(coordinate_list[i]))]), np.array([coordinate_list[j][k] - reference_point[k] for k in range(len(coordinate_list[j]))])
                vec_x1, vec_x2 = x1 - reference_point, x2 - reference_point
                norm_x1, norm_x2 = np.linalg.norm(vec_x1), np.linalg.norm(vec_x2)
                cos_value = np.dot(vec_x1, vec_x2) / (norm_x1 * norm_x2)
                edge_features[str([i, j])] = [dist, cos_value]
    return node_features, edge_features, edge_list

if __name__ == '__main__':
    base_path = 'pdbbind2019/'
    general_path = base_path + 'general-set/'
    refined_path = base_path + 'refined-set/'
    general_set = os.listdir(general_path)
    refined_set = os.listdir(refined_path)

    id_list = os.listdir(base_path)
    # all_id_list = pd.read_csv('GraphscoreDTA/test_set/labels_train13851.csv')['PDBID'].to_numpy().tolist()
    id_list = os.listdir('/home/user/zky/rwrSage/pdbbind2019/case_study/CSAR-HiQ_36/global/')


    for id in set(id_list):
        exist_list = '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_node_feature/'
        if id.split('.')[0] + '.pt' in exist_list:
            pass
        if id in general_set:
            path = general_path
        else:
            path = refined_path
        # path = '/home/user/zky/multimodal/data/protein_pdb/'
        path = '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/'
        node_feature, edge_feature, edge_list = getProteinProperties(path + id.split('.')[0] + '.pdb')
        G = nx.Graph()
        node_list = [i for i in range(len(node_feature))]
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)
        adj = nx.adjacency_matrix(G)
        edge_name, g = create_edge_adj(adj)
        edge_feature = edge_feature_transfer(edge_feature, edge_name)
        node_edge_index, node_edge_scatter_index, edge_node_index, edge_node_scatter_index = get_node_to_edge_index(node_list, edge_name)
        target_data = node_feature, edge_feature, edge_list

        # data augment, execute the code if genenrate training dataset else not execute the code
        target_graph_augmentation(id, target_data, times=2, percent=0.1)

        node_feature = torch.from_numpy(np.array(node_feature)).to(torch.float)
        edge_feature = torch.from_numpy(np.array(list(edge_feature.values()))).to(torch.float)
        edge_index = torch.LongTensor(edge_list).transpose(-1, 0)
        line_edge_index = torch.LongTensor(list(g.edges)).transpose(-1, 0)
        node_edge_index = torch.from_numpy(np.array(node_edge_index))
        edge_node_index = torch.from_numpy(np.array(edge_node_index))
        node_edge_scatter_index = torch.from_numpy(np.array(node_edge_scatter_index))
        edge_node_scatter_index = torch.from_numpy(np.array(edge_node_scatter_index))
        torch.save(node_feature, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_node_feature/' + id.split('.')[0] + '.pt')
        torch.save(edge_feature, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_edge_feature/' + id.split('.')[0] + '.pt')
        torch.save(edge_index, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_edge_index/' + id.split('.')[0] + '.pt')
        torch.save(line_edge_index, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_line_edge_index/' + id.split('.')[0] + '.pt')
        torch.save(node_edge_index, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_node_edge_index/' + id.split('.')[0] + '.pt')
        torch.save(edge_node_index, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_edge_node_index/' + id.split('.')[0] + '.pt')
        torch.save(node_edge_scatter_index, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_node_edge_scatter_index/' + id.split('.')[0] + '.pt')
        torch.save(edge_node_scatter_index, '/home/user/zky/rwrSage/pdbbind2019/CSAR_HiQ_36/data/target_edge_node_scatter_index/' + id.split('.')[0] + '.pt')