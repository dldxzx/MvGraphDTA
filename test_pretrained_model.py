import torch
import torch.nn as nn
from models.MvGraphDTA import *
from evaluate_metrics import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def load_data(id, data_set='test'):
    path = '/home/user/zky/MvGraphDTA/data/binding_affinity/PDBbindv2016/core_set/'
    # path = '/home/user/zky/MvGraphDTA/data/binding_affinity/li_data/casf2013/'

    # print(id[0])
    drug_node_feature = torch.load(path + 'drug_node_feature/' + id[0] + '.pt').to(device)
    drug_edge_feature = torch.load(path + 'drug_edge_feature/' + id[0] + '.pt').to(device)
    drug_edge_index = torch.load(path + 'drug_edge_index/' + id[0] + '.pt').to(device)
    drug_line_edge_index = torch.load(path + 'drug_line_edge_index/' + id[0] + '.pt').to(device)
    drug_node_edge_index = torch.load(path + 'drug_node_edge_index/' + id[0] + '.pt').to(device)
    drug_edge_node_index = torch.load(path + 'drug_edge_node_index/' + id[0] + '.pt').to(device)
    drug_node_edge_scatter_index = torch.load(path + 'drug_node_edge_scatter_index/' + id[0] + '.pt').to(device)
    drug_edge_node_scatter_index = torch.load(path + 'drug_edge_node_scatter_index/' + id[0] + '.pt').to(device)
    
    target_node_feature = torch.load(path + 'target_node_feature/' + id[0] + '.pt').to(device)
    target_edge_feature = torch.load(path + 'target_edge_feature/' + id[0] + '.pt').to(device)
    target_edge_index = torch.load(path + 'target_edge_index/' + id[0] + '.pt').to(device)
    target_line_edge_index = torch.load(path + 'target_line_edge_index/' + id[0] + '.pt').to(device)
    target_node_edge_index = torch.load(path + 'target_node_edge_index/' + id[0] + '.pt').to(device)
    target_edge_node_index = torch.load(path + 'target_edge_node_index/' + id[0] + '.pt').to(device)
    target_node_edge_scatter_index = torch.load(path + 'target_node_edge_scatter_index/' + id[0] + '.pt').to(device)
    target_edge_node_scatter_index = torch.load(path + 'target_edge_node_scatter_index/' + id[0] + '.pt').to(device)
            
    drug_data = drug_node_feature, drug_edge_index, drug_edge_feature, drug_line_edge_index, drug_node_edge_index, drug_edge_node_index, drug_node_edge_scatter_index, drug_edge_node_scatter_index
    
    target_data = target_node_feature, target_edge_index, target_edge_feature, target_line_edge_index, target_node_edge_index, target_edge_node_index, target_node_edge_scatter_index, target_edge_node_scatter_index

    return drug_data, target_data

def validation(model, loader, epoch=1, epochs=1, data_set='test'):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch, (id, affinity) in enumerate(loader):
            # if id[0] not in error_list:
            if data_set == 'test':
                drug_data, target_data = load_data(id)
            else:
                drug_data, target_data = load_data(id, data_set='case_study')
            output = model(drug_data, target_data)
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, affinity.view(-1, 1).cpu()), 0)
    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    return total_labels, total_preds


if __name__ == '__main__':
    device = torch.device('cuda:0')
    drug_n_dim = 29
    drug_v_dim = 5
    target_n_dim = 20
    target_v_dim = 2
    emb_dim = 128
    out_dim = 256
    n_class = 1
    batch_size = 1
    num_layers = 2
    hidden_dim = [128, 128]
    test_data = dataset('/home/user/zky/MvGraphDTA/data/binding_affinity/PDBbindv2016/testing.csv')
    # test_data = dataset('/home/user/zky/MvGraphDTA/data/binding_affinity/li_data/casf2013.csv')

    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=32, shuffle=False)

    model = PredicterDTA(drug_n_dim, drug_v_dim ,target_n_dim, target_v_dim, hidden_dim, out_dim, num_layers, n_class).to(device).eval()

    # model.load_state_dict(torch.load('/home/user/zky/MvGraphDTA/data/best_model/MvGraphDTA.pt'))
    model.load_state_dict(torch.load('/home/user/zky/MvGraphDTA/data/best_model/MvGraphDTA_Drug_Similarity.pt'))
    test_labels, test_preds = validation(model, test_loader, data_set='test')
    test_result = [mae(test_labels, test_preds), rmse(test_labels, test_preds), pearson(test_labels, test_preds), spearman(test_labels, test_preds), ci(test_labels, test_preds), r_squared(test_labels, test_preds)]

    print(test_result)