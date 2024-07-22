import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.MvGraphDTA import *
import prettytable as pt
from sklearn.model_selection import KFold
from utils import *
from evaluate_metrics import *

writer = SummaryWriter()

def load_data(id):
    path = '/home/user/zky/MvGraphDTA/data/binding_affinity/PDBbindv2016/training_set/'
    target_node_feature = torch.load(path + 'target_node_feature/' + id[0] + '.pt').to(device)
    target_edge_feature = torch.load(path + 'target_edge_feature/' + id[0] + '.pt').to(device)
    target_edge_index = torch.load(path + 'target_edge_index/' + id[0] + '.pt').to(device)
    target_line_edge_index = torch.load(path + 'target_line_edge_index/' + id[0] + '.pt').to(device)
    target_node_edge_index = torch.load(path + 'target_node_edge_index/' + id[0] + '.pt').to(device)
    target_edge_node_index = torch.load(path + 'target_edge_node_index/' + id[0] + '.pt').to(device)
    target_node_edge_scatter_index = torch.load(path + 'target_node_edge_scatter_index/' + id[0] + '.pt').to(device)
    target_edge_node_scatter_index = torch.load(path + 'target_edge_node_scatter_index/' + id[0] + '.pt').to(device)

    drug_node_feature = torch.load(path + 'drug_node_feature/' + id[0] + '.pt').to(device)
    drug_edge_feature = torch.load(path + 'drug_edge_feature/' + id[0] + '.pt').to(device)
    drug_edge_index = torch.load(path + 'drug_edge_index/' + id[0] + '.pt').to(device)
    drug_line_edge_index = torch.load(path + 'drug_line_edge_index/' + id[0] + '.pt').to(device)
    drug_node_edge_index = torch.load(path + 'drug_node_edge_index/' + id[0] + '.pt').to(device)
    drug_edge_node_index = torch.load(path + 'drug_edge_node_index/' + id[0] + '.pt').to(device)
    drug_node_edge_scatter_index = torch.load(path + 'drug_node_edge_scatter_index/' + id[0] + '.pt').to(device)
    drug_edge_node_scatter_index = torch.load(path + 'drug_edge_node_scatter_index/' + id[0] + '.pt').to(device)

    drug_data = drug_node_feature, drug_edge_index, drug_edge_feature, drug_line_edge_index, drug_node_edge_index, drug_edge_node_index, drug_node_edge_scatter_index, drug_edge_node_scatter_index
    
    target_data = target_node_feature, target_edge_index, target_edge_feature, target_line_edge_index, target_node_edge_index, target_edge_node_index, target_node_edge_scatter_index, target_edge_node_scatter_index


    return drug_data, target_data

def training(model, train_loader, optimizer, epoch, epochs):
    model.train()
    # loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red')
    training_loss = 0.0
    for batch, (id, affinity) in enumerate(train_loader):
        drug_data, target_data = load_data(id)
        output = model(drug_data, target_data)
        loss = criterion(output, affinity.view(-1, 1).to(torch.float).to(device))
            # print('pred: {}, label: {}, loss: {}'.format(output, affinity, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        # print(loss.item())
            # loop.set_description(f'Training Epoch [{epoch} / {epochs}]')
        # loop.set_description(f'Fold [{i + 1} / 5]')
        # loop.set_postfix(loss=loss.item())
    writer.add_scalar('Training loss', training_loss, epoch)
    print('Epoch:[{} / {}], Fold:[{} / {}], Mean Loss: {}'.format(epoch, epochs, i + 1, 5, training_loss / 10394))
    # print('Training Epoch:[{} / {}], Mean Loss: {}'.format(epoch, epochs, training_loss / 12993))


def validation(model, loader, epoch=1, epochs=1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        # loop = tqdm(enumerate(loader), total=len(loader), colour='blue')
        for batch, (id, affinity) in enumerate(loader):
            # if id[0] not in error_list:
            drug_data, target_data = load_data(id)
            output = model(drug_data, target_data)
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, affinity.view(-1, 1).cpu()), 0)
    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    return total_labels, total_preds



if __name__ == '__main__':
    device = torch.device('cuda:0')
    drug_n_dim = 27
    drug_v_dim = 5
    target_n_dim = 20
    target_v_dim = 2
    emb_dim = 128
    out_dim = 256
    n_class = 1
    batch_size = 1
    hidden_dim = [128, 128, 128]
    num_layers = 3
    epochs = 100
    best_rmse = [1000, 1000, 1000, 1000, 1000]
    model = PredicterDTA(drug_n_dim, drug_v_dim, target_n_dim, target_v_dim, hidden_dim, out_dim, num_layers, n_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss().to(device)
    train_data = dataset('/home/user/zky/MvGraphDTA/data/binding_affinity/PDBbindv2016/training.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=32, shuffle=True)
    print(len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
    for epoch in range(1, epochs + 1):
        print('Cross Validation [{} / {}]'.format(epoch, epochs))
        for i, (train_index, val_index) in enumerate(kfold.split(train_data)):
            compound_train_loader = DataLoader(torch.utils.data.dataset.Subset(train_data, train_index), batch_size=batch_size, shuffle=True)
            compound_val_loader = DataLoader(torch.utils.data.dataset.Subset(train_data, val_index), batch_size=batch_size, shuffle=True)
            training(model, compound_train_loader, optimizer, epoch, epochs)
            val_labels, val_preds = validation(model, compound_val_loader, epoch, epochs)
            val_result = [mae(val_labels, val_preds), rmse(val_labels, val_preds), pearson(val_labels, val_preds), spearman(val_labels, val_preds), r_squared(val_labels, val_preds)]
            tb = pt.PrettyTable()
            tb.field_names = ['Epoch / Epochs', 'LR', 'Fold / Folds', 'Set', 'MAE', 'RMSE', 'Pearson', 'Spearman', 'R-Squared']
            tb.add_row(['{} / {}'.format(epoch, epochs), scheduler.get_last_lr(), '{} / {}'.format(i + 1, 5), 'Validation', val_result[0], val_result[1], val_result[2], val_result[3], val_result[-1]])
            print(tb)
            writer.add_scalar('RMSE/Val RMSE', val_result[1], epoch)
            with open('data/result/kfold/augment/result_' + str(i + 1) + '.txt', 'a') as write:
                write.writelines(str(tb) + '\n')
            if float(val_result[1]) < best_rmse[i]:
                best_rmse[i] = float(val_result[1])
                torch.save(model.state_dict(), 'data/best_model/MvGraphDTA_' + str(i + 1) + '.pt')
        scheduler.step()
