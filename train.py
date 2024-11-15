import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test import calculate_relative_error, calculate_structural_similarity, calculate_topology_accuracy

def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    total_mse = 0
    total_mse_per_area = 0
    total_hidden_ratio = 0
    total_topology_acc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        if hasattr(data, 'edge_index'):
            data.edge_index = data.edge_index.to(device)
        optimizer.zero_grad()
        output, mask = model(data)
        target = data.y
        loss = criterion(output[mask], target[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mse = torch.mean((output[mask] - target[mask]) ** 2).item()
        total_mse += mse
        total_mse_per_area += mse / (930 * 930)
        hidden_ratio = 1 - mask.float().mean().item()
        total_hidden_ratio += hidden_ratio
        
        topology_acc = calculate_topology_accuracy(output, target, data.edge_index)
        total_topology_acc += topology_acc

    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_mse_per_area = total_mse_per_area / len(train_loader)
    avg_hidden_ratio = total_hidden_ratio / len(train_loader)
    avg_topology_acc = total_topology_acc / len(train_loader)

    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/MSE', avg_mse, epoch)
    writer.add_scalar('Train/MSE_per_Area', avg_mse_per_area, epoch)
    writer.add_scalar('Train/Hidden_Ratio', avg_hidden_ratio, epoch)
    writer.add_scalar('Train/Topology_Accuracy', avg_topology_acc, epoch)

    return avg_loss, avg_mse, avg_mse_per_area, avg_hidden_ratio, avg_topology_acc

def validate(model, val_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mse_per_area = 0
    total_hidden_ratio = 0
    total_topology_acc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            if hasattr(data, 'edge_index'):
                data.edge_index = data.edge_index.to(device)
            output, mask = model(data)
            target = data.y
            loss = criterion(output[mask], target[mask])

            total_loss += loss.item()
            mse = torch.mean((output[mask] - target[mask]) ** 2).item()
            total_mse += mse
            total_mse_per_area += mse / (930 * 930)
            hidden_ratio = 1 - mask.float().mean().item()
            total_hidden_ratio += hidden_ratio
            
            topology_acc = calculate_topology_accuracy(output, target, data.edge_index)
            total_topology_acc += topology_acc

    avg_loss = total_loss / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    avg_mse_per_area = total_mse_per_area / len(val_loader)
    avg_hidden_ratio = total_hidden_ratio / len(val_loader)
    avg_topology_acc = total_topology_acc / len(val_loader)

    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/MSE', avg_mse, epoch)
    writer.add_scalar('Validation/MSE_per_Area', avg_mse_per_area, epoch)
    writer.add_scalar('Validation/Hidden_Ratio', avg_hidden_ratio, epoch)
    writer.add_scalar('Validation/Topology_Accuracy', avg_topology_acc, epoch)

    return avg_loss, avg_mse, avg_mse_per_area, avg_hidden_ratio, avg_topology_acc
