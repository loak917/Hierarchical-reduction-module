import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader import create_dataloader
from model import PredictMiddleViewResidualGCN
from train import train, validate
from test import test
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiplicativeLR
import math

def combined_annealing_with_decay(optimizer, epoch, base_lr=0.001, period=10, decay_factor=0.8):
    """
    Custom learning rate scheduler: 1/6 period cosine annealing + linear decay + exponential decay
    """
    # Calculate the learning rate within the period
    x = epoch % period

    if x < 0.8 * period:
        decay = 0.5 * (1 + math.cos(math.pi * x / (0.8 * period)))  
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * 0.8))  
        linear_decay = (1 - (x - 0.8 * period) / (0.2 * period))  
        decay = cosine_decay * linear_decay  

    lr = base_lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if epoch > 0 and epoch % period == 0:
        base_lr *= decay_factor

    return base_lr

def main():
    parser = argparse.ArgumentParser(description='Train a GCN model to predict middle view coordinates.')
    parser.add_argument('--data_path', default=r'path_to_your_data', type=str,  help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save TensorBoard logs')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the model checkpoints')
    parser.add_argument('--num_gcn_layers', type=int, default=10, help='Number of GCN layers')
    parser.add_argument('--original_laz_dir', type=str, default=r'path_to_your_data', help='Directory containing original LAZ files')
    args = parser.parse_args()

    train_loader = create_dataloader(os.path.join(args.data_path, 'train'), args.batch_size)
    val_loader = create_dataloader(os.path.join(args.data_path, 'val'), args.batch_size)
    test_loader = create_dataloader(os.path.join(args.data_path, 'test'), args.batch_size)

    input_dim = 4
    hidden_dim = 64
    output_dim = 2 
    model = PredictMiddleViewResidualGCN(input_dim, hidden_dim, output_dim, args.num_gcn_layers)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=args.log_dir)

    os.makedirs(args.model_dir, exist_ok=True)

    best_mse = float('inf')

    base_lr = args.learning_rate
    
    for epoch in range(args.epochs):
        base_lr = combined_annealing_with_decay(optimizer, epoch, base_lr=base_lr, period=10, decay_factor=0.8)

        train_loss, train_mse, train_mse_per_area, train_hidden_ratio, train_topology_acc = train(model, train_loader, criterion, optimizer, epoch, writer)
        val_loss, val_mse, val_mse_per_area, val_hidden_ratio, val_topology_acc = validate(model, val_loader, criterion, epoch, writer)
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Train MSE per Area: {train_mse_per_area:.4f}, Train Hidden Ratio: {train_hidden_ratio:.4f}, Train Topology Accuracy: {train_topology_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MSE per Area: {val_mse_per_area:.4f}, Val Hidden Ratio: {val_hidden_ratio:.4f}, Val Topology Accuracy: {val_topology_acc:.4f}')
        
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f"New best model saved with MSE: {best_mse:.4f}")

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    test_loss, test_mse, test_mse_per_area, test_hidden_ratio, test_topology_acc = test(model, test_loader, criterion, args.model_dir, args.original_laz_dir)
    print(f'Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, Test MSE per Area: {test_mse_per_area:.4f}, Test Hidden Ratio: {test_hidden_ratio:.4f}, Test Topology Accuracy: {test_topology_acc:.4f}')

    writer.close()

if __name__ == '__main__':
    main()
