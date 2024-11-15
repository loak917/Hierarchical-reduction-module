import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import networkx as nx
from rebuilt import process_reconstruction

def create_image(positions, colors, edges, parent_ids):
    if len(positions) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)  # Return a blank image

    G = nx.Graph()
    max_depth = max(get_node_depth(parent_ids, i) for i in range(len(positions)))
    if max_depth == 0:
        max_depth = 1  

    for i, pos in enumerate(positions):
        if pos[0] != -1 and pos[1] != -1:  
            depth = get_node_depth(parent_ids, i)
            size = 1000 * (1 - 0.8 * depth / max_depth) ** 2  
            color = colors[i] if i < len(colors) and isinstance(colors[i], str) else 'black'
            G.add_node(i, pos=(pos[0], pos[1]), color=color, size=size)

    for edge in edges:
        if G.has_node(edge[0]) and G.has_node(edge[1]):  
            G.add_edge(edge[0], edge[1])

    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, 
            node_size=node_sizes, font_size=8, font_color='black', edge_color='gray')

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def get_node_depth(parent_ids, node_id):
    depth = 0
    while parent_ids[node_id] != -1:
        node_id = parent_ids[node_id]
        depth += 1
    return depth

def test(model, test_loader, criterion, output_dir, original_laz_dir):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mse_per_area = 0
    total_hidden_ratio = 0
    total_topology_acc = 0
    total_samples = 0
    total_points = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output, mask = model(data) 
            target = data.y
            
            if output.shape != target.shape:
                continue
                
            edge_index = data.edge_index
            output_diffs = output[edge_index[0]] - output[edge_index[1]]
            target_diffs = target[edge_index[0]] - target[edge_index[1]]
            
            output_diffs = F.normalize(output_diffs, p=2, dim=1)
            target_diffs = F.normalize(target_diffs, p=2, dim=1)
            
            valid_edges = mask[edge_index[0]] & mask[edge_index[1]]
            if valid_edges.sum() > 0:
                loss = criterion(output_diffs[valid_edges], target_diffs[valid_edges])

                mse = torch.mean((output_diffs[valid_edges] - target_diffs[valid_edges]) ** 2)
                

                batch_size = valid_edges.sum().item()
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_mse += mse.item() * batch_size
                total_mse_per_area += (mse.item() / (930 * 930)) * batch_size
            
            total_hidden_ratio += (1 - mask.float().mean().item()) * len(mask)
            total_points += len(mask)
            
            topology_acc = calculate_topology_accuracy(output, target, data.edge_index)
            total_topology_acc += topology_acc
            
    process_reconstruction(output.cpu().numpy(), data.middle_data, output_dir, original_laz_dir)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_mse = total_mse / total_samples if total_samples > 0 else 0
    avg_mse_per_area = total_mse_per_area / total_samples if total_samples > 0 else 0
    avg_hidden_ratio = total_hidden_ratio / total_points if total_points > 0 else 0
    avg_topology_acc = total_topology_acc / len(test_loader)

    return avg_loss, avg_mse, avg_mse_per_area, avg_hidden_ratio, avg_topology_acc

def calculate_relative_error(output, target, edge_index):
    relative_errors = []
    for edge in edge_index.T:
        i, j = edge
        if i < output.shape[0] and j < output.shape[0]:
            pred_diff = output[i] - output[j]
            true_diff = target[i] - target[j]
            if torch.norm(true_diff) > 1e-6:  # Avoid division by zero
                relative_error = torch.norm(pred_diff - true_diff) / torch.norm(true_diff)
                relative_errors.append(relative_error.item())
    return np.mean(relative_errors) if relative_errors else 0.0

def calculate_structural_similarity(output, target, edge_index):
    pred_adj = torch.zeros((output.shape[0], output.shape[0]))
    true_adj = torch.zeros((target.shape[0], target.shape[0]))
    for edge in edge_index.T:
        i, j = edge
        if i < output.shape[0] and j < output.shape[0]:
            pred_adj[i, j] = pred_adj[j, i] = torch.norm(output[i] - output[j])
            true_adj[i, j] = true_adj[j, i] = torch.norm(target[i] - target[j])
    return F.mse_loss(pred_adj, true_adj).item()

def calculate_topology_accuracy(output, target, edge_index):
    """Calculate branch topology accuracy"""
    def build_graph(coords, edges):
        G = nx.Graph()
        for i in range(len(coords)):
            G.add_node(i, pos=coords[i].detach().cpu().numpy())
        for e in edges.T:
            G.add_edge(e[0].item(), e[1].item())
        return G
    
    # Build predicted and true graph structures
    pred_graph = build_graph(output, edge_index)
    true_graph = build_graph(target, edge_index)
    
    # Calculate graph similarity
    total_edges = len(true_graph.edges())
    correct_edges = 0
    
    for edge in true_graph.edges():
        if pred_graph.has_edge(*edge):
            true_length = np.linalg.norm(
                np.array(true_graph.nodes[edge[0]]['pos']) - 
                np.array(true_graph.nodes[edge[1]]['pos'])
            )
            pred_length = np.linalg.norm(
                np.array(pred_graph.nodes[edge[0]]['pos']) - 
                np.array(pred_graph.nodes[edge[1]]['pos'])
            )
            if true_length > 1e-6:  # Add a small threshold
                if abs(true_length - pred_length) / true_length < 0.1:  
                    correct_edges += 1
            elif abs(true_length - pred_length) < 1e-6: 
                correct_edges += 1
    
    return correct_edges / total_edges if total_edges > 0 else 0.0
