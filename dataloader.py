import os
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def read_txt_file(file_path):
    """
    Read txt file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:] 
    data = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) > 1:
            angle = float(parts[0])
            if angle not in data:
                data[angle] = []
            data[angle].append({
                'object_id': int(parts[1]),
                'object_type': parts[2],
                'color': parts[3],
                'image_center_x': float(parts[7]),
                'image_center_y': float(parts[8]),
                'parent_id': int(parts[10]),
                'distance_to_plane': float(parts[11])
            })
    return data

def find_core_point(data):
    """
    Find core point
    """
    center = np.array([465, 465])  # Image center
    distances = [np.linalg.norm(np.array([d['image_center_x'], d['image_center_y']]) - center) for d in data if d['parent_id'] == -1]
    core_index = distances.index(min(distances))
    return core_index

def create_graph_data(left_data, right_data):
    num_nodes = len(left_data)
    edge_index = []
    x = []
    y = []
    colors = []
    parent_ids = []
    levels = []
    hidden_mask = []
    
    # Find core point
    core_index = find_core_point(left_data)
    
    # Initialize level dictionary
    level_dict = {}
    
    for i, (left_item, right_item) in enumerate(zip(left_data, right_data)):
        parent_id = left_item['parent_id']
        parent_ids.append(parent_id)
        
        # Calculate level
        if parent_id == -1:
            level = 1
        elif parent_id in level_dict:
            level = level_dict[parent_id] + 1
        else:
            level = 2  

        levels.append(level)
        level_dict[left_item['object_id']] = level
        
        is_hidden_left = is_hidden(left_item, next((d for d in left_data if d['object_id'] == left_item['parent_id']), None)) if left_item['parent_id'] != -1 else False
        is_hidden_right = is_hidden(right_item, next((d for d in right_data if d['object_id'] == right_item['parent_id']), None)) if right_item['parent_id'] != -1 else False
        is_hidden_both = is_hidden_left and is_hidden_right
        
        hidden_mask.append(not is_hidden_both)
        
        if not is_hidden_both:
            x.append([left_item['image_center_x'], left_item['image_center_y'],
                      right_item['image_center_x'], right_item['image_center_y']])
            y.append([(left_item['image_center_x'] + right_item['image_center_x']) / 2,
                      (left_item['image_center_y'] + right_item['image_center_y']) / 2])
        else:
            # For hidden points, save the position relative to the parent node
            left_parent = next(d for d in left_data if d['object_id'] == parent_id)
            right_parent = next(d for d in right_data if d['object_id'] == parent_id)
            left_rel_x = left_item['image_center_x'] - left_parent['image_center_x']
            left_rel_y = left_item['image_center_y'] - left_parent['image_center_y']
            right_rel_x = right_item['image_center_x'] - right_parent['image_center_x']
            right_rel_y = right_item['image_center_y'] - right_parent['image_center_y']
            x.append([left_rel_x, left_rel_y, right_rel_x, right_rel_y])
            y.append([0, 0])  # For completely hidden points, we do not predict their position
        
        colors.append(left_item['color'].strip().lower() if isinstance(left_item['color'], str) else 'black')
        
        if parent_id == -1:
            if i != core_index:
                edge_index.append([i, core_index])
                edge_index.append([core_index, i])
        else:
            parent_index = next(j for j, d in enumerate(left_data) if d['object_id'] == parent_id)
            edge_index.append([i, parent_index])
            edge_index.append([parent_index, i])
    
    graph_data = Data(x=torch.tensor(x, dtype=torch.float), 
                      edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
                      y=torch.tensor(y, dtype=torch.float))
    graph_data.level = torch.tensor(levels, dtype=torch.long)
    graph_data.color = colors
    graph_data.parent_ids = torch.tensor(parent_ids, dtype=torch.long)
    graph_data.hidden_mask = torch.tensor(hidden_mask, dtype=torch.bool)
    
    return graph_data

def is_hidden(item, parent):
    if item['parent_id'] == -1 or parent is None:
        return False
    distance = np.sqrt((item['image_center_x'] - parent['image_center_x'])**2 + 
                       (item['image_center_y'] - parent['image_center_y'])**2)
    return item['distance_to_plane'] > parent['distance_to_plane'] and distance < 930 * 0.05

def create_dataloader(data_path, batch_size):
    files = []
    for file in os.listdir(data_path):
        if file.endswith('.txt'):
            files.append(os.path.join(data_path, file))

    data_list = []
    for file in files:
        data = read_txt_file(file)
        angles = sorted(data.keys())
        for i in range(len(angles) - 2):
            left_data = data[angles[i]]
            middle_data = data[angles[i+1]]
            right_data = data[angles[i+2]]
            graph_data = create_graph_data(left_data, right_data)
            graph_data.middle_data = middle_data  
            data_list.append(graph_data)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    return loader

def custom_collate(batch):
    max_nodes = max([data.x.size(0) for data in batch])
    

    batch_x = []
    batch_y = []
    batch_edge_index = []
    batch_mask = []
    batch_level = []
    batch_color = []
    batch_parent_ids = []
    batch_hidden_mask = []
    
    cumsum = 0
    for i, data in enumerate(batch):
        num_nodes = data.x.size(0)
        
        padded_x = torch.zeros((max_nodes, data.x.size(1)))
        padded_x[:num_nodes] = data.x
        batch_x.append(padded_x)
        
        padded_y = torch.zeros((max_nodes, data.y.size(1)))
        padded_y[:num_nodes] = data.y
        batch_y.append(padded_y)
        
        if len(data.edge_index.size()) > 0:
            edge_index = data.edge_index + cumsum
            batch_edge_index.append(edge_index)
        
        padded_level = torch.zeros(max_nodes, dtype=torch.long)
        padded_level[:num_nodes] = data.level
        batch_level.append(padded_level)
        
        padded_hidden_mask = torch.zeros(max_nodes, dtype=torch.bool)
        padded_hidden_mask[:num_nodes] = data.hidden_mask
        batch_hidden_mask.append(padded_hidden_mask)
        
        cumsum += num_nodes
    
    batch_data = Data(
        x=torch.stack(batch_x),
        y=torch.stack(batch_y),
        edge_index=torch.cat(batch_edge_index, dim=1) if batch_edge_index else None,
        level=torch.stack(batch_level),
        hidden_mask=torch.stack(batch_hidden_mask)
    )
    
    return batch_data
