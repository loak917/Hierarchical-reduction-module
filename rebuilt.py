import numpy as np
import laspy
import os
import glob
from tqdm import tqdm
import json
import math
import matplotlib.pyplot as plt

def compute_transform_matrix(predicted_coords):
    """
    Calculate transformation matrix
    """
    transform = np.eye(4)
    
    if predicted_coords is None or len(predicted_coords) == 0:
        return transform
    
    try:
        if predicted_coords.shape[1] == 2:
            z_coords = np.zeros((predicted_coords.shape[0], 1))
            predicted_coords = np.hstack((predicted_coords, z_coords))
            
        center = np.mean(predicted_coords, axis=0)
        transform[0:3, 3] = center
        return transform
    except Exception as e:
        print(f"Warning: Error computing transform matrix: {e}")
        return transform

def reconstruct_point_cloud(predicted_coords, middle_data):
    """
    Reconstruct point cloud 
    """
    reconstructed_points = {}
    
    try:
        # 获取变换矩阵
        transform_matrix = compute_transform_matrix(predicted_coords)
        camera_params = get_camera_parameters()
        
        for orig in middle_data:
            obj_id = orig['object_id']
            if obj_id not in reconstructed_points:
                # 获取图像坐标和深度信息
                image_x = float(orig['image_center_x'])
                image_y = float(orig['image_center_y'])
                distance = float(orig['distance_to_plane'])
                
                # 反投影到3D空间
                x = (image_x - camera_params['principal_point'][0]) * distance / camera_params['focal_length']
                y = (image_y - camera_params['principal_point'][1]) * distance / camera_params['focal_length']
                z = distance
                
                # 应用变换矩阵
                point_3d = np.array([x, y, z, 1.0])
                transformed_point = np.dot(transform_matrix, point_3d)
                
                reconstructed_points[obj_id] = {
                    'object_id': obj_id,
                    'x': float(np.clip(transformed_point[0], -1e5, 1e5)),
                    'y': float(np.clip(transformed_point[1], -1e5, 1e5)),
                    'z': float(np.clip(transformed_point[2], -1e5, 1e5)),
                    'parent_id': orig['parent_id'],
                    'is_hidden': False,
                    'hidden_count': 0
                }
        
        # 处理隐藏点
        for obj_id, point in reconstructed_points.items():
            parent_id = point['parent_id']
            if parent_id != -1 and parent_id in reconstructed_points:
                parent = reconstructed_points[parent_id]
                if is_hidden(point, parent):
                    point['hidden_count'] += 1
        
        # 更新隐藏状态
        for point in reconstructed_points.values():
            point['is_hidden'] = point['hidden_count'] >= 3
            del point['hidden_count']
        
        return list(reconstructed_points.values())
    
    except Exception as e:
        print(f"Error in point cloud reconstruction: {e}")
        return []

def get_camera_parameters():
    """
    Return camera parameters
    """
    return {
        'image_size': 930,
        'elevation': 10.0,  # 与3D转2D中的elev参数一致
        'radius_factor': 1.5  # 相机距离中心点的倍数
    }

def is_hidden(item, parent):
    distance = np.sqrt((item['x'] - parent['x'])**2 + (item['y'] - parent['y'])**2 + (item['z'] - parent['z'])**2)
    return item['z'] > parent['z'] and distance < 930 * 0.05

def save_reconstructed_txt(points, output_file):
    with open(output_file, 'w') as f:
        f.write("Object ID,Real Center X,Real Center Y,Real Center Z,Parent ID,Is Hidden\n")
        for point in points:
            f.write(f"{point['object_id']},{point['x']:.6f},{point['y']:.6f},{point['z']:.6f},{point['parent_id']},{int(point['is_hidden'])}\n")

def reconstruct_laz_file(reconstructed_txt, original_laz_files, output_laz):
    """
    重构LAZ文件 - 保持点云簇的相对结构
    """
    print("\nStarting point cloud reconstruction...")
    
    try:
        reconstructed_centers = {}
        with open(reconstructed_txt, 'r') as f:
            next(f)  
            for line in f:
                obj_id, x, y, z, parent_id, is_hidden = line.strip().split(',')
                reconstructed_centers[int(obj_id)] = {
                    'new_center': np.array([float(x), float(y), float(z)]),
                    'is_hidden': bool(int(is_hidden))
                }
        
        first_las = laspy.read(original_laz_files[0])
        header = laspy.LasHeader(point_format=first_las.header.point_format, 
                               version=first_las.header.version)
        header.scales = first_las.header.scales
        header.offsets = first_las.header.offsets
        
        all_points = []
        
        for laz_file in original_laz_files:
            las = laspy.read(laz_file)
            obj_id = int(os.path.splitext(os.path.basename(laz_file))[0])
            
            if obj_id in reconstructed_centers:
                current_center = np.array([
                    np.mean(las.x),
                    np.mean(las.y),
                    np.mean(las.z)
                ])
                
                new_center = reconstructed_centers[obj_id]['new_center']
                offset = new_center - current_center
                
                new_points = {
                    'x': las.x + offset[0],
                    'y': las.y + offset[1],
                    'z': las.z + offset[2],
                    'classification': las.classification,
                    'intensity': las.intensity,
                    'red': las.red,
                    'green': las.green,
                    'blue': las.blue,
                    'return_number': las.return_number,
                    'number_of_returns': las.number_of_returns,
                    'scan_direction_flag': las.scan_direction_flag,
                    'edge_of_flight_line': las.edge_of_flight_line,
                    'scan_angle_rank': las.scan_angle_rank,
                    'user_data': las.user_data,
                    'point_source_id': las.point_source_id
                }
                all_points.append(new_points)
        
        total_points = sum(len(points['x']) for points in all_points)
        merged_las = laspy.LasData(header)
        
        if total_points > 0:
            for dimension in all_points[0].keys():
                merged_data = np.concatenate([points[dimension] for points in all_points])
                setattr(merged_las, dimension, merged_data)
            
            merged_las.write(output_laz)
            print(f"Successfully wrote reconstructed LAZ file to {output_laz}")
        else:
            print("Error: No points found for reconstruction")
            
    except Exception as e:
        print(f"Error in LAZ reconstruction: {e}")

def process_reconstruction(predicted_coords, middle_data, output_dir, original_laz_dir):
    """
    Process reconstruction - based on multi-view data
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        view_data = {}  
        txt_files = glob.glob(os.path.join(original_laz_dir, "*.txt"))
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                header = next(f)
                for line in f:
                    angle, obj_id, _, _, real_x, real_y, real_z, img_x, img_y, _, parent_id, distance = line.strip().split(',')
                    if obj_id not in view_data:
                        view_data[int(obj_id)] = {}
                    view_data[int(obj_id)][float(angle)] = {
                        'object_id': int(obj_id),
                        'image_center_x': float(img_x),
                        'image_center_y': float(img_y),
                        'real_center_x': float(real_x),
                        'real_center_y': float(real_y),
                        'real_center_z': float(real_z),
                        'distance_to_plane': float(distance),
                        'parent_id': int(parent_id),
                        'is_hidden': False
                    }
        
        reconstructed_points = []
        for obj_id, angles in view_data.items():
            all_centers = np.array([[data['real_center_x'], data['real_center_y'], data['real_center_z']] 
                                   for data in angles.values()])
            scene_center = np.mean(all_centers, axis=0)
            
            visible_coords = []
            for angle, data in angles.items():
                if not data['is_hidden']:
                    x, y, z = project_2d_to_3d(
                        data['image_center_x'],
                        data['image_center_y'],
                        data['distance_to_plane'],
                        angle,
                        scene_center
                    )
                    visible_coords.append([x, y, z])
            
            if visible_coords:
                final_coord = np.mean(visible_coords, axis=0)
            else:
                first_data = list(angles.values())[0]
                final_coord = [
                    first_data['real_center_x'],
                    first_data['real_center_y'],
                    first_data['real_center_z']
                ]
            
            reconstructed_points.append({
                'object_id': obj_id,
                'x': float(final_coord[0]),
                'y': float(final_coord[1]),
                'z': float(final_coord[2]),
                'parent_id': list(angles.values())[0]['parent_id'],
                'is_hidden': len(visible_coords) == 0
            })
        
        txt_output = os.path.join(output_dir, 'reconstructed.txt')
        save_reconstructed_txt(reconstructed_points, txt_output)

        original_laz_files = glob.glob(os.path.join(original_laz_dir, "split_*", "cluster_*", "cluster_*", "*.laz"))
        if original_laz_files:
            laz_output = os.path.join(output_dir, 'reconstructed.laz')
            reconstruct_laz_file(txt_output, sorted(set(original_laz_files)), laz_output)
            
    except Exception as e:
        print(f"Error in reconstruction process: {e}")

def project_2d_to_3d(image_x, image_y, distance, angle, scene_center):
    """
    将2D图像坐标投影回3D空间
    """

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))
    
    ax.view_init(elev=10., azim=angle)
    
    angle_rad = math.radians(angle)
    camera_pos = (
        scene_center[0] + (distance * 1.5) * math.cos(angle_rad),
        scene_center[1] + (distance * 1.5) * math.sin(angle_rad),
        scene_center[2]
    )
    
    proj_matrix = ax.get_proj()
    
    x2d = image_x / 930
    y2d = 1 - (image_y / 930)
    
    camera_direction = np.array([
        scene_center[0] - camera_pos[0],
        scene_center[1] - camera_pos[1],
        scene_center[2] - camera_pos[2]
    ])
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    
    point = camera_pos[0] + camera_direction[0] * distance
    point_y = camera_pos[1] + camera_direction[1] * distance
    point_z = camera_pos[2] + camera_direction[2] * distance
    
    plt.close(fig)
    return point, point_y, point_z
