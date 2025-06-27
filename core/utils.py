import os
import logging
import random
import numpy as np
import torch
# from sklearn.manifold import TSNE
# from scipy.spatial import ConvexHull 
# import matplotlib.pyplot as plt



def setup_logger(log_path):
    """
    设置日志记录器
    
    Args:
        log_path (str): 日志文件路径
    """
    # 创建日志目录
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 配置日志记录器
    logger = logging.getLogger('classification_logger')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



def set_seed(seed=42, logger=None):
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed (int): 随机种子值
        logger: 日志记录器
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    if logger:
        logger.info(f"随机种子已设置为: {seed}")


def get_minimize_lr_epochs(T_0, T_mult):
    res = [T_0]
    for i in range(1, 10):
        res.append(res[-1] + T_0*(T_mult**i))
    return res




# def plot_tsne(features: np.ndarray, labels: np.ndarray, save_path: str, title: str, label_name_dict: dict = None, return_circle5_idx: bool = False, return_class0_outlier_idx: bool = False):
#     print(f"Plotting t-SNE for {title} with {features.shape} samples.")

#     if features.ndim != 2 or labels.ndim != 1:
#         raise ValueError("Features should be 2D and labels should be 1D arrays")

#     if label_name_dict is None:
#         label_name_dict = {i: str(i) for i in np.unique(labels)}
#     color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'black'][:len(label_name_dict)]

#     tsne = TSNE(n_components=2, random_state=42)
#     features_2d = tsne.fit_transform(features)
#     print(f"t-SNE completed. Reduced features shape: {features_2d.shape}")
#     plt.figure(figsize=(8, 8))

#     unique_labels = np.unique(labels)
#     for i, lab in enumerate(unique_labels):
#         idx = labels == lab
#         color = color_list[i % len(color_list)]
#         label_name = label_name_dict[lab]
#         plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label_name, alpha=0.6, s=10, color=color)

#     idx_0 = labels == 0
#     features_0 = features_2d[idx_0]
#     mask = None
#     hull = None
#     if features_0.shape[0] > 2:
#         # IQR方法去除离群点
#         Q1 = np.percentile(features_0, 25, axis=0)
#         Q3 = np.percentile(features_0, 75, axis=0)
#         IQR = Q3 - Q1
#         mask = np.all((features_0 >= (Q1 - 1.5 * IQR)) & (features_0 <= (Q3 + 1.5 * IQR)), axis=1)
#         features_0_inlier = features_0[mask]
#         if features_0_inlier.shape[0] > 2:
#             # 计算凸包
#             hull = ConvexHull(features_0_inlier, qhull_options='QJ')
#             hull_points = features_0_inlier[hull.vertices]
#             # 计算中心点
#             center = features_0_inlier.mean(axis=0)
#             # 向内缩小10%
#             shrinked_hull_points = center + 1.0 * (hull_points - center)
#             # 画多边形
#             plt.plot(
#                 np.append(shrinked_hull_points[:, 0], shrinked_hull_points[0, 0]),
#                 np.append(shrinked_hull_points[:, 1], shrinked_hull_points[0, 1]),
#                 color='red', linewidth=2, linestyle='--', label='Class 0 Polygon'
#             )
#             hull_points = shrinked_hull_points  # 后续用于Path判断
#             # 画中心点
#             center = features_0_inlier.mean(axis=0)
#             plt.scatter(center[0], center[1], color='red', marker='x', s=80, label='Class 0 Center')
#             # 构造多边形路径用于点内判断
#             from matplotlib.path import Path
#             hull_path = Path(hull_points)

#     plt.legend()
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def plot_tsne2(features: np.ndarray, labels: np.ndarray, save_path: str, title: str, 
#               label_name_dict: dict = None, target_classid=5, dim: int = 2):
#     print(f"Plotting {dim}D t-SNE for {title} with {features.shape} samples.")
    
#     if features.ndim != 2 or labels.ndim != 1:
#         raise ValueError("Features should be 2D and labels should be 1D arrays")

#     if label_name_dict is None:
#         label_name_dict = {i: str(i) for i in np.unique(labels)}
#     color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'black'][:len(label_name_dict)]

#     tsne = TSNE(n_components=dim, random_state=42)
#     features_reduced = tsne.fit_transform(features)
#     print(f"t-SNE completed. Reduced features shape: {features_reduced.shape}")
    
#     # Initialize variables to store indices of points inside the polygon
#     inside_polygon_indices = np.array([], dtype=int)
    
#     if dim == 2:
#         plt.figure(figsize=(8, 8))
        
#         unique_labels = np.unique(labels)
#         for i, lab in enumerate(unique_labels):
#             idx = labels == lab
#             color = color_list[i % len(color_list)]
#             label_name = label_name_dict[lab]
#             plt.scatter(features_reduced[idx, 0], features_reduced[idx, 1], label=label_name, alpha=0.6, s=10, color=color)

#         idx_0 = labels == target_classid
#         features_0 = features_reduced[idx_0]
#         mask = None
#         hull = None
#         if features_0.shape[0] > 2:
#             # IQR method to remove outliers
#             Q1 = np.percentile(features_0, 25, axis=0)
#             Q3 = np.percentile(features_0, 75, axis=0)
#             IQR = Q3 - Q1
#             mask = np.all((features_0 >= (Q1 - 1.5 * IQR)) & (features_0 <= (Q3 + 1.5 * IQR)), axis=1)
#             features_0_inlier = features_0[mask]
#             if features_0_inlier.shape[0] > 2:
#                 # Calculate convex hull
#                 hull = ConvexHull(features_0_inlier, qhull_options='QJ')
#                 hull_points = features_0_inlier[hull.vertices]
#                 # Calculate center point
#                 center = features_0_inlier.mean(axis=0)
#                 # Shrink hull by 10%
#                 shrinked_hull_points = center + 1.0 * (hull_points - center)
#                 # Draw polygon
#                 plt.plot(
#                     np.append(shrinked_hull_points[:, 0], shrinked_hull_points[0, 0]),
#                     np.append(shrinked_hull_points[:, 1], shrinked_hull_points[0, 1]),
#                     color='red', linewidth=2, linestyle='--', label=f'Class {target_classid} Polygon'
#                 )
#                 # Draw center point
#                 plt.scatter(center[0], center[1], color='red', marker='x', s=80, label='Class Center')
                
#                 # Create polygon path for point containment check
#                 from matplotlib.path import Path
#                 hull_path = Path(shrinked_hull_points)
                
#                 # Find all points inside the polygon
#                 inside_mask = hull_path.contains_points(features_reduced)
#                 inside_polygon_indices = np.where(inside_mask)[0]

#         plt.legend()
#         plt.title(title)
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close()
        
#     elif dim == 3:
#         from mpl_toolkits.mplot3d import Axes3D
        
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         unique_labels = np.unique(labels)
#         for i, lab in enumerate(unique_labels):
#             idx = labels == lab
#             color = color_list[i % len(color_list)]
#             label_name = label_name_dict[lab]
#             ax.scatter(
#                 features_reduced[idx, 0], 
#                 features_reduced[idx, 1], 
#                 features_reduced[idx, 2],
#                 label=label_name, alpha=0.6, s=10, color=color
#             )
            
#         idx_0 = labels == target_classid
#         features_0 = features_reduced[idx_0]
#         if features_0.shape[0] > 3:
#             # IQR method to remove outliers
#             Q1 = np.percentile(features_0, 25, axis=0)
#             Q3 = np.percentile(features_0, 75, axis=0)
#             IQR = Q3 - Q1
#             mask = np.all((features_0 >= (Q1 - 1.5 * IQR)) & (features_0 <= (Q3 + 1.5 * IQR)), axis=1)
#             features_0_inlier = features_0[mask]
            
#             if features_0_inlier.shape[0] > 3:
#                 # Calculate center point
#                 center = features_0_inlier.mean(axis=0)
#                 # Calculate 3D convex hull
#                 hull = ConvexHull(features_0_inlier)
                
#                 # Draw hull faces
#                 for simplex in hull.simplices:
#                     simplex = np.append(simplex, simplex[0])  # Close the loop
#                     ax.plot(features_0_inlier[simplex, 0], 
#                             features_0_inlier[simplex, 1], 
#                             features_0_inlier[simplex, 2], 
#                             'r-', alpha=0.3)
                
#                 # Draw center point
#                 ax.scatter(center[0], center[1], center[2], 
#                            color='red', marker='x', s=100, label=f'Class {target_classid} Center')
                
#                 # For 3D, finding points inside a convex hull requires additional calculations
#                 from scipy.spatial import Delaunay
#                 hull_delaunay = Delaunay(features_0_inlier[hull.vertices])
#                 inside_mask = hull_delaunay.find_simplex(features_reduced) >= 0
#                 inside_polygon_indices = np.where(inside_mask)[0]
                
#         ax.set_xlabel('t-SNE 1')
#         ax.set_ylabel('t-SNE 2')
#         ax.set_zlabel('t-SNE 3')
#         ax.legend()
#         plt.title(title)
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close()
    
#     else:
#         raise ValueError("Only 2D and 3D visualizations are supported (dim=2 or dim=3)")
    
#     # Return indices of points inside the polygon
#     return inside_polygon_indices


def compute_similarity(features_a, features_b, method='both'):
    """
    Compute similarity between feature vectors using cosine similarity and/or Euclidean distance.
    
    Parameters:
    -----------
    features_a : numpy array
        First feature vector or array of feature vectors
    features_b : numpy array
        Second feature vector or array of feature vectors
    method : str, optional (default='both')
        Similarity method to use: 'cosine', 'euclidean', or 'both'
        
    Returns:
    --------
    dict or float
        If method='both': dictionary containing both similarity scores
        If method='cosine' or 'euclidean': single similarity score
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import euclidean
    
    # Reshape if necessary to ensure 2D arrays
    if features_a.ndim == 1:
        features_a = features_a.reshape(1, -1)
    if features_b.ndim == 1:
        features_b = features_b.reshape(1, -1)
    
    if method == 'cosine' or method == 'both':
        # Cosine similarity ranges from -1 (opposite) to 1 (identical)
        cos_sim = cosine_similarity(features_a, features_b)
        
        # If single vector comparison, return scalar
        if cos_sim.size == 1:
            cos_sim = float(cos_sim[0, 0])
    
    if method == 'euclidean' or method == 'both':
        # For single vector comparison
        if features_a.shape[0] == 1 and features_b.shape[0] == 1:
            euc_dist = euclidean(features_a[0], features_b[0])
            
            # Convert distance to similarity (higher is more similar)
            # Use a negative exponential transformation
            euc_sim = np.exp(-euc_dist / 100)  # Dividing by constant to scale
        else:
            # For multiple vector comparisons
            from sklearn.metrics.pairwise import euclidean_distances
            euc_dist = euclidean_distances(features_a, features_b)
            
            # Convert distances to similarities
            euc_sim = np.exp(-euc_dist / 100)
            
            # If single comparison result, return scalar
            if euc_sim.size == 1:
                euc_sim = float(euc_sim[0, 0])
    
    # Return results based on method
    if method == 'cosine':
        return cos_sim
    elif method == 'euclidean':
        return euc_sim
    else:  # both
        return {
            'cosine_similarity': cos_sim,
            'euclidean_similarity': euc_sim
        }
