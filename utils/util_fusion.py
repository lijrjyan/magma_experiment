import copy
import logging
import warnings
import traceback
from typing import Any, Dict, List, Tuple, Optional, Union, Literal
from pathlib import Path

import torch
import torch.utils.data as data
import tenseal as ts
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from utils.util_crypto import context_ckks
from utils.util_model import (
    extract_parameters,
    flatten_model_parameters,
    load_model_from_parameters,
    get_gaussian_noise,
    get_laplace_noise,
)
from utils.util_sys import wrap_torch_median
from utils.util_sys import wrap_torch_sort

from utils.util_logger import logger

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")


def fusion_avg(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    avgerage_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            weighted_params = torch.zeros_like(
                next(iter(model_updates.values())).state_dict()[key].float()
            )
            for _, model in model_updates.items():
                param = model.state_dict()[key]
                # 处理不同类型的参数，确保类型兼容性
                if param.dtype == torch.long or param.dtype == torch.int64 or param.dtype == torch.int32 or param.dtype == torch.int16 or param.dtype == torch.int8:
                    # 对于整数类型的参数，先转换为浮点数，计算后再转回原类型
                    weighted_params += (param.float() * (1.0 / len(model_updates))).to(param.dtype)
                else:
                    weighted_params += param.float() * (1.0 / len(model_updates))
            avgerage_params[key] = weighted_params

    return avgerage_params


def fusion_fedavg(
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
) -> Dict[str, torch.Tensor]:

    total_data_size = sum(data_size.values())
    weighted_avg_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            weighted_params = torch.zeros_like(
                next(iter(model_updates.values())).state_dict()[key].float()
            )
            for client_id, model in model_updates.items():
                weight = data_size[client_id] / total_data_size
                param = model.state_dict()[key]
                # 处理不同类型的参数，确保类型兼容性
                if param.dtype == torch.long or param.dtype == torch.int64 or param.dtype == torch.int32 or param.dtype == torch.int16 or param.dtype == torch.int8:
                    # 对于整数类型的参数，先转换为浮点数，计算后再转回原类型
                    weighted_params += (param.float() * weight).to(param.dtype)
                else:
                    weighted_params += param.float() * weight
            weighted_avg_params[key] = weighted_params

    return weighted_avg_params


def fusion_krum(
    model_updates: Dict[int, torch.nn.Module],
    max_expected_adversaries=1,
    device=torch.device("cpu"),
) -> Dict[str, torch.Tensor]:

    with torch.no_grad():
        ids = list(model_updates.keys())
        updates = [extract_parameters(model_updates[id]) for id in ids]
        updates = [update.to(device) for update in updates]
        num_updates = len(updates)
        updates_stack = torch.stack(updates)

        dist_matrix = torch.cdist(updates_stack, updates_stack, p=2)
        values, indices = torch.topk(
            dist_matrix,
            k=num_updates - max_expected_adversaries - 1,
            dim=1,
            largest=False,
            sorted=True,
        )
        # logger.debug(f"current krum values: {values}")
        scores = values.sum(dim=1)
        # logger.debug(f"current krum scores: {scores}")
        min_indices = torch.argmin(scores).item()
        logger.debug(f"current krum min index: {min_indices}")
        selected_id = ids[min_indices]
        logger.info(f"selected client id: {selected_id}")

    selected_model = model_updates[selected_id]
    krum_params = selected_model.state_dict()
    return krum_params


def fusion_median(
    model_updates: Dict[int, torch.nn.Module],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            median_params[key] = wrap_torch_median(params, dim=0, device=device)

    return median_params


def fusion_clipping_median(
    model_updates: Dict[int, torch.nn.Module],
    clipping_threshold=0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            median_params[key] = wrap_torch_median(params, dim=0, device=device)
            median_params[key] = torch.clamp(
                median_params[key], -clipping_threshold, clipping_threshold
            )

    return median_params


def fusion_trimmed_mean(
    model_updates: Dict[int, torch.nn.Module],
    trimmed_ratio: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    trimmed_mean_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            lower = int(params.size(0) * trimmed_ratio)
            upper = int(params.size(0) * (1 - trimmed_ratio))
            params = wrap_torch_sort(params, dim=0, device=device)[lower:upper]
            trimmed_mean_params[key] = torch.mean(params, dim=0)

    return trimmed_mean_params


def fusion_cos_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int] = None,
    similarity_threshold: float = None,
) -> Dict[str, Any]:

    global_last_layer = list(global_model.parameters())[-2].view(-1)
    models = list(model_updates.values())
    client_ids = list(model_updates.keys())
    last_layers = [list(model.parameters())[-2].view(-1) for model in models]

    with torch.no_grad():
        scores = torch.abs(
            torch.nn.functional.cosine_similarity(
                torch.stack(last_layers),
                global_last_layer,
            )
        )
        # print(scores)
        logger.info(f"current fusion scores: {scores}")
        min_score = torch.min(scores)
        scores = (scores - min_score) / (torch.max(scores) - min_score)
        logger.info(f"normalized fusion scores: {scores}")

        if similarity_threshold is None:
            similarity_threshold = torch.mean(scores)
        logger.info(f"similarity threshold: {similarity_threshold}")

        benign_indices = scores >= similarity_threshold
        if torch.sum(benign_indices) == 0:
            logger.warning("No models are considered benign based on the threshold.")
            logger.warning("Return global model of last round.")
            return global_model.state_dict()

        logger.info(f"current round client list: {model_updates.keys()}")
        logger.info(f"potential malicide indices: {benign_indices}")
        logger.info(f"checked benign indices: {benign_indices}")

        # 获取被选为良性的客户端ID
        benign_client_ids = [client_ids[i] for i in range(len(client_ids)) if benign_indices[i]]
        logger.info(f"benign client IDs: {benign_client_ids}")
        
        # 如果提供了数据大小，使用数据大小进行加权
        if data_size is not None and all(client_id in data_size for client_id in benign_client_ids):
            # 计算总数据大小
            total_size = sum(data_size[client_id] for client_id in benign_client_ids)
            # 计算每个良性客户端的权重
            fractions = torch.zeros_like(benign_indices.float())
            for i, client_id in enumerate(client_ids):
                if client_id in benign_client_ids:
                    fractions[i] = data_size[client_id] / total_size
            logger.info(f"Using data size for weighting. Fractions: {fractions}")
        else:
            # 如果没有提供数据大小，使用均等权重
            weight = 1 / torch.sum(benign_indices).float()
            fractions = benign_indices.float() * weight
            logger.info(f"Using equal weighting. Fractions: {fractions}")

        weighted_params = copy.deepcopy(global_model.state_dict())
        for param_key in weighted_params.keys():
            temp_param = torch.zeros_like(
                global_model.state_dict()[param_key], dtype=torch.float32
            )
            for model, fraction in zip(models, fractions):
                param = model.state_dict()[param_key]
                # 处理不同类型的参数，确保类型兼容性
                if param.dtype == torch.long or param.dtype == torch.int64 or param.dtype == torch.int32 or param.dtype == torch.int16 or param.dtype == torch.int8:
                    # 对于整数类型的参数，先转换为浮点数，计算后再转回原类型
                    temp_param += (param.float() * fraction).to(param.dtype)
                else:
                    temp_param += param * fraction
            weighted_params[param_key].copy_(temp_param)
            # OUR OPTIMIZATION FOR DEFENSE
            # weighted_params[param_key] = torch.clamp(
            #     weighted_params[param_key], -0.1, 0.1
            # )

    return weighted_params



def fusion_dual_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    _round_idx: int = None,
    log_dir: str = None,
    attacker_list: list = None,
    use_fhe: bool = True,
    gradient_collector: Optional[Dict] = None,
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Tuple[Dict[str, torch.Tensor], list]:
    """
    Original dual defense implementation using secure cosine similarity.
    
    Args:
        global_model: The current global model
        model_updates: Dictionary of client models to be aggregated
        data_size: Dictionary of client data sizes for weighted averaging
        _round_idx: The current round index (unused in this implementation)
        log_dir: The directory for logging (unused in this implementation)
        attacker_list: List of attacker client IDs (unused in this implementation)
        use_fhe: Whether to use Homomorphic Encryption (default: True)
        gradient_collector: Optional dictionary for collecting gradient data (unused)
        similarity_threshold: Optional threshold for cosine similarity filtering
        epsilon: Optional epsilon for differential privacy noise
        
    Returns:
        Tuple of (fused model parameters, list of benign clients)
    """
    # Extract the last layer parameters for both global and local models
    global_last_layer = list(global_model.parameters())[-2].view(-1)
    last_layers = {
        client_id: list(model.parameters())[-2].view(-1)
        for client_id, model in model_updates.items()
    }
    normalized_global = global_last_layer / torch.norm(global_last_layer)
    normalized_locals = {
        client_id: last_layer / torch.norm(last_layer)
        for client_id, last_layer in last_layers.items()
    }
    
    # Branch based on whether to use FHE or plaintext operations
    if use_fhe:
        logger.info("Using FHE for secure dual defense")
        # 1) Encrypt and send to the fusion server
        encrypted_global = ts.ckks_vector(
            context_ckks, normalized_global.flatten().tolist()
        )
        encrypted_locals = {
            client_id: ts.ckks_vector(context_ckks, normalized_local.flatten().tolist())
            for client_id, normalized_local in normalized_locals.items()
        }
        encrypted_updates = {}
        for client_id, model in model_updates.items():
            flattened_parameters = flatten_model_parameters(model)
            encrypted_parameters = [
                ts.ckks_vector(context_ckks, param) for param in flattened_parameters
            ]
            encrypted_updates[client_id] = encrypted_parameters

        # 2) Apply differential privacy noise if needed
        if epsilon is not None and isinstance(epsilon, float):
            gaussian_noise = get_gaussian_noise(
                1, epsilon=epsilon, delta=1.0 / encrypted_global.size(), sensitivity=1
            )
            encrypted_global = (
                encrypted_global + gaussian_noise.tolist() * encrypted_global.size()
            )

        # 3) Compute encrypted cosine similarity scores
        encrypted_scores = {
            client_id: encrypted_local.dot(encrypted_global)
            for client_id, encrypted_local in encrypted_locals.items()
        }

        # 4) Decrypt scores and perform client selection
        client_selections = {}
        for client_id in model_updates.keys():
            scores = {
                client_id: np.abs(encrypted_score.decrypt())
                for client_id, encrypted_score in encrypted_scores.items()
            }
            logger.debug(f"client {client_id} scores: {scores}")
            min_score = np.min(list(scores.values()))
            max_score = np.max(list(scores.values()))
            diff_score = max_score - min_score
            scores_norm = {
                client_id: (score - min_score) / diff_score
                for client_id, score in scores.items()
            }
            logger.debug(f"client {client_id} norm scores: {scores_norm}")
            if similarity_threshold is None:
                similarity_threshold = np.mean(list(scores_norm.values()))
            logger.debug(f"client {client_id} similarity threshold: {similarity_threshold}")
            selected_benigns = [
                id for id, score in scores_norm.items() if score >= similarity_threshold
            ]
            logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")
            if len(selected_benigns) == 0:
                raise ValueError("No models are considered benign based on the threshold.")
            client_selections[client_id] = selected_benigns
    else:
        logger.info("Using plaintext operations for dual defense (no FHE)")
        # 1) Calculate cosine similarity directly with PyTorch
        client_selections = {}
        for client_id in model_updates.keys():
            # Compute cosine similarity between each client and global model
            scores = {
                cid: np.abs(torch.nn.functional.cosine_similarity(
                    normalized_locals[cid].unsqueeze(0), 
                    normalized_global.unsqueeze(0)
                ).item())
                for cid in model_updates.keys()
            }
            logger.debug(f"client {client_id} scores: {scores}")
            
            # Apply differential privacy noise if needed
            if epsilon is not None and isinstance(epsilon, float):
                noise_scale = 1.0 / normalized_global.size(0)
                for cid in scores:
                    noise = np.random.normal(0, noise_scale * epsilon)
                    scores[cid] = max(0, min(1, scores[cid] + noise))  # Clamp to [0,1]
            
            # Normalize scores
            min_score = np.min(list(scores.values()))
            max_score = np.max(list(scores.values()))
            diff_score = max_score - min_score
            scores_norm = {
                cid: (score - min_score) / diff_score
                for cid, score in scores.items()
            }
            logger.debug(f"client {client_id} norm scores: {scores_norm}")
            
            # Select benign clients
            if similarity_threshold is None:
                similarity_threshold = np.mean(list(scores_norm.values()))
            logger.debug(f"client {client_id} similarity threshold: {similarity_threshold}")
            
            selected_benigns = [
                id for id, score in scores_norm.items() if score >= similarity_threshold
            ]
            logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")
            if len(selected_benigns) == 0:
                raise ValueError("No models are considered benign based on the threshold.")
            client_selections[client_id] = selected_benigns

    # 5) Find majority vote for benign client selection (same for both approaches)
    count = {}
    for _, benigns in client_selections.items():
        _tuple = tuple(benigns)
        if _tuple in count:
            count[_tuple] += 1
        else:
            count[_tuple] = 1
    benigns_tuple = None
    max_count = 0
    for _benigns, _cnt in count.items():
        if _cnt > max_count:
            max_count = _cnt
            benigns_tuple = _benigns
    
    # Convert tuple to list for return value consistency
    benigns = list(benigns_tuple) if benigns_tuple else []
    logger.debug(f"final fusion benigns: {benigns}")
    
    # 6) Final aggregation
    if not benigns:
        logger.error("No benign clients selected for aggregation! Using global model as fallback.")
        return global_model.state_dict(), []

    total_size = sum(data_size[benign_id] for benign_id in benigns)
    if total_size == 0:
        logger.error("Total data size of benign clients is zero. Using global model as fallback.")
        return global_model.state_dict(), benigns
    
    # 7) Perform aggregation (encrypted or plaintext)
    try:
        if use_fhe:
            # Secure aggregation with FHE
            fused_enc_params = [0] * len(encrypted_updates[benigns[0]])
            for benign_id in benigns:
                enc_param = encrypted_updates[benign_id]
                fusion_weight = data_size[benign_id] / total_size
                weighted_enc_param = [_p * fusion_weight for _p in enc_param]
                fused_enc_params = [x + y for x, y in zip(fused_enc_params, weighted_enc_param)]

            # Decrypt and load parameters
            _params = [param.decrypt() for param in fused_enc_params]
            fused_model = load_model_from_parameters(_params, global_model)
            fused_params = fused_model.state_dict()
        else:
            # Plaintext weighted averaging
            fused_params = {}
            for key in global_model.state_dict():
                weighted_params = torch.zeros_like(global_model.state_dict()[key].float())
                for benign_id in benigns:
                    model = model_updates[benign_id]
                    param = model.state_dict()[key]
                    fusion_weight = data_size[benign_id] / total_size
                    
                    # Handle different parameter types
                    if param.dtype in [torch.long, torch.int64, torch.int32, torch.int16, torch.int8]:
                        weighted_params += (param.float() * fusion_weight).to(param.dtype)
                    else:
                        weighted_params += param.float() * fusion_weight
                fused_params[key] = weighted_params
        
        # Check for NaN/Inf values (same for both approaches)
        for key, param in fused_params.items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                fused_params[key] = global_model.state_dict()[key]
                logger.warning(f"Replaced NaN/Inf in parameter {key} with global model parameter.")
                
        return fused_params, benigns
    except Exception as e:
        logger.error(f"Error during dual defense aggregation: {str(e)}")
        return global_model.state_dict(), benigns


def plot_dendrogram(Z, client_ids, attacker_list, _round_idx, log_dir):
    """
    Generate a dendrogram visualization to identify client groupings.
    
    Args:
        Z: Linkage matrix from hierarchical clustering
        client_ids: List of client IDs
        attacker_list: List of known attackers (for visualization only)
        _round_idx: Current round index
        log_dir: Directory to save the visualization
        
    Returns:
        Path to the saved dendrogram image
    """
    # 创建并设置图像
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 设置白色背景
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 为攻击者和良性客户端设置颜色映射
    label_colors = []
    for client_id in client_ids:
        if client_id in attacker_list:
            label_colors.append('#E41A1C')  # 攻击者用红色标记
        else:
            label_colors.append('#377EB8')  # 正常客户端用蓝色标记
    
    # 绘制简洁的树状图
    dendrogram_plot = dendrogram(
        Z, 
        labels=client_ids, 
        leaf_font_size=14,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True
    )
    
    # 自定义叶子节点标签的外观
    xlbls = ax.get_xticklabels()
    for i, lbl in enumerate(xlbls):
        try:
            client_id = int(lbl.get_text())
            if client_id in attacker_list:
                lbl.set_color('#E41A1C')  # 攻击者标签为红色
                lbl.set_weight('bold')
                lbl.set_fontsize(14)  # 增大字体
                lbl.set_bbox(dict(facecolor='#FFEEEE', edgecolor='#E41A1C', alpha=0.7, boxstyle='round,pad=0.2'))
            else:
                lbl.set_color('#377EB8')  # 良性客户端标签为蓝色
                lbl.set_fontsize(14)  # 增大字体
                lbl.set_bbox(dict(facecolor='#EEF7FF', edgecolor='#377EB8', alpha=0.7, boxstyle='round,pad=0.2'))
        except ValueError:
            continue
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#377EB8', markersize=10, 
               label='Benign Clients', markeredgecolor='white', linewidth=1.5),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#E41A1C', markersize=10, 
               label='Attackers', markeredgecolor='white', linewidth=1.5)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    # 添加标题和轴标签
    plt.title(f"Hierarchical Clustering Dendrogram (Round {_round_idx})", 
              fontsize=18, fontweight='normal', pad=20)
    plt.xlabel("Client IDs", fontsize=14, labelpad=10)
    plt.ylabel("Euclidean Distance", fontsize=14, labelpad=10)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 增大刻度标签的字体
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    
    # 确保log_dir存在
    os.makedirs(log_dir, exist_ok=True)
    dendrogram_path = os.path.join(log_dir, f"dendrogram_round_{_round_idx}.png")
    plt.savefig(dendrogram_path, bbox_inches='tight', dpi=300)
    logger.info(f"Enhanced dendrogram visualization saved to: {dendrogram_path}")
    plt.close()  # 关闭图表释放内存
    
    return dendrogram_path


def plot_pca_visualization(raw_gradient_vectors, client_ids, attacker_list, _round_idx, log_dir):
    """
    Generate a PCA visualization of client gradients.
    
    Args:
        raw_gradient_vectors: Array of gradient vectors
        client_ids: List of client IDs
        attacker_list: List of known attackers (for visualization only)
        _round_idx: Current round index
        log_dir: Directory to save the visualization
        
    Returns:
        Path to the saved PCA visualization image
    """
    # 应用PCA降维到2维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(raw_gradient_vectors)
    
    # 创建高质量可视化
    plt.figure(figsize=(8, 6), dpi=300)
    
    # 设置白色背景
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    
    # 绘制数据点 - 使用更清晰的视觉区分
    benign_points = []
    attack_points = []
    benign_ids = []
    attack_ids = []
    benign_coords = []
    attack_coords = []
    
    for i, client_id in enumerate(client_ids):
        if client_id in attacker_list:
            attack_points.append((pca_result[i, 0], pca_result[i, 1]))
            attack_ids.append(client_id)
            attack_coords.append((pca_result[i, 0], pca_result[i, 1]))
        else:
            benign_points.append((pca_result[i, 0], pca_result[i, 1]))
            benign_ids.append(client_id)
            benign_coords.append((pca_result[i, 0], pca_result[i, 1]))
    
    # 绘制良性客户端
    if benign_points:
        benign_x, benign_y = zip(*benign_points)
        plt.scatter(benign_x, benign_y, c='#377EB8', s=120, marker='o', alpha=0.8, 
                   label='Benign Clients', edgecolor='white', linewidth=1.5)
        
        # 添加客户端ID标签
        for client_id, (x, y) in zip(benign_ids, benign_coords):
            plt.annotate(f"{client_id}", (x, y), fontsize=14, ha='center', va='center',
                       color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc='#377EB8', ec="none", alpha=0.7))
    
    # 绘制攻击者
    if attack_points:
        attack_x, attack_y = zip(*attack_points)
        plt.scatter(attack_x, attack_y, c='#E41A1C', s=150, marker='X', alpha=0.8, 
                   label='Attackers', edgecolor='white', linewidth=1.5)
        
        # 添加攻击者ID标签
        for client_id, (x, y) in zip(attack_ids, attack_coords):
            plt.annotate(f"{client_id}", (x, y), fontsize=14, ha='center', va='center',
                       color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc='#E41A1C', ec="none", alpha=0.7))
    
    # 计算数据的边界以放置可能的决策边界
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    
    # 添加标题和轴标签
    plt.title(f"PCA Visualization of Client Gradients (Round {_round_idx})", 
             fontsize=18, fontweight='normal', pad=20)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=14, labelpad=10)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=14, labelpad=10)
    
    # 图例
    legend = plt.legend(loc='upper right', fontsize=14)
    
    # 增大刻度标签的字体
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    
    # 保存PCA可视化
    pca_path = os.path.join(log_dir, f"pca_gradients_round_{_round_idx}.png")
    plt.savefig(pca_path, bbox_inches='tight', dpi=300)
    logger.info(f"PCA visualization saved to: {pca_path}")
    plt.close()
    
    # 打印PCA解释方差
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca_path


def fusion_dendro_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    _round_idx: int,
    log_dir: str,
    attacker_list: list,
    use_fhe: bool = True,
    gradient_collector: Optional[Dict] = None,
) -> Tuple[Dict[str, torch.Tensor], list]:
    """
    An advanced, unified defense mechanism with robust geometry-based filtering.
    Features:
    - Secure distance calculation in FHE mode to prevent privacy leaks.
    - Hierarchical clustering with dendrogram visualization to identify client groups.
    - Automatic decision based on merge distance ratios to detect attackers.
    
    Args:
        global_model: The current global model
        model_updates: Dictionary of client models to be aggregated
        data_size: Dictionary of client data sizes for weighted averaging
        _round_idx: The current round index
        log_dir: The directory for logging
        attacker_list: List of attacker client IDs
        use_fhe: Whether to use Homomorphic Encryption (default: True)
        gradient_collector: Optional dictionary for collecting gradient data for visualization
        
    Returns:
        Tuple of (fused model parameters, list of benign clients)
    """
    # 步骤 1: 提取模型更新的最后一层参数 (两个模式共用)
    global_last_layer = list(global_model.state_dict().values())[-2].view(-1)
    last_layers = {
        client_id: list(model.state_dict().values())[-2].view(-1) - global_last_layer
        for client_id, model in model_updates.items()
    }

    client_ids = list(model_updates.keys())
    n_clients = len(client_ids)
    distance_matrix = np.zeros((n_clients, n_clients))
    encrypted_updates = {}

    # ==================== 步骤 2: 构建距离矩阵 (根据模式选择不同方法) ====================
    if use_fhe:
        logger.info("FHE Mode: Building distance matrix using SECURE privacy-preserving protocol.")
        
        # 2a) 客户端加密梯度 - 使用未归一化的梯度
        encrypted_locals = {
            client_id: ts.ckks_vector(context_ckks, last_layer.flatten().tolist())
            for client_id, last_layer in last_layers.items()
        }
        for cid, model in model_updates.items():
            params = flatten_model_parameters(model)
            encrypted_updates[cid] = [ts.ckks_vector(context_ckks,  p) for p in params]

        # CHANGE 1: SECURE FHE DISTANCE CALCULATION
        # 2b) 服务器同态计算距离的平方，以避免隐私泄露
        logger.info("Server computing encrypted SQUARED distances to prevent privacy leaks.")
        encrypted_sq_distances = {cid: {} for cid in client_ids}
        for i, c_i in enumerate(client_ids):
            for j, c_j in enumerate(client_ids):
                if i < j:
                    enc_diff = encrypted_locals[c_i] - encrypted_locals[c_j]
                    encrypted_sq_distances[c_i][c_j] = (enc_diff * enc_diff).sum()
        
        # 2c) 客户端解密标量值并计算最终距离
        logger.info("Clients decrypting SCALAR squared distances.")
        for i, c_i in enumerate(client_ids):
            for j, c_j in enumerate(client_ids):
                if i < j:
                    sq_dist = encrypted_sq_distances[c_i][c_j].decrypt()[0]
                    distance = np.sqrt(max(0, sq_dist))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
    
    else: # 非FHE模式
        logger.info("Plaintext Mode: Building distance matrix directly on the server.")
        client_vectors = [last_layers[cid] for cid in client_ids]
        for i in range(n_clients):
            for j in range(i, n_clients):
                if i == j: continue
                distance = torch.norm(client_vectors[i] - client_vectors[j]).item()
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    logger.debug(f"Distance matrix (shape: {distance_matrix.shape}):\n{distance_matrix}")

    # ==================== 步骤 3: 统一的聚类与决策逻辑 (两个模式共用) ====================
    logger.info("Unified Defense: Performing hierarchical clustering and validation.")
    benigns = client_ids # 默认情况下，如果出错，则接受所有客户端

    if n_clients > 2:
        try:
            # 3a) 将距离矩阵转换为层次聚类所需的压缩距离向量（上三角形）
            # scipy的linkage函数需要一个一维距离向量，它是距离矩阵的上三角部分
            distance_vector = []
            for i in range(n_clients):
                for j in range(i+1, n_clients):
                    distance_vector.append(distance_matrix[i, j])
            
            # 3b) 执行层次聚类 (Ward方法通常在多数情况下表现稳健)
            Z = linkage(distance_vector, method='ward')
            logger.info(f"Hierarchical clustering completed. Linkage matrix shape: {Z.shape}")
            
            # 3c) 生成并保存树状图可视化结果 - 使用抽取的函数
            dendrogram_path = plot_dendrogram(Z, client_ids, attacker_list, _round_idx, log_dir)
            logger.info(f"Enhanced dendrogram visualization saved to: {dendrogram_path}")
            
            # 3c-2) 生成并保存PCA降维可视化结果 - 使用抽取的函数
            # 转换梯度为numpy数组以便进行PCA
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raw_gradient_vectors = np.array([last_layers[cid].to(device).cpu().numpy() for cid in client_ids])
            
            # 应用PCA并生成可视化
            pca_path = plot_pca_visualization(raw_gradient_vectors, client_ids, attacker_list, _round_idx, log_dir)
            logger.info(f"PCA visualization saved to: {pca_path}")
            
            # 3d) 实现自动化决策逻辑 - 增强版本，处理多个攻击者集群
            
            # 获取所有合并距离并按降序排列
            merge_distances = Z[:, 2]
            
            # 新方法：从最小距离开始检查，识别第一个显著跳跃
            # 按从小到大排序合并距离（linkage Z矩阵中的距离是按照合并过程的顺序排列的）
            sorted_merge_distances = sorted(merge_distances)
            logger.info(f"Sorted merge distances from smallest to largest: {sorted_merge_distances}")
            
            # 设置显著跳跃的比率阈值
            JUMP_RATIO_THRESHOLD = 1.5
            
            # 初始化变量
            significant_jump_idx = None
            significant_jump_ratio = 0
            
            # 从最小距离开始检查相邻距离的比率
            for i in range(len(sorted_merge_distances) - 1):
                if sorted_merge_distances[i] < 1e-9:  # 防止除零
                    continue
                
                # 计算下一个距离与当前距离的比率
                ratio = sorted_merge_distances[i+1] / sorted_merge_distances[i]
                logger.info(f"Distance jump check: {sorted_merge_distances[i]:.4f} -> {sorted_merge_distances[i+1]:.4f}, ratio: {ratio:.4f}")
                
                # 如果发现比率超过阈值，记录跳跃点并停止
                if ratio > JUMP_RATIO_THRESHOLD:
                    significant_jump_idx = i
                    significant_jump_ratio = ratio
                    logger.info(f"Found significant jump at distance {sorted_merge_distances[i]:.4f} -> {sorted_merge_distances[i+1]:.4f}, ratio: {ratio:.4f}")
                    break
            
            # 根据是否找到显著跳跃点来决定聚类方案
            if significant_jump_idx is not None:
                # 找到了显著跳跃点，将阈值设置为跳跃点的距离
                threshold_distance = sorted_merge_distances[significant_jump_idx]
                logger.info(f"Using threshold distance: {threshold_distance:.4f} (before the significant jump)")
                
                # 将所有距离小于阈值的合并归为同一类
                # 使用距离阈值进行聚类
                cluster_labels = fcluster(Z, t=threshold_distance, criterion='distance')
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                logger.info(f"Hierarchical clustering results: labels={unique_labels}, counts={counts}")
                
                # 改进的逻辑: 在阈值下形成的所有簇中，选择最大的簇作为良性客户端
                # 这种方法更有效地处理类似图中显示的场景，其中良性客户端形成一个大簇
                
                # 使用距离阈值进行聚类后，每个客户端被分配到某个簇
                # 现在选择最大的簇作为良性客户端组
                majority_label = unique_labels[np.argmax(counts)]
                logger.info(f"Largest cluster has label {majority_label} with {max(counts)} clients")
                
                # 提取最大簇中的客户端索引
                benign_indices = np.where(cluster_labels == majority_label)[0]
                benigns = [client_ids[i] for i in benign_indices]
                logger.info(f"Selected largest cluster as benign: {len(benigns)} clients with label {majority_label}")
                
                # 记录被排除的客户端（可能是攻击者）
                rejected_clients = [cid for idx, cid in enumerate(client_ids) if idx not in benign_indices]
                if rejected_clients:
                    logger.info(f"Rejected {len(rejected_clients)} clients as potential attackers: {rejected_clients}")
                
            else:
                # 没有找到显著跳跃点，认为所有客户端都是良性的
                logger.info(f"No significant jump found (all ratios <= {JUMP_RATIO_THRESHOLD}). Accepting all clients as benign.")
                benigns = client_ids
        except Exception as e:
            logger.error(f"Error during hierarchical clustering: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Accepting all clients as a fallback.")
            benigns = client_ids
    else:
        logger.warning(f"Only {n_clients} clients, not enough for clustering. Accepting all.")
    
    logger.info(f"Final selected benign clients: {benigns}")

    # ==================== 步骤 4: 最终聚合 (根据模式选择不同方法) ====================
    if not benigns:
        logger.error("No benign clients selected for aggregation! Using global model as fallback.")
        return global_model.state_dict(), []

    total_size = sum(data_size[bid] for bid in benigns)
    if total_size == 0:
        logger.error("Total data size of benign clients is zero. Using global model as fallback.")
        return global_model.state_dict(), benigns

    fused_params = {}
    if use_fhe:
        logger.info("FHE Mode: Performing final secure aggregation.")
        try:
            first_id = benigns[0]
            weight = data_size[first_id] / total_size
            fused_enc_params = [p * weight for p in encrypted_updates[first_id]]

            for bid in benigns[1:]:
                weight = data_size[bid] / total_size
                for i in range(len(fused_enc_params)):
                    fused_enc_params[i] += encrypted_updates[bid][i] * weight
            
            _params = [p.decrypt() for p in fused_enc_params]
            temp_model = copy.deepcopy(global_model)
            fused_model = load_model_from_parameters(_params, temp_model)
            fused_params = fused_model.state_dict()
        except Exception as e:
            logger.error(f"FHE aggregation/decryption failed: {e}. Using global model as fallback.")
            fused_params = global_model.state_dict()
    else:
        logger.info("Plaintext Mode: Performing final aggregation.")
        fused_params = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        for bid in benigns:
            weight = data_size[bid] / total_size
            for k, v in model_updates[bid].state_dict().items():
                if fused_params[k].dtype.is_floating_point:
                    fused_params[k] += v * weight
                else:
                    fused_params[k] += (v.float() * weight).round().to(fused_params[k].dtype)
    
    # 最后检查参数有效性
    for key, param in fused_params.items():
        if torch.isnan(param).any() or torch.isinf(param).any():
            fused_params[key] = global_model.state_dict()[key]
            logger.warning(f"Replaced NaN/Inf in parameter {key} with global model parameter.")

    return fused_params, benigns
