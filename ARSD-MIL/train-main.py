import json
import os
import gc
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, \
    precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import math
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import random
import sys
import time

# ==========================================
# 0. 全局工具函数
# ==========================================

GLOBAL_SEED = 42
SPLIT_PROTOCOL = 'patient_stratified_rotating_outer_cv_inner_validation_full_distribution_v2'

def seed_everything(seed=44):
    global GLOBAL_SEED
    GLOBAL_SEED = int(seed)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> [Setup] Seed set to {seed}")


# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================

DATASET_SPECS = {
    'YNC_OC': {
        'h5_dir': r'E:\YNC-OC\set-E\20x_256px_0px_overlap\features_uni',
        'csv_path': r'data\YNC-OC.csv',
        'benign_h5_dir': r'E:\YNC-OC\set-L\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'J': 0, 'N': 1, 'T': 2, 'Z': 3},
        'num_classes': 4,
        'minibag_size': 50,
        'top_k_loss': 20,
        'max_minibags': 256,
        'aux_weight': 0.1,
    },
    'UBC_OC': {
        'h5_dir': r'E:\UBC\set-E\20x_256px_0px_overlap\features_uni_v1',
        'csv_path': r'data\UBC_OC_A.csv',
        'benign_h5_dir': r'E:\set\beni\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'HGSC': 0, 'LGSC': 1, 'MUC': 2, 'CCOC': 3, 'ENOC': 4},
        'num_classes': 5,
        'minibag_size': 50,
        'top_k_loss': 20,
        'max_minibags': 256,
        'aux_weight': 0.1,
    },
    'TCGA_STAD': {
        'h5_dir': r'F:\TCGA-STAD\set-E\20x_256px_0px_overlap\features_uni_v1',
        'csv_path': r'data\TCGA_STAD.csv',
        'benign_h5_dir': r'F:\TCGA-STAD\set-L\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'INT': 0, 'DIF': 1},
        'num_classes': 2,
        'minibag_size':50,
        'top_k_loss': 20,
        'max_minibags': 256,
        'aux_weight': 0.1,
    },
    'TCGA_LUNG': {
        'h5_dir': r'F:\TCGA-LUNG\set-E\20x_256px_0px_overlap\features_uni_v1',
        'csv_path': r'data\TCGA_LUNG.csv',
        'benign_h5_dir': r'F:\TCGA-Fei\set-L\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'LUSC': 0, 'LUAD': 1},
        'num_classes': 2,
        'minibag_size': 50,
        'top_k_loss': 20,
        'max_minibags': 256,
        'aux_weight': 0.1,
    },
    'TCGA_BRCA': {
        'h5_dir': r'D:\TCGA-BRCA\set-E\20x_256px_0px_overlap\features_uni_v1',
        'csv_path': r'data\TCGA-BRCA.csv',
        'benign_h5_dir': r'D:\TCGA-BRCA\set-L\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'IDC': 0, 'ILC': 1},
        'num_classes': 2,
        'minibag_size': 50, 
        'top_k_loss': 20,
        'max_minibags': 256,
        'aux_weight': 0.1,
    }
}



def get_base_config(dataset_name):
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_SPECS.keys())}")

    spec = DATASET_SPECS[dataset_name]
#top_k_loss：控制有多少个 实例 logits 参与辅助蒸馏损失
# top_k_ratio：在实例级重要性排序后，保留前多少比例的 minibag 进入 Transformer。1.0 代表全保留。
# head_dropout：分类头里的 dropout，单独控制最终分类层前的正则强度。
# ablation_settings：消融开关。
# use_agg：是否用注意力聚合器（否则退化为均值聚合）。
# use_info：是否融合坐标和分数信息。
# depth：Transformer block 层数。
# aux_weight：辅助蒸馏损失权重。0 或很小会更偏主分类损失。 YNC: 0.01, STAD: 1，LUNG:0.1
# weight_decay：AdamW 的权重衰减系数。
# batch_size：每次迭代的样本数（WSI 数量）。
# patience：早停耐心值，验证集连续多少轮不提升就停。
    config = {
        **spec,
        'dataset_name': dataset_name,
        'bank_save_path': f'./ubiquity_aware_bank_benign_only_{dataset_name}.npy',
        'feature_dim': 1024,
        'n_neighbor': 9,
        'max_memory_patches': 50000,
        'top_k_ratio': 1.0,  
        'dropout': 0.3,
        'head_dropout': 0.5,
        'weight_decay': 1e-3,
        'k_folds': 5,
        'test_size': 0.2,
        'patience': 15,
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'preprocess_device': 'auto',
        'gpu_query_batch_size': 2048,
        'gpu_center_chunk_size': 8192,
        'gpu_kmeans_steps': 100,
        'mode': 'train',
        'cache_dir': f"./{dataset_name}/processed_minibag_cache_v5_n{spec['minibag_size']}_len{spec['max_minibags']}",
        'ablation_settings': {
            'use_agg': True,
            'use_info': True,
            'depth': 2
        }
    }
    if not os.path.exists(config['cache_dir']):
        try:
            os.makedirs(config['cache_dir'], exist_ok=True)
        except Exception:
            pass
    return config


# ==========================================
# GPU preprocessing: chunked neighbors and online MiniBatch K-means
# GPU initialization and updates differ from sklearn.
# ==========================================
@torch.inference_mode()
def nearest_centers(samples, centers, center_chunk_size=8192):
    """Return Euclidean distances and assignments without a full N x K matrix."""
    if len(centers) == 0 or center_chunk_size < 1:
        raise ValueError('Centers must be nonempty and chunk size positive.')
    best_squared = torch.full((len(samples),), float('inf'), device=samples.device)
    best_indices = torch.zeros(len(samples), dtype=torch.long, device=samples.device)
    sample_norms = samples.square().sum(1, keepdim=True)
    for start in range(0, len(centers), center_chunk_size):
        block = centers[start:start + center_chunk_size]
        squared = sample_norms + block.square().sum(1).unsqueeze(0)
        squared.addmm_(samples, block.T, beta=1, alpha=-2)
        distances, indices = squared.min(dim=1)
        better = distances < best_squared
        best_squared = torch.where(better, distances, best_squared)
        best_indices = torch.where(better, indices + start, best_indices)
    # Recompute selected distances by subtraction to avoid sqrt cancellation
    # (especially for a patch that exactly matches a bank entry).
    distances = torch.linalg.vector_norm(samples - centers[best_indices], dim=1)
    return distances, best_indices


@torch.inference_mode()
def minibatch_kmeans(data, n_clusters, device, batch_size=2048,
                     steps=100, n_init=1, seed=42, center_chunk_size=8192):
    """Online centroid updates; only centers and one data batch occupy GPU RAM."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or not 1 <= n_clusters <= len(data):
        raise ValueError('Expected a feature matrix with 1 <= clusters <= samples.')
    if min(batch_size, steps, n_init, center_chunk_size) < 1:
        raise ValueError('K-means settings must be positive.')
    rng = np.random.default_rng(seed)
    validation = data[rng.choice(len(data), min(len(data), 8192), replace=False)]
    best_centers, best_inertia = None, float('inf')
    for _ in range(n_init):
        centers = torch.as_tensor(
            data[rng.choice(len(data), n_clusters, replace=False)], device=device).clone()
        counts = torch.zeros(n_clusters, device=device)
        for _ in range(steps):
            batch = torch.as_tensor(
                data[rng.choice(len(data), min(batch_size, len(data)), replace=False)],
                device=device)
            _, assignments = nearest_centers(batch, centers, center_chunk_size)
            # Aggregate only occupied clusters: avoids a K x D temporary per step.
            occupied, inverse, batch_counts = assignments.unique(
                return_inverse=True, return_counts=True)
            sums = torch.zeros((len(occupied), data.shape[1]), device=device)
            sums.index_add_(0, inverse, batch)
            old_counts = counts[occupied]
            new_counts = old_counts + batch_counts
            centers[occupied] = (
                centers[occupied] * old_counts.unsqueeze(1) + sums
            ) / new_counts.unsqueeze(1)
            counts[occupied] = new_counts
        inertia = 0.0
        for start in range(0, len(validation), batch_size):
            batch = torch.as_tensor(validation[start:start + batch_size], device=device)
            distances, _ = nearest_centers(batch, centers, center_chunk_size)
            inertia += distances.square().sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.cpu().numpy().copy()
    return best_centers


@torch.inference_mode()
def score_features(features, bank, query_batch_size=2048, center_chunk_size=8192):
    """Exact exhaustive nearest-neighbor search in FP32, streamed over patches."""
    if min(query_batch_size, center_chunk_size) < 1:
        raise ValueError('Distance batch sizes must be positive.')
    scores = torch.empty(len(features), device=bank.device)
    for start in range(0, len(features), query_batch_size):
        batch = torch.as_tensor(features[start:start + query_batch_size],
                                dtype=torch.float32, device=bank.device)
        distances, _ = nearest_centers(batch, bank, center_chunk_size)
        scores[start:start + len(batch)] = distances
    return scores


@torch.inference_mode()
def normalize_and_sort(scores):
    normalized = ((scores - scores.min()) /
                  (scores.max() - scores.min() + 1e-6)).clamp(1e-4, 1.0)
    # Match NumPy's default quicksort interface: ties have no guaranteed order.
    indices = torch.argsort(normalized, descending=True)
    return normalized.cpu().numpy(), indices.cpu().numpy()


# ==========================================
# 2. 核心模块: AbRank
# ==========================================
class AbRankScorer:
    def __init__(self, config):
        self.config = config
        self.memory_bank = None
        self.knn = None
        requested = config.get('preprocess_device', 'auto')
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GPU preprocessing requires a CUDA-enabled PyTorch installation.')
        self.preprocess_device = torch.device(requested)
        self.bank_tensor = None
        print(f'[Info] Preprocessing device: {self.preprocess_device}')

    def _cluster(self, data, n_clusters, batch_size, n_init):
        if self.preprocess_device.type == 'cuda':
            return minibatch_kmeans(
                data, n_clusters, self.preprocess_device, batch_size=batch_size,
                steps=self.config.get('gpu_kmeans_steps', 100), n_init=n_init,
                seed=GLOBAL_SEED,
                center_chunk_size=self.config.get('gpu_center_chunk_size', 8192))
        return MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                               n_init=n_init).fit(data).cluster_centers_

    def load_or_build_bank(self):
        if os.path.exists(self.config['bank_save_path']):
            print(f"[Info] Loading Memory Bank from {self.config['bank_save_path']}...")
            self.memory_bank = np.load(self.config['bank_save_path'])
        else:
            print("[Info] Building Memory Bank from benign slides...")
            self.build_bank_from_benign_dir()

        if self.preprocess_device.type == 'cuda':
            print('[Info] Loading bank onto GPU for chunked nearest-neighbor search...')
            self.bank_tensor = torch.as_tensor(
                self.memory_bank, dtype=torch.float32, device=self.preprocess_device)
        else:
            print("[Info] Fitting CPU KNN...")
            self.knn = NearestNeighbors(n_neighbors=self.config['n_neighbor'], metric='euclidean', n_jobs=-1)
            self.knn.fit(self.memory_bank)

    def build_bank_from_benign_dir(self):
        benign_dir = self.config['benign_h5_dir']
        if not os.path.exists(benign_dir):
            raise FileNotFoundError(f"Benign dir not found: {benign_dir}")

        h5_files = glob.glob(os.path.join(benign_dir, "*.h5"))
        if not h5_files:
            raise RuntimeError(f"No .h5 files found in {benign_dir}")

        target_total = self.config['max_memory_patches']
        prototypes_per_slide = max(32, int((target_total * 1.5) / len(h5_files)))
        slide_prototypes = []

        for h5_path in tqdm(h5_files, desc="Extracting Prototypes"):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'features' in f:
                        data = f['features'][:]
                        if data.shape[0] <= prototypes_per_slide:
                            slide_prototypes.append(data)
                        else:
                            slide_prototypes.append(self._cluster(
                                data, prototypes_per_slide, batch_size=2048, n_init=1))
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:
                print(f"Warning: Error reading {h5_path}: {e}")

        if not slide_prototypes:
            raise RuntimeError("Failed to extract any prototypes.")

        all_prototypes = np.concatenate(slide_prototypes, axis=0)

        if all_prototypes.shape[0] > target_total:
            self.memory_bank = self._cluster(
                all_prototypes, target_total, batch_size=4096, n_init=3)
        else:
            self.memory_bank = all_prototypes

        np.save(self.config['bank_save_path'], self.memory_bank)

    def compute_raw_scores(self, features, return_tensor=False):
        if len(features) == 0: return np.array([])
        if self.preprocess_device.type == 'cuda':
            scores = score_features(
                features, self.bank_tensor,
                query_batch_size=self.config.get('gpu_query_batch_size', 2048),
                center_chunk_size=self.config.get('gpu_center_chunk_size', 8192))
            return scores if return_tensor else scores.cpu().numpy()
        if isinstance(features, torch.Tensor): features = features.cpu().numpy()
        batch_size = 8192
        dists_list = []
        num_samples = len(features)
        for i in range(0, num_samples, batch_size):
            batch = features[i:i + batch_size]
            dists, _ = self.knn.kneighbors(batch, n_neighbors=1)
            dists_list.append(dists)
        return np.concatenate(dists_list).flatten()

    def precompute_and_cache_minibags(self, df):
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        to_process = [row for _, row in df.iterrows() if
                      not os.path.exists(os.path.join(self.config['cache_dir'], f"{row['slide_id']}.pt"))]
        if not to_process:
            print("[Info] All minibags are already cached.")
            return

        for row in tqdm(to_process, desc="Processing MiniBags"):
            slide_id = row['slide_id']
            label = self.config['label_map'].get(row['label'])
            if label is None: continue
            h5_path = os.path.join(self.config['h5_dir'], f"{slide_id}.h5")
            cache_path = os.path.join(self.config['cache_dir'], f"{slide_id}.pt")
            if not os.path.exists(h5_path): continue
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:] if 'features' in f else None
                    coords = f['coords'][:] if 'coords' in f else None
                if features is None or len(features) < self.config['minibag_size']: continue
                coords = coords.astype(np.float32)
                c_min, c_max = coords.min(axis=0), coords.max(axis=0)
                coords_norm = (coords - c_min) / (c_max - c_min + 1e-6)
                raw_dists = self.compute_raw_scores(features, return_tensor=True)
                if isinstance(raw_dists, torch.Tensor):
                    norm_scores, sort_idx = normalize_and_sort(raw_dists)
                else:
                    d_min, d_max = raw_dists.min(), raw_dists.max()
                    norm_scores = (raw_dists - d_min) / (d_max - d_min + 1e-6)
                    norm_scores = np.clip(norm_scores, 1e-4, 1.0)
                    sort_idx = np.argsort(norm_scores)[::-1]
                n_minibags = min(len(sort_idx) // self.config['minibag_size'], self.config['max_minibags'])
                if n_minibags == 0: continue
                idx_groups = sort_idx[:n_minibags * self.config['minibag_size']].reshape(n_minibags,
                                                                                         self.config['minibag_size'])
                torch.save({
                    'mb_raw_feats': torch.tensor(features[idx_groups], dtype=torch.float32),
                    'mb_raw_scores': torch.tensor(norm_scores[idx_groups], dtype=torch.float32),
                    'mb_coords': torch.tensor(coords_norm[idx_groups][:, 0, :], dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.long)
                }, cache_path)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:
                print(f'[Warning] Failed to cache {slide_id}: {e}')


# ==========================================
# 3. Dataset & Collate
# ==========================================
class MiniBagDataset(Dataset):
    def __init__(self, df, cache_dir):
        self.df = df
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        slide_id = self.df.iloc[idx]['slide_id']
        file_path = os.path.join(self.cache_dir, f"{slide_id}.pt")
        try:
            data = torch.load(file_path, weights_only=False)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load cached data for {slide_id}: {e}")


def collate_minibags(batch):
    labels = torch.stack([b['label'] for b in batch])
    max_len = max([b['mb_raw_feats'].shape[0] for b in batch])
    n_size, dim = batch[0]['mb_raw_feats'].shape[1], batch[0]['mb_raw_feats'].shape[2]
    padded_feats = torch.zeros(len(batch), max_len, n_size, dim)
    padded_scores = torch.zeros(len(batch), max_len, n_size)
    padded_coords = torch.zeros(len(batch), max_len, 2)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b['mb_raw_feats'].shape[0]
        padded_feats[i, :L] = b['mb_raw_feats']
        padded_scores[i, :L] = b['mb_raw_scores']
        padded_coords[i, :L] = b['mb_coords']
        mask[i, :L] = False
    return padded_feats, padded_scores, padded_coords, mask, labels


# ==========================================
# 4. 模型定义 (AMB_MIL）
# ==========================================
class RobustMiniBagAggregator(nn.Module):
    def __init__(self, dim, hidden_dim=128, mode='attention'):
        super().__init__()
        self.mode = mode
        if self.mode == 'attention':
            self.attn_U = nn.Linear(dim, hidden_dim)
            self.attn_V = nn.Linear(dim, hidden_dim)
            self.attn_w = nn.Linear(hidden_dim, 1)
            self.score_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, raw_scores):
        B, N, n, D = x.shape
        if self.mode == 'mean':
            return torch.mean(x, dim=2), torch.ones(B, N, device=x.device) / n
        else:
            x_flat = x.view(B * N, n, D)
            scores_flat = raw_scores.view(B * N, n, 1)
            a = torch.tanh(self.attn_U(x_flat))
            b = torch.sigmoid(self.attn_V(x_flat))
            alpha = self.attn_w(a * b)
            alpha = alpha + (self.score_gate * scores_flat)
            weights = F.softmax(alpha, dim=1)
            learned_feat = torch.sum(x_flat * weights, dim=1)
            bag_importance = torch.mean(raw_scores, dim=2)
            return learned_feat.view(B, N, D), bag_importance


class GuidedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)[0]
        return x + self.mlp(self.norm2(x))


class AMB_ViT(nn.Module):
    def __init__(self, num_classes=4, feature_dim=1024, hidden_dim=256, depth=2, heads=4, dropout=0.1, head_dropout=0.2,
                 ablation_cfg=None, top_k_ratio=1.0, min_keep=16):
        super().__init__()
        self.cfg = {'use_agg': True, 'use_info': True, 'depth': depth}
        if ablation_cfg: self.cfg.update(ablation_cfg)
        self.top_k_ratio = top_k_ratio
        self.min_keep = min_keep
        agg_mode = 'attention' if self.cfg['use_agg'] else 'mean'
        self.aggregator = RobustMiniBagAggregator(feature_dim, hidden_dim=128, mode=agg_mode)
        self.fc_feat = nn.Linear(feature_dim, hidden_dim)
        if self.cfg['use_info']:
            self.fc_coord = nn.Linear(2, hidden_dim)
            self.fc_score = nn.Linear(1, hidden_dim)
            self.ln_coord = nn.LayerNorm(hidden_dim)
            self.ln_score = nn.LayerNorm(hidden_dim)
        self.ln_feat = nn.LayerNorm(hidden_dim)
        self.pos_drop = nn.Dropout(p=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.blocks = nn.ModuleList([GuidedBlock(hidden_dim, heads, drop=dropout) for _ in range(self.cfg['depth'])])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(head_dropout),
                                  nn.Linear(hidden_dim, num_classes))
        self.instance_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_raw_bags, x_raw_scores, x_coords, mask=None):
        B, N, n, D = x_raw_bags.shape
        x_fused, bag_importance = self.aggregator(x_raw_bags, x_raw_scores)
        feat_emb = self.ln_feat(self.fc_feat(x_fused))
        if self.cfg['use_info']:
            score_mean = torch.mean(x_raw_scores, dim=2, keepdim=True)
            x_instance = feat_emb + self.ln_coord(self.fc_coord(x_coords)) + self.ln_score(self.fc_score(score_mean))
        else:
            x_instance = feat_emb
        instance_logits = self.instance_head(x_instance)
        sorted_scores, sorted_indices = torch.sort(bag_importance, dim=1, descending=True)
        idx_expanded_logits = sorted_indices.unsqueeze(-1).expand(-1, -1, instance_logits.shape[-1])
        sorted_instance_logits = torch.gather(instance_logits, 1, idx_expanded_logits)
        sorted_importance = sorted_scores
        current_k = int(N * self.top_k_ratio)
        current_k = min(N, max(current_k, self.min_keep))
        if current_k < N:
            keep_indices = sorted_indices[:, :current_k]
            idx_feat = keep_indices.unsqueeze(-1).expand(-1, -1, x_instance.shape[-1])
            x_selected = torch.gather(x_instance, 1, idx_feat)
            mask_selected = torch.gather(mask, 1, keep_indices) if mask is not None else None
        else:
            x_selected, mask_selected = x_instance, mask
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_selected), dim=1)
        full_mask = torch.cat((torch.zeros(B, 1, dtype=torch.bool, device=x.device), mask_selected),
                              dim=1) if mask_selected is not None else None
        x = self.pos_drop(x)
        for blk in self.blocks: x = blk(x, mask=full_mask)
        bag_logits = self.head(self.norm(x)[:, 0])
        return bag_logits, sorted_instance_logits, sorted_importance


# ==========================================
# 5. 测试与可视化 (Enhanced Test Function)
# ==========================================
def test_model(model, test_loader, device, label_map, dataset_name, save_dir, save_plots=True, eval_name=None):
    """
    测试模型性能并生成可视化图表
    """
    phase_name = f" ({eval_name})" if eval_name else ""
    print(f"\n>>> Evaluation Phase{phase_name}...")

    # --- 可调节字体大小参数 ---
    FONT_SIZE = 17  # 坐标轴、标签、刻度的字体大小
    TITLE_SIZE = 20  # 图表标题的字体大小
    # -----------------------

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            feats, raw_scores, coords, mask, labels = [x.to(device) for x in batch]
            logits, _, _ = model(feats, raw_scores, coords, mask)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        print("[Error] No labels found in test set!")
        return {'acc': 0, 'f1': 0, 'auc': 0}

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    # 1. 准备类别信息
    all_classes = sorted(label_map.keys(), key=lambda k: label_map[k])
    class_indices = [label_map[k] for k in all_classes]
    n_classes = len(all_classes)

    # 2. 计算 AUC (OvR)
    try:
        y_onehot = label_binarize(all_labels, classes=class_indices)
        if n_classes == 2:
            # 二分类情形下 label_binarize 返回 (N, 1)，需转换
            y_onehot = np.hstack((1 - y_onehot, y_onehot))
        auc_score = roc_auc_score(y_onehot, all_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"[Warning] AUC Calculation failed: {e}")
        auc_score = 0.0

    print(
        f"Test ACC: {acc * 100:.2f}% | "
        f"Macro-F1: {f1 * 100:.2f}% | "
        f"Macro-AUC: {auc_score * 100:.2f}%"
    )
    if not save_plots:
        return {'acc': acc * 100, 'f1': f1, 'auc': auc_score}

    # 3. 绘制混淆矩阵 (Confusion Matrix)
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.set_context("paper", rc={"font.size": FONT_SIZE, "axes.titlesize": TITLE_SIZE, "axes.labelsize": FONT_SIZE})
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_classes, yticklabels=all_classes,
                    annot_kws={"size": FONT_SIZE})
        plt.xlabel('Predicted Label', fontsize=FONT_SIZE)
        plt.ylabel('True Label', fontsize=FONT_SIZE)
        plt.xticks(fontsize=FONT_SIZE - 2)
        plt.yticks(fontsize=FONT_SIZE - 2)
        plt.title(f'Confusion Matrix - {dataset_name}', fontsize=TITLE_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{dataset_name}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[Warning] Failed to plot CM: {e}")

    # 4. 绘制 ROC 曲线 (One-vs-Rest)
    try:
        plt.figure(figsize=(10, 8))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # 计算每一类的 ROC
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算宏平均 (Macro-average) ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        # 绘制宏平均曲线
        plt.plot(all_fpr, mean_tpr,
                 label=f'Macro-average (area = {auc_score:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        # 绘制每一类的曲线
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {all_classes[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=FONT_SIZE)
        plt.ylabel('True Positive Rate', fontsize=FONT_SIZE)
        plt.title(f'ROC Curve - {dataset_name}', fontsize=TITLE_SIZE)
        plt.legend(loc="lower right", fontsize=FONT_SIZE - 2)
        plt.xticks(fontsize=FONT_SIZE - 2)
        plt.yticks(fontsize=FONT_SIZE - 2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'roc_curve_{dataset_name}.png'), dpi=300)
        plt.close()
        print(f"[Info] ROC Curve saved to {save_dir}/roc_curve_{dataset_name}.png")
    except Exception as e:
        print(f"[Warning] Failed to plot ROC: {e}")

    return {'acc': acc * 100, 'f1': f1, 'auc': auc_score}


def save_cv_records(meta_path, fold_records):
    payload = {
        'fold_records': [
            dict(
                {
                    'fold': int(record['fold']),
                    'val_acc': float(record['val_acc']),
                    'val_loss': float(record['val_loss']),
                    'model_path': record['model_path'],
                    'split_protocol': record.get('split_protocol'),
                    **(
                        {'val_auc': float(record['val_auc'])}
                        if 'val_auc' in record else {}
                    ),
                    **(
                        {'val_f1': float(record['val_f1'])}
                        if 'val_f1' in record else {}
                    )
                },
                **(
                    {
                        'test_acc': float(record['test_acc']),
                        'test_f1': float(record['test_f1']),
                        'test_auc': float(record['test_auc'])
                    } if all(k in record for k in ('test_acc', 'test_f1', 'test_auc')) else {}
                )
            )
            for record in fold_records
        ]
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def load_cv_records(meta_path):
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload.get('fold_records', [])
    except Exception as e:
        print(f"[Warning] Failed to load CV metadata: {e}")
        return []


def evaluate_fold_models(fold_records, config, fold_splits, save_dir):
    available_records = [record for record in fold_records if os.path.exists(record['model_path'])]
    if not available_records:
        print("[Warning] No saved fold models found for fold evaluation.")
        return None

    test_frames = {int(fold): test_df for fold, _, _, test_df in fold_splits}
    expected_protocol = SPLIT_PROTOCOL
    incompatible_records = [
        int(record['fold']) for record in available_records
        if record.get('split_protocol') != expected_protocol
    ]
    if incompatible_records:
        raise ValueError(
            "Saved fold metadata was not produced by the current rotating-test protocol. "
            f"Retrain these folds with the new split before testing: {incompatible_records}"
        )

    missing_test_metrics = any(
        not all(key in record for key in ('test_acc', 'test_f1', 'test_auc'))
        for record in available_records
    )

    if missing_test_metrics:
        for record in sorted(available_records, key=lambda item: int(item['fold'])):
            print(
                f"\n>>> Evaluating Fold {record['fold']} | "
                f"Best Val: {float(record['val_acc']):.2f}% | "
                f"Loss: {float(record.get('val_loss', float('nan'))):.4f}"
            )
            model = AMB_ViT(
                num_classes=config['num_classes'],
                feature_dim=config['feature_dim'],
                ablation_cfg=config['ablation_settings'],
                top_k_ratio=config['top_k_ratio']
            ).to(config['device'])
            model.load_state_dict(torch.load(record['model_path'], weights_only=True, map_location=config['device']))

            fold_number = int(record['fold'])
            if fold_number not in test_frames:
                raise ValueError(f"No held-out test split found for fold {fold_number}.")
            test_loader = DataLoader(
                MiniBagDataset(test_frames[fold_number], config['cache_dir']),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                collate_fn=collate_minibags,
                pin_memory=True
            )
            metrics = test_model(
                model,
                test_loader,
                config['device'],
                config['label_map'],
                config['dataset_name'],
                save_dir,
                save_plots=False,
                eval_name=f"Fold {record['fold']}"
            )
            record['test_acc'] = float(metrics['acc'])
            record['test_f1'] = float(metrics['f1'])
            record['test_auc'] = float(metrics['auc'])

    # The representative fold used for plots is selected from validation results only.
    # Test metrics are never used for checkpoint or representative-fold selection.
    ranked_records = sorted(
        available_records,
        key=lambda record: (
            -float(record['val_acc']),
            float(record.get('val_loss', float('inf'))),
            int(record['fold'])
        )
    )
    best_record = ranked_records[0]

    print(
        f"\n>>> Rendering Plots From Validation-Selected Fold {best_record['fold']} | "
        f"Val Acc: {float(best_record['val_acc']):.2f}%"
    )
    best_model = AMB_ViT(
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim'],
        ablation_cfg=config['ablation_settings'],
        top_k_ratio=config['top_k_ratio']
    ).to(config['device'])
    best_model.load_state_dict(torch.load(best_record['model_path'], weights_only=True, map_location=config['device']))
    best_fold_number = int(best_record['fold'])
    best_test_loader = DataLoader(
        MiniBagDataset(test_frames[best_fold_number], config['cache_dir']),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_minibags,
        pin_memory=True
    )
    best_metrics = test_model(
        best_model,
        best_test_loader,
        config['device'],
        config['label_map'],
        config['dataset_name'],
        save_dir,
        save_plots=True,
        eval_name=f"Best Fold {best_record['fold']}"
    )
    best_record['test_acc'] = float(best_metrics['acc'])
    best_record['test_f1'] = float(best_metrics['f1'])
    best_record['test_auc'] = float(best_metrics['auc'])

    all_records = sorted(available_records, key=lambda record: int(record['fold']))
    accs = np.array([float(record['test_acc']) for record in all_records], dtype=np.float64)
    f1s = np.array([float(record['test_f1']) for record in all_records], dtype=np.float64)
    aucs = np.array([float(record['test_auc']) for record in all_records], dtype=np.float64)

    summary = {
        'acc': float(np.mean(accs)),
        'acc_std': float(np.std(accs)),
        'f1': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'auc': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'best_fold': int(best_record['fold']),
        'split_protocol': expected_protocol,
        'folds': [int(record['fold']) for record in all_records],
        'fold_records': available_records,
    }

    print(f"\n>>> {len(all_records)}-Fold Test Summary")
    print(f"    Folds: {summary['folds']}")
    print(f"    ACC:      {summary['acc']:.2f}% (+/- {summary['acc_std']:.2f})")
    print(f"    Macro-F1: {summary['f1'] * 100:.2f}% (+/- {summary['f1_std'] * 100:.2f})")
    print(f"    Macro-AUC: {summary['auc'] * 100:.2f}% (+/- {summary['auc_std'] * 100:.2f})")
    print(f"    Validation-selected plot source fold: {summary['best_fold']}")

    return summary


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _normalise_split_columns(df):
    required_columns = ['case_id', 'slide_id', 'label']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required split columns: {missing_columns}")

    split_df = df.copy()
    if split_df[required_columns].isnull().any().any():
        null_counts = split_df[required_columns].isnull().sum().to_dict()
        raise ValueError(f"Null values found in split columns: {null_counts}")
    for column in required_columns:
        split_df[column] = split_df[column].astype(str).str.strip()
        if (split_df[column] == '').any():
            raise ValueError(f"Empty values found in split column '{column}'.")
    if split_df['slide_id'].duplicated().any():
        duplicates = split_df.loc[split_df['slide_id'].duplicated(keep=False), 'slide_id'].unique().tolist()
        raise ValueError(f"Duplicate slide_id values found: {duplicates[:20]}")
    return split_df


def _build_case_table(df):
    case_rows = []
    mixed_label_cases = []

    for case_id, case_df in df.groupby('case_id', sort=True):
        label_counts = case_df['label'].value_counts()
        if len(label_counts) > 1:
            mixed_label_cases.append((case_id, label_counts.to_dict()))

        max_count = int(label_counts.max())
        majority_labels = sorted(label_counts[label_counts == max_count].index.astype(str).tolist())
        if len(majority_labels) != 1:
            raise ValueError(
                f"Case '{case_id}' has a tied majority label and cannot be stratified automatically: "
                f"{label_counts.to_dict()}"
            )
        case_rows.append({
            'case_id': str(case_id),
            'strat_label': majority_labels[0],
            'n_slides': int(len(case_df))
        })

    if mixed_label_cases:
        print(f"[Warning] Found {len(mixed_label_cases)} case(s) with multiple WSI labels.")
        for case_id, label_counts in mixed_label_cases:
            print(f"          case_id={case_id}, label_counts={label_counts}; majority label is used only for stratification.")

    return pd.DataFrame(case_rows)


def _split_paths(config):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = os.path.join(project_dir, 'data', 'split')
    dataset_name = config['dataset_name']
    split_path = os.path.join(
        split_dir,
        f"{dataset_name}_outercv{config['k_folds']}_inner20_fullprop_seed{GLOBAL_SEED}.csv"
    )
    return split_dir, split_path


def _warn_for_small_classes(case_table, n_splits, context):
    class_counts = case_table['strat_label'].value_counts().sort_index()
    too_small = class_counts[class_counts < n_splits]
    if not too_small.empty:
        print(
            f"[Warning] {context}: the following patient-level classes have fewer than {n_splits} cases, "
            f"so presence in every split cannot be guaranteed: {too_small.to_dict()}"
        )
    return class_counts


def _generate_stratified_group_splits(case_table, n_splits, random_state):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    return list(splitter.split(
        case_table,
        case_table['strat_label'],
        groups=case_table['case_id']
    ))


def _choose_outer_cv(case_table, config):
    n_splits = int(config['k_folds'])
    if n_splits < 2:
        raise ValueError("k_folds must be at least 2.")
    if len(case_table) < n_splits:
        raise ValueError(f"Only {len(case_table)} patients are available for {n_splits}-fold CV.")

    class_counts = _warn_for_small_classes(case_table, n_splits, 'Outer CV')
    all_labels = sorted(class_counts.index.astype(str).tolist())
    full_props = class_counts.reindex(all_labels, fill_value=0) / len(case_table)
    target_test_size = len(case_table) / n_splits
    candidates = []
    n_candidates = 128

    for candidate_index in range(n_candidates):
        splits = _generate_stratified_group_splits(
            case_table, n_splits, GLOBAL_SEED + candidate_index
        )
        test_deviations, train_deviations, test_sizes = [], [], []
        missing_test_total = 0
        for train_idx, test_idx in splits:
            train_cases = case_table.iloc[train_idx]
            test_cases = case_table.iloc[test_idx]
            train_counts = train_cases['strat_label'].value_counts().reindex(all_labels, fill_value=0)
            test_counts = test_cases['strat_label'].value_counts().reindex(all_labels, fill_value=0)
            missing_test_total += int((test_counts == 0).sum())
            test_deviations.append(float((test_counts / len(test_cases) - full_props).abs().max()))
            train_deviations.append(float((train_counts / len(train_cases) - full_props).abs().max()))
            test_sizes.append(len(test_cases))
        priority_key = (
            missing_test_total,
            max(test_deviations),
            sum(test_deviations),
            max(train_deviations),
            sum(abs(size - target_test_size) for size in test_sizes),
            candidate_index
        )
        candidates.append((priority_key, candidate_index, splits))

    _, selected_candidate, selected_splits = min(candidates, key=lambda item: item[0])
    print(
        f"[Info] Selected outer {n_splits}-fold candidate "
        f"{selected_candidate + 1}/{n_candidates} using patient-level class balance only."
    )
    return selected_splits, class_counts, all_labels


def _choose_inner_validation(
        outer_train_cases,
        validation_size,
        fold_index,
        full_dataset_props,
        full_dataset_labels):
    n_inner_splits = int(round(1.0 / float(validation_size)))
    if n_inner_splits < 2 or not np.isclose(1.0 / n_inner_splits, float(validation_size)):
        raise ValueError(
            f"validation_size={validation_size} cannot be represented by StratifiedGroupKFold."
        )
    if len(outer_train_cases) < n_inner_splits:
        raise ValueError(
            f"Fold {fold_index}: only {len(outer_train_cases)} outer-training patients are available "
            f"for an inner {n_inner_splits}-way split."
        )

    class_counts = _warn_for_small_classes(
        outer_train_cases, n_inner_splits, f'Fold {fold_index} inner validation'
    )
    all_labels = list(full_dataset_labels)
    reference_props = full_dataset_props.reindex(all_labels, fill_value=0)
    target_val_size = len(outer_train_cases) * float(validation_size)
    candidates = []
    n_candidates = 128

    for candidate_index in range(n_candidates):
        splits = _generate_stratified_group_splits(
            outer_train_cases,
            n_inner_splits,
            GLOBAL_SEED + fold_index * 10000 + candidate_index
        )
        for inner_fold_index, (train_idx, val_idx) in enumerate(splits):
            train_cases = outer_train_cases.iloc[train_idx]
            val_cases = outer_train_cases.iloc[val_idx]
            train_counts = train_cases['strat_label'].value_counts().reindex(all_labels, fill_value=0)
            val_counts = val_cases['strat_label'].value_counts().reindex(all_labels, fill_value=0)
            priority_key = (
                abs(len(val_cases) - target_val_size),
                float((val_counts / len(val_cases) - reference_props).abs().max()),
                float((val_counts / len(val_cases) - reference_props).abs().sum()),
                float((train_counts / len(train_cases) - reference_props).abs().max()),
                float((train_counts / len(train_cases) - reference_props).abs().sum()),
                int((val_counts == 0).sum()) + int((train_counts == 0).sum()),
                candidate_index,
                inner_fold_index
            )
            candidates.append((priority_key, candidate_index, inner_fold_index, train_idx, val_idx))

    _, selected_candidate, selected_inner_fold, train_idx, val_idx = min(
        candidates, key=lambda item: item[0]
    )
    print(
        f"[Info] Fold {fold_index}: selected inner validation candidate "
        f"{selected_candidate + 1}/{n_candidates}, inner fold {selected_inner_fold + 1}/{n_inner_splits}."
    )
    return train_idx, val_idx, class_counts, all_labels


def _build_nested_cv_manifest(df, config):
    case_table = _build_case_table(df)
    outer_splits, full_class_counts, full_labels = _choose_outer_cv(case_table, config)
    full_dataset_props = full_class_counts.reindex(full_labels, fill_value=0) / len(case_table)
    cv_parts = []

    for fold_index, (outer_train_idx, test_idx) in enumerate(outer_splits, start=1):
        outer_train_cases = case_table.iloc[outer_train_idx].reset_index(drop=True)
        test_cases = case_table.iloc[test_idx]
        inner_train_idx, val_idx, inner_counts, inner_labels = _choose_inner_validation(
            outer_train_cases,
            config['test_size'],
            fold_index,
            full_dataset_props,
            full_labels
        )
        train_case_ids = set(outer_train_cases.iloc[inner_train_idx]['case_id'])
        val_case_ids = set(outer_train_cases.iloc[val_idx]['case_id'])
        test_case_ids = set(test_cases['case_id'])

        test_counts = test_cases['strat_label'].value_counts().reindex(full_labels, fill_value=0)
        missing_test = test_counts[test_counts == 0].index.tolist()
        if missing_test:
            print(
                f"[Warning] Fold {fold_index} test set is missing classes {missing_test}; "
                f"full patient counts={full_class_counts.to_dict()}"
            )
        val_case_table = outer_train_cases.iloc[val_idx]
        val_counts = val_case_table['strat_label'].value_counts().reindex(inner_labels, fill_value=0)
        missing_val = val_counts[val_counts == 0].index.tolist()
        if missing_val:
            print(
                f"[Warning] Fold {fold_index} validation set is missing classes {missing_val}; "
                f"outer-training patient counts={inner_counts.to_dict()}"
            )

        fold_df = df[['case_id', 'slide_id', 'label']].copy()
        fold_df['fold'] = fold_index
        fold_df['role'] = 'train'
        fold_df.loc[fold_df['case_id'].isin(val_case_ids), 'role'] = 'val'
        fold_df.loc[fold_df['case_id'].isin(test_case_ids), 'role'] = 'test'
        assigned_train = set(fold_df.loc[fold_df['role'] == 'train', 'case_id'])
        if assigned_train != train_case_ids:
            raise RuntimeError(f"Fold {fold_index} train case assignment is inconsistent.")
        cv_parts.append(fold_df[['fold', 'case_id', 'slide_id', 'label', 'role']])

    return pd.concat(cv_parts, ignore_index=True)


def _triples_with_counts(df):
    return df.groupby(['case_id', 'slide_id', 'label'], dropna=False).size().sort_index()


def _validate_patient_splits(df, cv_df, config):
    expected_cv_columns = ['fold', 'case_id', 'slide_id', 'label', 'role']
    if list(cv_df.columns) != expected_cv_columns:
        raise ValueError(f"CV split columns must be exactly {expected_cv_columns}; got {list(cv_df.columns)}")

    cv_df = cv_df.copy()
    for column in ['case_id', 'slide_id', 'label', 'role']:
        cv_df[column] = cv_df[column].astype(str).str.strip()
    cv_df['fold'] = pd.to_numeric(cv_df['fold'], errors='raise').astype(int)

    expected_folds = set(range(1, config['k_folds'] + 1))
    if set(cv_df['fold']) != expected_folds:
        raise ValueError(f"CV split folds must be {sorted(expected_folds)}; got {sorted(cv_df['fold'].unique())}")
    if set(cv_df['role']) != {'train', 'val', 'test'}:
        raise ValueError(f"CV split role values are invalid: {sorted(cv_df['role'].unique())}")

    for fold_index in sorted(expected_folds):
        fold_df = cv_df[cv_df['fold'] == fold_index]
        if not _triples_with_counts(df).equals(_triples_with_counts(fold_df)):
            raise ValueError(f"Fold {fold_index} rows do not exactly match the current dataset.")
        if (fold_df.groupby('case_id')['role'].nunique() > 1).any():
            bad_cases = fold_df.groupby('case_id')['role'].nunique().loc[lambda values: values > 1].index.tolist()
            raise ValueError(f"Fold {fold_index} assigns a case to multiple subsets: {bad_cases[:20]}")

        train_case_ids = set(fold_df.loc[fold_df['role'] == 'train', 'case_id'])
        val_case_ids = set(fold_df.loc[fold_df['role'] == 'val', 'case_id'])
        test_case_ids = set(fold_df.loc[fold_df['role'] == 'test', 'case_id'])
        overlaps = {
            'train-val': train_case_ids & val_case_ids,
            'train-test': train_case_ids & test_case_ids,
            'val-test': val_case_ids & test_case_ids
        }
        non_empty = {name: sorted(values) for name, values in overlaps.items() if values}
        if non_empty:
            raise ValueError(f"Fold {fold_index} patient overlap detected: {non_empty}")

    test_assignments = cv_df.loc[cv_df['role'] == 'test', ['fold', 'case_id']].drop_duplicates()
    test_counts_per_case = test_assignments.groupby('case_id')['fold'].nunique()
    all_case_ids = set(df['case_id'])
    if set(test_counts_per_case.index) != all_case_ids or not (test_counts_per_case == 1).all():
        raise ValueError("Every case_id must occur in the test subset of exactly one outer fold.")

    return cv_df


def _class_split_summary(subset_df, case_label_map, label_order):
    subset_case_ids = set(subset_df['case_id'])
    patient_counts = pd.Series(
        [case_label_map[case_id] for case_id in subset_case_ids], dtype='object'
    ).value_counts()
    wsi_counts = subset_df['label'].value_counts()
    return {
        label: {
            'patients': int(patient_counts.get(label, 0)),
            'WSIs': int(wsi_counts.get(label, 0))
        }
        for label in label_order
    }


def _print_patient_split_report(df, cv_df, case_table, config):
    case_label_map = dict(zip(case_table['case_id'], case_table['strat_label']))
    label_order = [label for label in config['label_map'] if label in set(case_table['strat_label'])]
    overall_props = case_table['strat_label'].value_counts(normalize=True).reindex(label_order, fill_value=0)

    print("\n========== Patient-level nested split check ==========")
    print(f"Dataset: {config['dataset_name']} | seed={GLOBAL_SEED}")
    print(f"Total: patients={df['case_id'].nunique()}, WSIs={len(df)}")
    for fold_index in range(1, config['k_folds'] + 1):
        fold_df = cv_df[cv_df['fold'] == fold_index]
        train_df = fold_df[fold_df['role'] == 'train']
        val_df = fold_df[fold_df['role'] == 'val']
        test_df = fold_df[fold_df['role'] == 'test']
        train_case_ids = set(train_df['case_id'])
        val_case_ids = set(val_df['case_id'])
        test_case_ids = set(test_df['case_id'])
        overlaps = {
            'train-val': len(train_case_ids & val_case_ids),
            'train-test': len(train_case_ids & test_case_ids),
            'val-test': len(val_case_ids & test_case_ids)
        }
        train_summary = _class_split_summary(train_df, case_label_map, label_order)
        val_summary = _class_split_summary(val_df, case_label_map, label_order)
        test_summary = _class_split_summary(test_df, case_label_map, label_order)
        print(
            f"Fold {fold_index}: train patients={len(train_case_ids)}, WSIs={len(train_df)}; "
            f"val patients={len(val_case_ids)}, WSIs={len(val_df)}; "
            f"test patients={len(test_case_ids)}, WSIs={len(test_df)}"
        )
        print(f"  Train class counts (patient/WSI):      {train_summary}")
        print(f"  Validation class counts (patient/WSI): {val_summary}")
        print(f"  Test class counts (patient/WSI):       {test_summary}")
        for role_name, subset_df, summary in (
                ('train', train_df, train_summary),
                ('validation', val_df, val_summary),
                ('test', test_df, test_summary)):
            patient_total = max(1, subset_df['case_id'].nunique())
            props = pd.Series({label: values['patients'] / patient_total for label, values in summary.items()})
            max_deviation = float((props - overall_props).abs().max())
            print(f"  {role_name} max patient-class proportion deviation: {max_deviation:.2%}")
        print(f"  Patient overlaps: {overlaps}")
    print("=======================================================\n")


def prepare_patient_level_splits(df, config, print_report=True):
    df = _normalise_split_columns(df)
    case_table = _build_case_table(df)
    split_dir, split_path = _split_paths(config)

    if os.path.exists(split_path):
        print(f"[Info] Reusing saved rotating-test CV split: {split_path}")
        cv_df = pd.read_csv(split_path, dtype=str)
    else:
        cv_df = _build_nested_cv_manifest(df, config)
        os.makedirs(split_dir, exist_ok=True)
        cv_df = _validate_patient_splits(df, cv_df, config)
        cv_df.to_csv(split_path, index=False)
        print(f"[Info] Saved rotating-test CV split: {split_path}")

    cv_df = _validate_patient_splits(df, cv_df, config)
    if print_report:
        _print_patient_split_report(df, cv_df, case_table, config)

    fold_splits = []
    for fold_index in range(1, config['k_folds'] + 1):
        fold_df = cv_df[cv_df['fold'] == fold_index]
        train_df = fold_df[fold_df['role'] == 'train'][['case_id', 'slide_id', 'label']].copy()
        val_df = fold_df[fold_df['role'] == 'val'][['case_id', 'slide_id', 'label']].copy()
        test_df = fold_df[fold_df['role'] == 'test'][['case_id', 'slide_id', 'label']].copy()
        fold_splits.append((fold_index, train_df, val_df, test_df))

    return fold_splits


# ==========================================
# 6. 主程序 pipeline
# ==========================================

def run_pipeline(CONFIG):
    print(f"\n{'=' * 40}\nPipeline Start | Dataset: {CONFIG['dataset_name']} | Mode: {CONFIG['mode']}\n{'=' * 40}")

    # 定义模型保存文件夹路径 (根据数据集名称)
    save_dir = CONFIG['dataset_name']
    os.makedirs(save_dir, exist_ok=True)
    global_model_path = os.path.join(save_dir, 'best_model_global.pth')
    cv_meta_path = os.path.join(save_dir, 'cv_fold_records.json')
    validation_only_search = bool(CONFIG.get('validation_only_search', False))
    if validation_only_search and CONFIG['mode'] != 'train':
        raise ValueError("validation_only_search is only supported in train mode.")

    if CONFIG['mode'] == 'train':
        print(f"[Info] Training mode: cleaning old models in {save_dir}...")
        for f in glob.glob(os.path.join(save_dir, "best_model_fold_*.pth")) + [global_model_path]:
            try:
                os.remove(f)
            except OSError:
                pass
        if os.path.exists(cv_meta_path):
            try:
                os.remove(cv_meta_path)
            except OSError:
                pass
    elif CONFIG['mode'] == 'test':
        if not os.path.exists(cv_meta_path) and not os.path.exists(global_model_path):
            print(f"[Error] Neither '{cv_meta_path}' nor '{global_model_path}' was found.")
            return None

    df = pd.read_csv(CONFIG['csv_path'])
    scorer = AbRankScorer(CONFIG)
    scorer.load_or_build_bank()
    scorer.precompute_and_cache_minibags(df)
    # The bank is not needed by training; release its GPU storage first.
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if CONFIG['mode'] == 'preprocess':
        print('[Info] Preprocessing complete.')
        return None

    valid_indices = [idx for idx, row in df.iterrows() if
                     os.path.exists(os.path.join(CONFIG['cache_dir'], f"{row['slide_id']}.pt"))]
    df = df.iloc[valid_indices].reset_index(drop=True)
    if len(df) == 0: return None

    fold_splits = prepare_patient_level_splits(
        df,
        CONFIG,
        print_report=not validation_only_search
    )
    if validation_only_search:
        print("[Info] Validation-only hyperparameter search: held-out test-fold evaluation is disabled.")

    cv_scores = []
    cv_auc_scores = []
    cv_f1_scores = []
    fold_records = []
    global_best_acc = 0.0
    global_best_loss = float('inf')

    if CONFIG['mode'] == 'train':
        scaler = torch.amp.GradScaler('cuda')
        for fold_number, fold_train_df, fold_val_df, fold_test_df in fold_splits:
            fold = fold_number - 1
            print(f"\n--- Fold {fold_number}/{CONFIG['k_folds']} ---")
            train_loader = DataLoader(MiniBagDataset(fold_train_df, CONFIG['cache_dir']),
                                      batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4,
                                      collate_fn=collate_minibags, pin_memory=True)
            val_loader = DataLoader(MiniBagDataset(fold_val_df, CONFIG['cache_dir']),
                                    batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4,
                                    collate_fn=collate_minibags, pin_memory=True)
            model = AMB_ViT(num_classes=CONFIG['num_classes'], feature_dim=CONFIG['feature_dim'],
                            dropout=CONFIG['dropout'], ablation_cfg=CONFIG['ablation_settings'],
                            top_k_ratio=CONFIG['top_k_ratio']).to(CONFIG['device'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
            criterion, kl_criterion = nn.CrossEntropyLoss(), nn.KLDivLoss(reduction='none')
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(
                len(train_loader) * CONFIG['epochs'] * 0.1), num_training_steps=len(train_loader) * CONFIG['epochs'])
            best_fold_acc, best_fold_auc, best_fold_f1, best_fold_loss, patience_counter = (
                0.0, 0.0, 0.0, float('inf'), 0
            )
            fold_model_path = os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth')
            for epoch in range(CONFIG['epochs']):
                model.train()
                total_loss, start_time = 0, time.time()
                pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{CONFIG['epochs']}", leave=True)
                for batch in pbar:
                    feats, raw_scores, coords, mask, labels = [x.to(CONFIG['device']) for x in batch]
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        bag_logits, sorted_instance_logits, sorted_importance = model(feats, raw_scores, coords, mask)
                        loss_main = criterion(bag_logits, labels)
                        distill_type = CONFIG.get('distill_type', 'D')
                        if distill_type == 'A' or CONFIG['aux_weight'] <= 0:
                            loss = loss_main
                        else:
                            actual_k = sorted_instance_logits.shape[1] if distill_type == 'B' else min(
                                CONFIG['top_k_loss'], sorted_instance_logits.shape[1])
                            top_k_inst_logits = sorted_instance_logits[:, :actual_k, :]
                            teacher_target = torch.softmax(bag_logits, dim=1).detach().unsqueeze(1).expand(-1, actual_k,
                                                                                                           -1)
                            kl_div = kl_criterion(torch.log_softmax(top_k_inst_logits, dim=2), teacher_target).sum(
                                dim=-1)
                            weighted_kl = (kl_div * sorted_importance[
                                :, :actual_k]).mean() if distill_type == 'D' else kl_div.mean()
                            loss = loss_main + CONFIG['aux_weight'] * weighted_kl
                    scaler.scale(loss).backward();
                    scaler.step(optimizer);
                    scaler.update();
                    scheduler.step()
                    total_loss += loss.item();
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                model.eval();
                correct, total = 0, 0
                val_labels, val_probs = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        feats, raw_scores, coords, mask, labels = [x.to(CONFIG['device']) for x in batch]
                        with torch.amp.autocast('cuda'): logits, _, _ = model(feats, raw_scores, coords, mask)
                        correct += (logits.argmax(dim=1) == labels).sum().item();
                        total += labels.size(0)
                        val_labels.extend(labels.cpu().numpy())
                        val_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
                epoch_loss = total_loss / len(train_loader)
                val_acc = 100 * correct / total
                val_labels = np.asarray(val_labels)
                val_probs = np.asarray(val_probs)
                val_preds = np.argmax(val_probs, axis=1)
                _, _, val_f1, _ = precision_recall_fscore_support(
                    val_labels,
                    val_preds,
                    average='macro',
                    zero_division=0
                )
                class_indices = sorted(CONFIG['label_map'].values())
                try:
                    val_onehot = label_binarize(val_labels, classes=class_indices)
                    if len(class_indices) == 2:
                        val_onehot = np.hstack((1 - val_onehot, val_onehot))
                    val_auc = roc_auc_score(
                        val_onehot,
                        val_probs,
                        multi_class='ovr',
                        average='macro'
                    )
                except Exception as e:
                    print(f"[Warning] Validation AUC calculation failed for fold {fold + 1}: {e}")
                    val_auc = 0.0
                msg = f"Epoch [{epoch + 1:03d}/{CONFIG['epochs']}] | Time: {time.time() - start_time:.1f}s | Loss: {epoch_loss:.4f} | Val: {val_acc:.2f}%"
                is_val_improved = val_acc > best_fold_acc
                is_fold_best = (
                    is_val_improved or
                    (np.isclose(val_acc, best_fold_acc) and epoch_loss < best_fold_loss)
                )
                if is_fold_best:
                    best_fold_acc = val_acc
                    best_fold_auc = val_auc
                    best_fold_f1 = val_f1
                    best_fold_loss = epoch_loss
                    torch.save(model.state_dict(), fold_model_path)
                    is_global_best = (
                        best_fold_acc > global_best_acc or
                        (np.isclose(best_fold_acc, global_best_acc) and best_fold_loss < global_best_loss)
                    )
                    if is_global_best:
                        global_best_acc = best_fold_acc
                        global_best_loss = best_fold_loss
                        if os.path.abspath(fold_model_path) != os.path.abspath(global_model_path):
                            shutil.copy(fold_model_path, global_model_path)
                    if is_val_improved:
                        patience_counter = 0
                        msg += " [BEST-VAL]"
                    else:
                        patience_counter += 1
                        msg += " [BEST-LOSS]"
                else:
                    patience_counter += 1
                print(msg)
                if patience_counter >= CONFIG['patience']: break
            cv_scores.append(best_fold_acc)
            cv_auc_scores.append(best_fold_auc)
            cv_f1_scores.append(best_fold_f1)
            if os.path.exists(fold_model_path):
                fold_record = {
                    'fold': fold + 1,
                    'val_acc': float(best_fold_acc),
                    'val_auc': float(best_fold_auc),
                    'val_f1': float(best_fold_f1),
                    'val_loss': float(best_fold_loss),
                    'model_path': os.path.abspath(fold_model_path),
                    'split_protocol': SPLIT_PROTOCOL
                }
                if not validation_only_search:
                    test_loader = DataLoader(
                        MiniBagDataset(fold_test_df, CONFIG['cache_dir']),
                        batch_size=CONFIG['batch_size'],
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_minibags,
                        pin_memory=True
                    )
                    fold_best_model = AMB_ViT(
                        num_classes=CONFIG['num_classes'],
                        feature_dim=CONFIG['feature_dim'],
                        ablation_cfg=CONFIG['ablation_settings'],
                        top_k_ratio=CONFIG['top_k_ratio']
                    ).to(CONFIG['device'])
                    fold_best_model.load_state_dict(
                        torch.load(fold_model_path, weights_only=True, map_location=CONFIG['device'])
                    )
                    fold_test_metrics = test_model(
                        fold_best_model,
                        test_loader,
                        CONFIG['device'],
                        CONFIG['label_map'],
                        CONFIG['dataset_name'],
                        save_dir,
                        save_plots=False,
                        eval_name=f"Fold {fold + 1}"
                    )
                    fold_record.update({
                        'test_acc': float(fold_test_metrics['acc']),
                        'test_f1': float(fold_test_metrics['f1']),
                        'test_auc': float(fold_test_metrics['auc'])
                    })
                fold_records.append(fold_record)
                save_cv_records(cv_meta_path, fold_records)

    metrics = None
    if CONFIG['mode'] == 'train':
        if validation_only_search:
            metrics = {}
        elif fold_records:
            save_cv_records(cv_meta_path, fold_records)
            metrics = evaluate_fold_models(fold_records, CONFIG, fold_splits, save_dir)
            if metrics is not None and 'fold_records' in metrics:
                save_cv_records(cv_meta_path, metrics['fold_records'])
        elif os.path.exists(global_model_path):
            raise ValueError(
                "A single global model cannot represent rotating held-out test folds. "
                "Fold-specific checkpoints and CV metadata are required."
            )

        if cv_scores:
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            cv_auc_mean = float(np.mean(cv_auc_scores))
            cv_auc_std = float(np.std(cv_auc_scores))
            cv_f1_mean = float(np.mean(cv_f1_scores))
            cv_f1_std = float(np.std(cv_f1_scores))

            print(f"\nCV Mean Acc: {cv_mean:.2f}% (+/- {cv_std:.2f})")
            print(f"CV Mean AUC: {cv_auc_mean * 100:.2f}% (+/- {cv_auc_std * 100:.2f})")
            print(f"CV Mean F1:  {cv_f1_mean * 100:.2f}% (+/- {cv_f1_std * 100:.2f})")

            if metrics is None:
                metrics = {}

            metrics['cv_mean'] = cv_mean
            metrics['cv_std'] = cv_std
            metrics['cv_auc_mean'] = cv_auc_mean
            metrics['cv_auc_std'] = cv_auc_std
            metrics['cv_f1_mean'] = cv_f1_mean
            metrics['cv_f1_std'] = cv_f1_std
    else:
        saved_records = load_cv_records(cv_meta_path)
        if saved_records:
            metrics = evaluate_fold_models(saved_records, CONFIG, fold_splits, save_dir)
            if metrics is not None and 'fold_records' in metrics:
                save_cv_records(cv_meta_path, metrics['fold_records'])
        elif os.path.exists(global_model_path):
            raise ValueError(
                "A single global model cannot be tested under rotating held-out folds. "
                "Fold-specific checkpoints and CV metadata are required."
            )

    return metrics



def main():
    parser = argparse.ArgumentParser(description="AbRank Deep Learning Pipeline")
    parser.add_argument('--dataset', type=str, default='TCGA_BRCA', choices=['YNC_OC', 'UBC_OC','TCGA_STAD','TCGA_LUNG','TCGA_BRCA'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'preprocess'])
    parser.add_argument('--preprocess-device', default='cuda', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--gpu-query-batch-size', type=int, default=2048)
    parser.add_argument('--gpu-center-chunk-size', type=int, default=8192)
    parser.add_argument('--gpu-kmeans-steps', type=int, default=100)
    parser.add_argument('--preprocess-tag', default=None,
                        help='Use a separate bank/cache version, e.g. gpu_v1; existing files are retained.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--distill_type', type=str, default='D', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--aux_weight', type=float, default=None)
    args = parser.parse_args()
    if min(args.gpu_query_batch_size, args.gpu_center_chunk_size, args.gpu_kmeans_steps) < 1:
        parser.error('GPU batch sizes and K-means steps must be positive.')
    if args.preprocess_tag is not None and (
            not args.preprocess_tag or
            any(not (char.isalnum() or char in '_-') for char in args.preprocess_tag)):
        parser.error('--preprocess-tag must contain only letters, numbers, underscores or hyphens.')
    seed_everything(args.seed)
    config = get_base_config(args.dataset)
    config.update({'mode': args.mode, 'distill_type': args.distill_type})
    config.update({
        'preprocess_device': args.preprocess_device,
        'gpu_query_batch_size': args.gpu_query_batch_size,
        'gpu_center_chunk_size': args.gpu_center_chunk_size,
        'gpu_kmeans_steps': args.gpu_kmeans_steps,
    })
    if args.preprocess_tag:
        bank_root, bank_extension = os.path.splitext(config['bank_save_path'])
        config['bank_save_path'] = f'{bank_root}_{args.preprocess_tag}{bank_extension}'
        config['cache_dir'] += f'_{args.preprocess_tag}'
    if args.aux_weight is not None:
        config['aux_weight'] = args.aux_weight
    run_pipeline(config)


if __name__ == '__main__':
    main()
