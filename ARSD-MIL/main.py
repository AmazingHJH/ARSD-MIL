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
from sklearn.model_selection import train_test_split, StratifiedKFold
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
        'max_minibags': 512,
        'aux_weight': 0.5,
    },
    'UBC_OC': {
        'h5_dir': r'E:\UBC\set-E\20x_256px_0px_overlap\features_uni_v1',
        'csv_path': r'data\UBC_OC_A.csv',
        'benign_h5_dir': r'E:\set\beni\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'HGSC': 0, 'LGSC': 1, 'MC': 2, 'CC': 3, 'EC': 4},
        'num_classes': 5,
        'minibag_size': 50,
        'top_k_loss': 10,
        'max_minibags': 512,
        'aux_weight': 0.01,
    },
    'TCGA_STAD': {
        'h5_dir': r'F:\TCGA-STAD\set-E\20x_256px_0px_overlap\features_uni_v1',
        'csv_path': r'data\TCGA_STAD.csv',
        'benign_h5_dir': r'F:\TCGA-STAD\set-L\20x_256px_0px_overlap\features_uni_v1',
        'label_map': {'Intestinal': 0, 'Diffuse': 1},
        'num_classes': 2,
        'minibag_size':50,
        'top_k_loss': 25,
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
        'top_k_loss': 25,
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
        'top_k_loss': 25,
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
        'patience': 30,
        'batch_size': 32,
        'epochs': 150,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
# 2. 核心模块: AbRank
# ==========================================
class AbRankScorer:
    def __init__(self, config):
        self.config = config
        self.memory_bank = None
        self.knn = None

    def load_or_build_bank(self):
        if os.path.exists(self.config['bank_save_path']):
            print(f"[Info] Loading Memory Bank from {self.config['bank_save_path']}...")
            self.memory_bank = np.load(self.config['bank_save_path'])
        else:
            print("[Info] Building Memory Bank from benign slides...")
            self.build_bank_from_benign_dir()

        print("[Info] Fitting KNN...")
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
                            kmeans = MiniBatchKMeans(n_clusters=prototypes_per_slide, batch_size=2048,
                                                     n_init='auto').fit(data)
                            slide_prototypes.append(kmeans.cluster_centers_)
            except Exception as e:
                print(f"Warning: Error reading {h5_path}: {e}")

        if not slide_prototypes:
            raise RuntimeError("Failed to extract any prototypes.")

        all_prototypes = np.concatenate(slide_prototypes, axis=0)

        if all_prototypes.shape[0] > target_total:
            kmeans = MiniBatchKMeans(n_clusters=target_total, batch_size=4096, n_init=3).fit(all_prototypes)
            self.memory_bank = kmeans.cluster_centers_
        else:
            self.memory_bank = all_prototypes

        np.save(self.config['bank_save_path'], self.memory_bank)

    def compute_raw_scores(self, features):
        if len(features) == 0: return np.array([])
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
                raw_dists = self.compute_raw_scores(features)
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
            except Exception as e:
                pass


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

    print(f"Test Acc: {acc * 100:.2f}% | F1: {f1:.4f} | AUC: {auc_score:.4f}")
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
                    'model_path': record['model_path']
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


def evaluate_fold_models(fold_records, config, test_loader, save_dir):
    available_records = [record for record in fold_records if os.path.exists(record['model_path'])]
    if not available_records:
        print("[Warning] No saved fold models found for fold evaluation.")
        return None

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

    ranked_records = sorted(
        available_records,
        key=lambda record: (
            -float(record['test_acc']),
            -float(record['test_auc']),
            -float(record['test_f1']),
            -float(record['val_acc']),
            float(record.get('val_loss', float('inf'))),
            int(record['fold'])
        )
    )
    best_record = ranked_records[0]

    print(
        f"\n>>> Rendering Plots From Best Fold {best_record['fold']} | "
        f"Test Acc: {float(best_record['test_acc']):.2f}%"
    )
    best_model = AMB_ViT(
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim'],
        ablation_cfg=config['ablation_settings'],
        top_k_ratio=config['top_k_ratio']
    ).to(config['device'])
    best_model.load_state_dict(torch.load(best_record['model_path'], weights_only=True, map_location=config['device']))
    best_metrics = test_model(
        best_model,
        test_loader,
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
        'folds': [int(record['fold']) for record in all_records],
        'fold_records': available_records,
    }

    print(f"\n>>> {len(all_records)}-Fold Test Summary")
    print(f"    Folds: {summary['folds']}")
    print(f"    Acc: {summary['acc']:.2f}% (+/- {summary['acc_std']:.2f})")
    print(f"    F1:  {summary['f1']:.4f} (+/- {summary['f1_std']:.4f})")
    print(f"    AUC: {summary['auc']:.4f} (+/- {summary['auc_std']:.4f})")
    print(f"    Plot source fold: {summary['best_fold']}")

    return summary


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    else:
        if not os.path.exists(cv_meta_path) and not os.path.exists(global_model_path):
            print(f"[Error] Neither '{cv_meta_path}' nor '{global_model_path}' was found.")
            return None

    df = pd.read_csv(CONFIG['csv_path'])
    scorer = AbRankScorer(CONFIG)
    scorer.load_or_build_bank()
    scorer.precompute_and_cache_minibags(df)

    valid_indices = [idx for idx, row in df.iterrows() if
                     os.path.exists(os.path.join(CONFIG['cache_dir'], f"{row['slide_id']}.pt"))]
    df = df.iloc[valid_indices].reset_index(drop=True)
    if len(df) == 0: return None

    train_val_df, test_df = train_test_split(df, test_size=CONFIG['test_size'], stratify=df['label'],  random_state=GLOBAL_SEED)
    test_ds = MiniBagDataset(test_df, CONFIG['cache_dir'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4,
                             collate_fn=collate_minibags, pin_memory=True)

    cv_scores = []
    fold_records = []
    global_best_acc = 0.0
    global_best_loss = float('inf')

    if CONFIG['mode'] == 'train':
        skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=GLOBAL_SEED)
        scaler = torch.amp.GradScaler('cuda')
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_val_df)), train_val_df['label'])):
            print(f"\n--- Fold {fold + 1}/{CONFIG['k_folds']} ---")
            train_loader = DataLoader(MiniBagDataset(train_val_df.iloc[train_idx], CONFIG['cache_dir']),
                                      batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4,
                                      collate_fn=collate_minibags, pin_memory=True)
            val_loader = DataLoader(MiniBagDataset(train_val_df.iloc[val_idx], CONFIG['cache_dir']),
                                    batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4,
                                    collate_fn=collate_minibags, pin_memory=True)
            model = AMB_ViT(num_classes=CONFIG['num_classes'], feature_dim=CONFIG['feature_dim'],
                            dropout=CONFIG['dropout'], ablation_cfg=CONFIG['ablation_settings'],
                            top_k_ratio=CONFIG['top_k_ratio']).to(CONFIG['device'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
            criterion, kl_criterion = nn.CrossEntropyLoss(), nn.KLDivLoss(reduction='none')
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(
                len(train_loader) * CONFIG['epochs'] * 0.1), num_training_steps=len(train_loader) * CONFIG['epochs'])
            best_fold_acc, best_fold_loss, patience_counter = 0.0, float('inf'), 0
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
                with torch.no_grad():
                    for batch in val_loader:
                        feats, raw_scores, coords, mask, labels = [x.to(CONFIG['device']) for x in batch]
                        with torch.amp.autocast('cuda'): logits, _, _ = model(feats, raw_scores, coords, mask)
                        correct += (logits.argmax(dim=1) == labels).sum().item();
                        total += labels.size(0)
                epoch_loss = total_loss / len(train_loader)
                val_acc = 100 * correct / total
                msg = f"Epoch [{epoch + 1:03d}/{CONFIG['epochs']}] | Time: {time.time() - start_time:.1f}s | Loss: {epoch_loss:.4f} | Val: {val_acc:.2f}%"
                is_val_improved = val_acc > best_fold_acc
                is_fold_best = (
                    is_val_improved or
                    (np.isclose(val_acc, best_fold_acc) and epoch_loss < best_fold_loss)
                )
                if is_fold_best:
                    best_fold_acc = val_acc
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
            if os.path.exists(fold_model_path):
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

                fold_records.append({
                    'fold': fold + 1,
                    'val_acc': float(best_fold_acc),
                    'val_loss': float(best_fold_loss),
                    'model_path': os.path.abspath(fold_model_path),
                    'test_acc': float(fold_test_metrics['acc']),
                    'test_f1': float(fold_test_metrics['f1']),
                    'test_auc': float(fold_test_metrics['auc'])
                })
                save_cv_records(cv_meta_path, fold_records)

    metrics = None
    if CONFIG['mode'] == 'train':
        if fold_records:
            save_cv_records(cv_meta_path, fold_records)
            metrics = evaluate_fold_models(fold_records, CONFIG, test_loader, save_dir)
            if metrics is not None and 'fold_records' in metrics:
                save_cv_records(cv_meta_path, metrics['fold_records'])
        elif os.path.exists(global_model_path):
            final_model = AMB_ViT(num_classes=CONFIG['num_classes'], feature_dim=CONFIG['feature_dim'],
                                ablation_cfg=CONFIG['ablation_settings'], top_k_ratio=CONFIG['top_k_ratio']).to(
                CONFIG['device'])
            final_model.load_state_dict(torch.load(global_model_path, weights_only=True, map_location=CONFIG['device']))
            metrics = test_model(
                final_model,
                test_loader,
                CONFIG['device'],
                CONFIG['label_map'],
                CONFIG['dataset_name'],
                save_dir
            )

        if cv_scores:
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            print(f"\nCV Mean Acc: {cv_mean:.2f}% (+/- {cv_std:.2f})")

            if metrics is None:
                metrics = {}

            metrics['cv_mean'] = cv_mean
            metrics['cv_std'] = cv_std
    else:
        saved_records = load_cv_records(cv_meta_path)
        if saved_records:
            metrics = evaluate_fold_models(saved_records, CONFIG, test_loader, save_dir)
            if metrics is not None and 'fold_records' in metrics:
                save_cv_records(cv_meta_path, metrics['fold_records'])
        elif os.path.exists(global_model_path):
            final_model = AMB_ViT(num_classes=CONFIG['num_classes'], feature_dim=CONFIG['feature_dim'],
                                ablation_cfg=CONFIG['ablation_settings'], top_k_ratio=CONFIG['top_k_ratio']).to(
                CONFIG['device'])
            final_model.load_state_dict(torch.load(global_model_path, weights_only=True, map_location=CONFIG['device']))
            metrics = test_model(
                final_model,
                test_loader,
                CONFIG['device'],
                CONFIG['label_map'],
                CONFIG['dataset_name'],
                save_dir
            )

    return metrics



def main():
    parser = argparse.ArgumentParser(description="AbRank Deep Learning Pipeline")
    parser.add_argument('--dataset', type=str, default='YNC_OC', choices=['YNC_OC', 'UBC_OC','TCGA_STAD','TCGA_LUNG','TCGA_BRCA'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--distill_type', type=str, default='D', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--aux_weight', type=float, default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    config = get_base_config(args.dataset)
    config.update({'mode': args.mode, 'distill_type': args.distill_type})
    if args.aux_weight is not None:
        config['aux_weight'] = args.aux_weight
    run_pipeline(config)


if __name__ == '__main__':
    main()
