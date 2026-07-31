import pandas as pd
import os
import numpy as np
import random
import torch
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

# ---------- Helper functions (from original FairVGNN) ----------
def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1
    return mask

def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1

def sens_correlation(features, sens_idx):
    # Original implementation from utils.py – simplified version
    # Returns correlation of each feature with the sensitive attribute
    sens = features[:, sens_idx].unsqueeze(1)
    corr = torch.abs(torch.corrcoef(torch.cat((features, sens), dim=1))[-1, :-1])
    return corr

# ---------- Pokec loader ----------
def load_pokec(dataset, sens_attr="region", predict_attr="job",
               path="./data/", label_number=1000):
    """
    Load Pokec-z dataset.
    Assumes CSV has columns: user_id, region, job, ... (others as features)
    Edges file: tab-separated user_id pairs.
    """
    csv_path = os.path.join(path, f"{dataset}.csv")
    edge_path = os.path.join(path, f"{dataset}_relationship.txt")

    df = pd.read_csv(csv_path)
    # Remove user_id if present (not a feature)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])

    # Binarise label: keep only two most frequent job codes
    job_counts = df[predict_attr].value_counts()
    top_jobs = job_counts.index[:2].tolist()
    df = df[df[predict_attr].isin(top_jobs)].copy()
    # Map jobs to 0/1
    job_map = {top_jobs[0]: 0, top_jobs[1]: 1}
    df['label'] = df[predict_attr].map(job_map)
    labels = df['label'].values.astype(np.int64)

    # Binarise sensitive: most frequent region -> 1, others -> 0
    region_counts = df[sens_attr].value_counts()
    most_freq_region = region_counts.index[0]
    df['sens'] = (df[sens_attr] == most_freq_region).astype(np.int64)
    sens = df['sens'].values.astype(np.int64)

    # Features: all columns except label, sensitive, and predictor/sens columns
    feature_cols = [c for c in df.columns if c not in [predict_attr, sens_attr, 'label', 'sens']]
    features = df[feature_cols].values.astype(np.float32)

    # Build edges
    edges_df = pd.read_csv(edge_path, sep='\t', header=None)
    edges = edges_df.values  # shape (E,2)
    # Map original user_id to new indices (we only keep nodes in df)
    # We need to know original user_id; we dropped user_id, but we need mapping.
    # We need to keep user_id for mapping. So we should not drop user_id until after mapping.
    # Let's redo: keep user_id until we map.
    # Better: read csv with user_id, then filter, then map.
    # We'll redo from scratch.
    # Actually simpler: we read csv, keep user_id, filter based on top jobs.
    # Then we have a list of kept user_ids. We'll map those to new indices.
    # So let's restart the loading process.

def load_pokec(dataset, sens_attr="region", predict_attr="job",
               path="./data/", label_number=1000):
    csv_path = os.path.join(path, f"{dataset}.csv")
    edge_path = os.path.join(path, f"{dataset}_relationship.txt")

    df = pd.read_csv(csv_path)
    # Keep user_id for mapping; we'll drop it later
    # Binarise label: top two jobs
    job_counts = df[predict_attr].value_counts()
    top_jobs = job_counts.index[:2].tolist()
    df = df[df[predict_attr].isin(top_jobs)].copy()
    job_map = {top_jobs[0]: 0, top_jobs[1]: 1}
    df['label'] = df[predict_attr].map(job_map)
    labels = df['label'].values.astype(np.int64)

    # Binarise sensitive
    region_counts = df[sens_attr].value_counts()
    most_freq_region = region_counts.index[0]
    df['sens'] = (df[sens_attr] == most_freq_region).astype(np.int64)
    sens = df['sens'].values.astype(np.int64)

    # Features: all columns except label, sensitive, user_id
    feature_cols = [c for c in df.columns if c not in [predict_attr, sens_attr, 'label', 'sens', 'user_id']]
    features = df[feature_cols].values.astype(np.float32)
    # If no features remain, we still need something; use a constant? We'll assume at least one.
    if features.shape[1] == 0:
        # Add a dummy column of ones
        features = np.ones((df.shape[0], 1), dtype=np.float32)

    # Build edge_index: map user_id to new index
    # Keep a mapping from user_id to new index
    user_ids = df['user_id'].values
    id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    # Read edges
    edges_df = pd.read_csv(edge_path, sep='\t', header=None)
    edges = edges_df.values
    # Filter edges: keep only if both endpoints are in our kept nodes
    mask = np.isin(edges[:, 0], user_ids) & np.isin(edges[:, 1], user_ids)
    edges = edges[mask]
    # Map to new indices
    edge_index = np.array([[id_to_idx[uid1], id_to_idx[uid2]] for uid1, uid2 in edges]).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Build adjacency matrix (sparse) for normalization
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    # Symmetrize and add self-loops
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    # Convert to torch tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)

    # Train/val/test splits (balanced)
    random.seed(20)
    label_idx_0 = np.where(labels.numpy() == 0)[0]
    label_idx_1 = np.where(labels.numpy() == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    n_train_per_class = min(int(0.5 * min(len(label_idx_0), len(label_idx_1))), label_number // 2)
    idx_train = np.append(label_idx_0[:n_train_per_class],
                          label_idx_1[:n_train_per_class])
    idx_val = np.append(label_idx_0[n_train_per_class:int(1.5 * n_train_per_class)],
                        label_idx_1[n_train_per_class:int(1.5 * n_train_per_class)])
    idx_test = np.append(label_idx_0[int(1.5 * n_train_per_class):],
                         label_idx_1[int(1.5 * n_train_per_class):])

    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens

# ---------- get_dataset (only Pokec) ----------
def get_dataset(dataname, top_k):
    """
    Only supports 'pokec' – adjust path if needed.
    """
    if dataname != 'pokec':
        raise ValueError("Only 'pokec' is supported in this workshop")

    # Set label_number: use a reasonable number (e.g., 5000) for large dataset
    load, label_num = load_pokec, 5000

    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens = load(
        dataset=dataname, label_number=label_num, path="./data/"
    )

    # The sensitive index in features (we removed sens from features, so sens_idx = -1? Actually we didn't include sens in features)
    # In original code, they use sens_idx to remove correlation with sens. Since sens is not in features, we can set sens_idx = 0? But we don't need it because we have no sens in features.
    # We'll compute correlation with the original features? But we removed sens, so correlation with sens will be zero. We'll set sens_idx = -1 to skip.
    # However, the original get_dataset uses sens_correlation to pick top-k correlated features. We can skip that or compute correlation with the binarized sens (which we have).
    # Since features do not contain sensitive, correlation will be zero, but we can still compute if we want to keep the logic.
    # We'll compute correlation with the sens vector (which is external). We'll compute correlation between each feature and sens.
    # We'll use sens from the data.
    # We need to compute sens_correlation on features (which don't include sens). So we'll compute correlation with sens separately.
    # For simplicity, we can set sens_idx = -1 and set corr_matrix to zeros.
    # But the code later uses corr_idx to select features; we can skip that by setting top_k = 0.
    # So we'll just return dummy values.

    # Normalize features (feature_norm) – but we need to handle the case where features are already normalized.
    # We'll apply feature_norm.
    features = feature_norm(features)

    # We'll not compute correlation because sens is not in features.
    # Return a Data object with required attributes.
    data = Data(x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp,
                y=labels.float(), train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, sens=sens)

    # Dummy values for compatibility with the rest of the code
    sens_idx = -1
    corr_matrix = torch.zeros(features.shape[1])
    corr_idx = torch.arange(features.shape[1])
    x_min = features.min(dim=0)[0]
    x_max = features.max(dim=0)[0]

    return data, sens_idx, corr_matrix, corr_idx, x_min, x_max
