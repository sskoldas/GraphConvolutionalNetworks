"""
================================================================================
GCN for Microbiome Disease Stage Prediction with Integrated Explanations
================================================================================

Architecture: Features → GCN(attention) → Weighted Features → MLP → Logits

================================================================================
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import coo_matrix
from scipy.stats import sem
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from utility.cm import ConfusionMatrixAggregator
from utility.roc_curve import ROCAggregator
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the explanation module
from utility.feature_importance import (
    MicrobiomeGCNExplainer,
    ExplanationResultSaver,
    ExplanationAggregator,
    ExplainerConfig,
    run_explanation_analysis
)

# ==============================================================================
class SimpleExperimentLog:
    """Collects experiment info and writes one summary file at the end."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.experiments = []
    
    def add_experiment(self, exp_num, train_size, test_size, 
                       used_ec_graph, early_stop_epoch=None):
        self.experiments.append({
            'experiment': exp_num,
            'train_samples': train_size,
            'test_samples': test_size,
            'graph_method': 'k-NN EC profiles' if used_ec_graph else 'Correlation fallback',
            'early_stopped': early_stop_epoch is not None,
            'early_stop_epoch': early_stop_epoch
        })
    
    def save(self):
        import json
        path = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(path, 'w') as f:
            json.dump({'experiments': self.experiments}, f, indent=2)
        
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(f"{'Exp':<5} {'Train':<8} {'Test':<8} {'Graph Method':<22} {'Early Stop':<12}")
        print("-"*70)
        for e in self.experiments:
            es = f"Epoch {e['early_stop_epoch']}" if e['early_stopped'] else "No"
            print(f"{e['experiment']:<5} {e['train_samples']:<8} {e['test_samples']:<8} {e['graph_method']:<22} {es:<12}")
        print("="*70)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration"""
    
    # File paths
    TAXONOMIC_DATA_PATH = "./data/merged.csv"
    EC_DATA_PATH = "./data/EC.csv"
    
    # Training parameters
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 0.001
    BATCH_SIZE = 32   
    MAX_EPOCHS = 250   
    TEST_SIZE = 0.2   
    EARLY_STOP_LOSS = 200    
    LOSS_MULTIPLIER = 100 
    
    # Model architecture
    CONV_CHANNELS = [1, 64, 64, 1]
    HIDDEN_WIDTH = 256
    
    # k-NN Graph construction
    K_NEIGHBORS = 5
    
    # Device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Experiment
    NUM_EXPERIMENTS = 2 
    OUTPUT_DIR = "gcn_results"
    
    # Explanation settings
    RUN_EXPLANATIONS = True
    EXPLANATION_EXPERIMENTS = "all"

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("="*70)
print("GCN MICROBIOME DISEASE STAGE PREDICTION WITH EXPLANATIONS")
print("="*70)
print(f"Device: {config.DEVICE}")
print(f"Architecture: Per-Sample Attention Weighting")
print(f"Learning Rate: {config.LEARNING_RATE}")
print(f"Weight Decay: {config.WEIGHT_DECAY}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Test Size: {config.TEST_SIZE}")

# ==============================================================================
# CUSTOM EXCEPTION FOR EARLY STOPPING
# ==============================================================================

class StopTraining(Exception):
    """Exception to stop training early"""
    pass

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_taxonomic_data(filepath):
    """
    Load merged taxonomic data from CSV.
    
    Expected columns:
    - 'SampleID' : sample identifier (optional)
    - 'label'    : disease label (required)
    - remaining columns: taxa (features)
    """
    print(f"\nLoading taxonomic data from {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Shape of raw taxonomic data: {df.shape}")
    
    # Check for NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        print(f"Warning: Found {nan_counts.sum()} NaN values, filling with 0")
        df = df.fillna(0)
    
    # Identify label column
    label_col = 'label'
    if label_col not in df.columns:
        raise ValueError(f"Required 'label' column not found! Available columns: {list(df.columns)}")
    
    print(f"Label column: '{label_col}'")

    # Check for sample ID column existence
    id_col = 'SampleID'
    if id_col in df.columns:
        sample_ids = df[id_col].values.astype(str)
        print(f"Sample ID column: '{id_col}' found with {len(sample_ids)} IDs")
    else:
        sample_ids = np.array([f"sample_{i}" for i in range(len(df))])
        print(f"No '{id_col}' column found, generating {len(sample_ids)} synthetic IDs")

    # Identify feature columns
    non_feature_cols = [label_col]
    if id_col in df.columns:
        non_feature_cols.append(id_col)
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found!")
    
    print(f"Number of features (taxa): {len(feature_cols)}")
    print(f"First 5 taxa: {feature_cols[:5]}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(str)
    
    # Validation
    if len(X) < 10:
        raise ValueError(f"Dataset too small: only {len(X)} samples")
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")
    
    # Check class distribution
    class_counts = Counter(y_encoded)
    print(f"Class distribution: {dict(class_counts)}")
    
    return X, y_encoded, feature_cols, label_encoder, y, sample_ids


def load_picrust2_ec_data(filepath):
    """
    Load EC table generated by PICRUSt2.
    
    Expected format:
    - rows: EC IDs (or samples, depending on orientation)
    - columns: Sample IDs (or EC IDs)
    
    We'll auto-detect orientation and return:
    - ec_matrix: DataFrame with samples as index, ECs as columns
    """
    print(f"\nLoading PICRUSt2 EC data from {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    print(f"Raw EC table shape: {df.shape}")
    
    # Heuristic: if index looks like EC numbers, assume rows = EC
    # and columns = samples; otherwise transpose.
    def looks_like_ec(s):
        # crude check: EC numbers like '1.1.1.1'
        return '.' in s and all(part.isdigit() for part in s.split('.') if part)
    
    index_looks_like_ec = looks_like_ec(str(df.index[0]))
    col_looks_like_ec = looks_like_ec(str(df.columns[0]))
    
    if index_looks_like_ec and not col_looks_like_ec:
        print("Detected rows as EC IDs; columns as samples → transposing")
        ec_matrix = df.T
    elif col_looks_like_ec and not index_looks_like_ec:
        print("Detected columns as EC IDs; rows as samples")
        ec_matrix = df
    else:
        # ambiguous; assume rows as EC, columns as samples
        print("Ambiguous EC orientation; assuming rows=EC, columns=samples and transposing")
        ec_matrix = df.T
    
    print(f"EC matrix (samples x ECs) shape: {ec_matrix.shape}")
    return ec_matrix

# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================

def build_knn_graph_from_ec(X_train, ec_matrix, taxon_names, tax_sample_ids_train,
                            ec_sample_ids, k=5):
    """
    Build a k-NN graph over taxa using PICRUSt2 EC functional profiles.
    
    """
    print("\nBuilding k-NN graph from EC profiles")

    if tax_sample_ids_train is None:
        raise ValueError("Training sample IDs are required")

    # Align samples
    tax_ids = set(tax_sample_ids_train)
    ec_ids = set(ec_sample_ids)
    common_ids = sorted(list(tax_ids.intersection(ec_ids)))

    if len(common_ids) < 2:
        raise ValueError("Not enough overlapping sample IDs")
    print(f"Overlapping training samples: {len(common_ids)}")

    # Map and align matrices
    tax_id_to_idx = {sid: idx for idx, sid in enumerate(tax_sample_ids_train)}
    tax_indices = [tax_id_to_idx[sid] for sid in common_ids]

    X_tax_matched = X_train[tax_indices, :]  # samples x taxa
    ec_matched = ec_matrix.loc[common_ids]    # samples x ECs

    print(f"Aligned taxonomic matrix: {X_tax_matched.shape}")
    print(f"Aligned EC matrix: {ec_matched.shape}")

    # Normalize EC
    ec_values = ec_matched.values.astype(np.float32)
    row_sums = ec_values.sum(axis=1, keepdims=True)
    non_zero_mask = row_sums.flatten() > 0
    
    ec_rel = np.zeros_like(ec_values)
    if non_zero_mask.any():
        ec_rel[non_zero_mask] = ec_values[non_zero_mask] / row_sums[non_zero_mask]
    else:
        raise ValueError("All samples have zero EC counts!")

    print("Computing taxon-EC functional profiles (vectorized)...")
    
    # Center both matrices
    X_centered = X_tax_matched - X_tax_matched.mean(axis=0, keepdims=True)
    ec_centered = ec_rel - ec_rel.mean(axis=0, keepdims=True)
    
    # Compute norms
    X_norms = np.linalg.norm(X_centered, axis=0, keepdims=True) + 1e-8  # [1, num_taxa]
    ec_norms = np.linalg.norm(ec_centered, axis=0, keepdims=True) + 1e-8  # [1, num_ec]
    
    # Normalize
    X_normalized = X_centered / X_norms  # [num_samples, num_taxa]
    ec_normalized = ec_centered / ec_norms  # [num_samples, num_ec]
    
    # Correlation: X_normalized.T @ ec_normalized gives [num_taxa, num_ec]
    taxon_ec_profiles = (X_normalized.T @ ec_normalized).astype(np.float32)
    
    # Handle zero-variance taxa (set their profiles to zero)
    zero_var_mask = (X_norms.flatten() < 1e-7)
    taxon_ec_profiles[zero_var_mask, :] = 0.0
    
    print(f"Taxon-EC profile matrix shape: {taxon_ec_profiles.shape}")

    # Build k-NN graph
    print(f"Building k-NN graph with k={k}")
    num_taxa = taxon_ec_profiles.shape[0]
    k_effective = min(k, num_taxa - 1)
    
    knn_graph = kneighbors_graph(
        taxon_ec_profiles,
        n_neighbors=k_effective,
        mode='connectivity',
        metric='cosine',
        include_self=False,
    )

    # Make undirected
    adj = knn_graph.maximum(knn_graph.T)
    adj_coo = adj.tocoo()
    edge_index = torch.stack([
        torch.from_numpy(adj_coo.row.astype(np.int64)),
        torch.from_numpy(adj_coo.col.astype(np.int64))
    ], dim=0)

    print(f"k-NN graph: {edge_index.size(1)} edges, {num_taxa} nodes")
    return edge_index


def build_correlation_graph_from_training(X_train, threshold=0.3):
    """
    Fallback: build an undirected graph based on feature-feature correlations
    over training samples.
    """
    print("\nBuilding correlation-based graph as fallback")
    num_samples, num_features = X_train.shape
    print(f"Training matrix shape: {X_train.shape}")
    
    # Compute Pearson correlation matrix between features
    X_centered = X_train - X_train.mean(axis=0, keepdims=True)
    cov = X_centered.T @ X_centered
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var)) + 1e-8
    corr = cov / denom
    
    # Build adjacency where |corr| > threshold
    adj = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    
    print(f"Correlation adjacency matrix shape: {adj.shape}")
    # Convert dense adjacency to COO and then to edge_index
    adj_coo = coo_matrix(adj)
    row = torch.from_numpy(adj_coo.row).long()  
    col = torch.from_numpy(adj_coo.col).long()
    edge_index = torch.stack([row, col], dim=0)

    print(f"Correlation graph built: {edge_index.size(1)} edges between {adj.shape[0]} taxa")
    return edge_index


# ==============================================================================
# DATASET AND COLLATE
# ==============================================================================
class MicrobiomeDataset(Dataset):
    """Dataset compatible with PyTorch Geometric-like batching."""
    
    def __init__(self, X, y, edge_index, sample_names):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.edge_index = edge_index
        self.sample_names = sample_names
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        label_idx = self.y[idx]
        return x, label_idx, self.sample_names[idx]


def collate_fn(batch):
    """
    Collate function that prepares a batch of samples.
    
    Returns:
    - batch_dict with:
        'examples'      : [batch_size, num_features]
        'label_index'   : [batch_size] integer labels
        'label'         : multi-hot (one-vs-rest) labels [batch_size, num_classes] (filled later)
        'sample_names'  : list of sample names
    """
    xs, label_indices, names = zip(*batch)
    X_batch = np.stack(xs, axis=0).astype(np.float32)
    label_indices = np.array(label_indices, dtype=np.int64)
    
    batch_dict = {
        'examples': torch.from_numpy(X_batch),
        'label_index': torch.from_numpy(label_indices),
        'sample_names': list(names),
        # 'label'
    }
    return batch_dict


# ==============================================================================
# OVERSAMPLER (class balancing)
# ==============================================================================

class OverSampler:
    """
    Simple oversampler that balances classes by oversampling minority classes 
    up to the size of the largest class.
    """
    def __init__(self, X, y, sample_names, random_state=None):
        self.X = X
        self.y = y
        self.sample_names = sample_names
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # Precompute class indices
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        self.max_class_size = max(len(indices) for indices in self.class_indices.values())
    
    def get_oversampled_data(self):
        """Return oversampled X, y, sample_names."""
        new_X = []
        new_y = []
        new_names = []
        
        for label, indices in self.class_indices.items():
            current_size = len(indices)
            
            # Add all original samples
            for idx in indices:
                new_X.append(self.X[idx])
                new_y.append(self.y[idx])
                new_names.append(self.sample_names[idx])
            
            # Oversample to match max class size
            if current_size < self.max_class_size:
                n_to_add = self.max_class_size - current_size
                sampled_indices = self.rng.choice(indices, size=n_to_add, replace=True)
                
                for idx in sampled_indices:
                    new_X.append(self.X[idx])
                    new_y.append(self.y[idx])
                    new_names.append(self.sample_names[idx] + "_oversample")
        
        return np.array(new_X), np.array(new_y), new_names


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class DeepConvNet(nn.Module):
    """Vectorized per-sample GCN attention module"""
    
    def __init__(self, num_nodes, channels, edge_index):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer('edge_index', edge_index)
        
        self.conv1 = GCNConv(channels[0], channels[1], add_self_loops=True)
        self.conv2 = GCNConv(channels[1], channels[2], add_self_loops=True)
        self.conv3 = GCNConv(channels[2], channels[3], add_self_loops=True)
        self.dropout = nn.Dropout(0.2)
        
        # Cache for batched edge indices
        self._edge_index_cache = {}
    
    def _get_batched_edge_index(self, batch_size, device):
        """Get or create cached batched edge_index."""
        cache_key = (batch_size, device)
        
        if cache_key not in self._edge_index_cache:
            if batch_size == 1:
                batched = self.edge_index
            else:
                # Create offsets: [0, num_nodes, 2*num_nodes, ...]
                offsets = torch.arange(batch_size, device=device) * self.num_nodes
                
                # Repeat edge_index for batch
                num_edges = self.edge_index.size(1)
                edge_index_repeated = self.edge_index.repeat(1, batch_size)
                
                # Add offsets to each replicated graph
                offset_vector = offsets.repeat_interleave(num_edges)
                batched = edge_index_repeated + offset_vector.unsqueeze(0)
            
            self._edge_index_cache[cache_key] = batched
        
        return self._edge_index_cache[cache_key]
    
    def forward(self, x):
        """
        x: [batch_size, num_taxa]
        Returns: [batch_size, num_taxa]
        """
        batch_size, num_nodes = x.shape
        device = x.device
        
        # Reshape to [batch_size * num_nodes, 1] for GCN
        node_features = x.reshape(-1, 1)
        
        # Get batched edge_index (cached)
        edge_index_batched = self._get_batched_edge_index(batch_size, device)
        
        # Forward through GCN layers (single pass for entire batch)
        h = F.relu(self.conv1(node_features, edge_index_batched))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index_batched))
        h = self.dropout(h)
        h = self.conv3(h, edge_index_batched)
        
        # Reshape back to [batch_size, num_nodes]
        h = h.view(batch_size, num_nodes)
        
        return torch.sigmoid(h)  # [batch_size, num_taxa]

class MetaphlanNet(nn.Module):
    """
    Dense network that follows the GCN
    """
    def __init__(self, input_dim, hidden_width, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_width)
        self.fc_out = nn.Linear(hidden_width, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: [batch, num_features]
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc_out(h)
        return logits, h


class MulticlassDiseaseClassifier(nn.Module):
    """
    model: GCN (attention) + MLP (classification)
    
    Architecture: Features → GCN(attention) → Weighted Features → MLP → Logits
    """
    def __init__(self, num_features, num_nodes, conv_channels,
                 hidden_width, num_classes, edge_index):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.conv_net = DeepConvNet(num_nodes=num_nodes,
                                    channels=conv_channels,
                                    edge_index=edge_index)
        
        self.deep_net = MetaphlanNet(input_dim=num_features,
                                     hidden_width=hidden_width,
                                     num_classes=num_classes)
    
    def forward(self, batch):
        """
        batch: dict with 'examples' key of shape [batch, num_features]
        """
        x = batch['examples']  # [batch_size, num_taxa]
        
        # Get sample-specific attention weights
        attention = self.conv_net(x)  # [batch_size, num_taxa]
        
        # Apply attention to features (element-wise multiplication)
        x_attended = x * attention  # [batch_size, num_taxa]
        
        # MLP classification
        logits, h_embed = self.deep_net(x_attended)  # [batch_size, num_classes]
        batch['logits'] = logits
        probs_softmax = F.softmax(logits, dim=1)
        
        # Store for interpretation and loss
        batch['attention_weights'] = attention
        batch['predictions'] = probs_softmax
        batch['embedding_fc1'] = h_embed
        return batch


# ==============================================================================
# LOSS & METRICS
# ==============================================================================

def attach_one_hot_labels(batch, num_classes):
    """
    Convert integer label_index into multi-hot label vectors.
    Each sample has exactly one active label.
    """
    label_indices = batch['label_index']
    batch_size = label_indices.size(0)
    label = torch.zeros(batch_size, num_classes, dtype=torch.float32,
                        device=label_indices.device)
    label[torch.arange(batch_size), label_indices] = 1.0
    batch['label'] = label
    return batch


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    
    for batch in train_loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # Build multi-hot labels
        batch = attach_one_hot_labels(batch, model.num_classes)
        
        # # Forward: logits + softmax predictions
        batch = model(batch)
        logits = batch['logits']
        targets = batch['label']
        
        # BCEWithLogitsLoss     
        loss = criterion(logits, targets)
        loss = config.LOSS_MULTIPLIER * loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Early stopping condition on batch scaled loss
        if loss.item() < config.EARLY_STOP_LOSS:
            raise StopTraining
    
    return epoch_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    
    all_probs = []
    all_labels = []
    all_indices = []
    all_embeddings = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch_size = batch['examples'].size(0)
            
            batch = attach_one_hot_labels(batch, model.num_classes)
            batch = model(batch)
            
            probs = batch['predictions']
            labels = batch['label']
            indices = batch['label_index']
            embeddings = batch['embedding_fc1']  # shape [batch, hidden_width]
            attention = batch['attention_weights']
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(indices.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            all_attention_weights.append(attention.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_attention = np.concatenate(all_attention_weights, axis=0)

    y_pred = np.argmax(all_probs, axis=1)
    y_true = all_indices
    
    metrics = {
        'correct': {},
        'top_3': {},
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': all_probs,
        'attention_weights': all_attention,
        'embeddings_fc1': all_embeddings
    }
    
    # For each sample, compute rank of true label
    for i in range(all_probs.shape[0]):
        probs = all_probs[i]
        label_index = all_indices[i]
        
        ranked_indices = np.argsort(-probs)
        true_rank = np.where(ranked_indices == label_index)[0][0] + 1
        
        correct = (true_rank == 1)
        top_3 = (true_rank <= 3)
        
        for key, val in zip(['correct', 'top_3'], [correct, top_3]):
            metrics[key].setdefault(int(label_index), []).append(int(val))
    
    return metrics


def print_accuracy(metrics, label_encoder_classes):
    """
    Print per-class and average (macro) accuracy.
    """
    correct = metrics['correct']
    top_3 = metrics['top_3']
    
    print("\nAccuracy for CNN")
    print("Label\tTop-1\tTop-3")
    
    macro_acc = []
    
    for i, label_name in enumerate(label_encoder_classes):
        if i not in correct:
            continue
        c1 = np.mean(correct[i])
        c3 = np.mean(top_3[i])
        macro_acc.append(c1)
        print(f"{label_name}\t{c1:.3f}\t{c3:.3f}")
    
    macro_acc_val = np.mean(macro_acc) if macro_acc else 0.0
    print(f"\nMacro top-1 accuracy: {macro_acc_val:.3f}")
    return macro_acc_val


# ==============================================================================
# MAIN EXPERIMENT LOOP
# ==============================================================================

def run_experiment(experiment_num, X, y, feature_names, label_encoder,
                   labels_dict, named_labels, study_labels, ec_matrix=None,
                   sample_ids=None, cm_aggregator=None, roc_aggregator=None,
                   explanation_aggregator=None, run_explanation=False):
    """Run a single experiment"""
    
    print("\n" + "="*70)
    print(f"EXPERIMENT {experiment_num}")
    print("="*70)
    
    num_features = X.shape[1]
    num_classes = len(label_encoder.classes_)
    
    # Set random seeds for reproducibility
    np.random.seed(experiment_num)
    torch.manual_seed(experiment_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(experiment_num)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    named_labels_shuffled = [named_labels[i] for i in indices]
    if sample_ids is not None:
        sample_ids_shuffled = sample_ids[indices]
    else:
        sample_ids_shuffled = None
    
    # Train/test split
    if sample_ids_shuffled is not None:
        split_data = train_test_split(
            X_shuffled, y_shuffled, named_labels_shuffled, sample_ids_shuffled,
            test_size=config.TEST_SIZE,
            random_state=experiment_num,
            stratify=y_shuffled
        )
        X_train, X_test, y_train, y_test, names_train, names_test, sample_ids_train, sample_ids_test = split_data
    else:
        split_data = train_test_split(
            X_shuffled, y_shuffled, named_labels_shuffled,
            test_size=config.TEST_SIZE,
            random_state=experiment_num,
            stratify=y_shuffled
        )
        X_train, X_test, y_train, y_test, names_train, names_test = split_data
        sample_ids_train = None
        sample_ids_test = None
    
    train_size = len(X_train)
    test_size = len(X_test)
    print(f"Initial split - Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Build graph using ONLY training data
    edge_index = None
    used_ec_graph = False
    
    if ec_matrix is not None and sample_ids_train is not None:
        ec_sample_ids = list(ec_matrix.index)
        try:
            edge_index = build_knn_graph_from_ec(
                X_train, ec_matrix, feature_names,
                sample_ids_train, ec_sample_ids,
                k=config.K_NEIGHBORS
            )
            used_ec_graph = True
        except Exception as e:
            print(f"Warning: EC graph construction failed: {e}")
            print("Falling back to correlation-based graph")
            edge_index = None
    else:
        print("EC information not available or no aligned sample IDs; skipping EC-based graph.")
    
    if edge_index is None:
        print("\nFalling back to correlation-based graph")
        edge_index = build_correlation_graph_from_training(X_train, threshold=0.3)
    


    # Normalize training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Oversample training data on normalized features
    oversampler = OverSampler(X_train_scaled, y_train, names_train, 
                             random_state=experiment_num)
    X_train_oversampled, y_train_oversampled, names_train_oversampled = oversampler.get_oversampled_data()

    # Create datasets
    train_dataset = MicrobiomeDataset(X_train_oversampled, y_train_oversampled, 
                                     edge_index, names_train_oversampled)
    test_dataset = MicrobiomeDataset(X_test_scaled, y_test, edge_index, names_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    print("\nInitializing Model...")
    print(f"Architecture: Per-Sample Attention")
    model = MulticlassDiseaseClassifier(
        num_features=num_features,
        num_nodes=num_features,
        conv_channels=config.CONV_CHANNELS,
        hidden_width=config.HIDDEN_WIDTH,
        num_classes=num_classes,
        edge_index=edge_index
    )
    model = model.to(config.DEVICE)
    model.conv_net.edge_index = model.conv_net.edge_index.to(config.DEVICE)
    
    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.LEARNING_RATE,
                                weight_decay=config.WEIGHT_DECAY)
    
    # Save labels dict
    with open(os.path.join(config.OUTPUT_DIR, f"deep_{experiment_num}_cnn_labels_dict.json"), "w") as f:
        json.dump(labels_dict, f)
    
    # Training
    print(f"\nNow training (max {config.MAX_EPOCHS} epochs)")
    print(f"LR: {config.LEARNING_RATE}, Weight Decay: {config.WEIGHT_DECAY}")
    try:
        early_stop_epoch = None
        for epoch in range(1, config.MAX_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}, loss: {loss:.4f}")
    except StopTraining:
        print(f"Early stopping at epoch {epoch} (loss < {config.EARLY_STOP_LOSS})")
        early_stop_epoch = epoch
    
    # Evaluation
    print("\nComputing accuracy for CNN")
    metrics = evaluate(model, test_loader, config.DEVICE)
    accuracy = print_accuracy(metrics, study_labels)

    emb = metrics['embeddings_fc1']
    lab = metrics['y_true']

    # PCA to 50D for stability
    X_pca = PCA(n_components=min(50, emb.shape[1])).fit_transform(emb)
    tsne = TSNE(n_components=2,perplexity=30,learning_rate=200, max_iter=3000,random_state=42)
    z = tsne.fit_transform(X_pca) 
    
    num_classes = len(study_labels)
    class_names = {
    0: "Advanced",
    1: "Early",
    2: "Healthy",
    3: "Intermediate"
}
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    plt.figure(figsize=(7,6))
    for cls in range(num_classes):
        idx = (lab == cls)
        plt.scatter(
            z[idx, 0],
            z[idx, 1],
            color=colors[cls],
            s=19,
            label=class_names[cls]
        )
    plt.title(f"t-SNE of learned embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="")
    plt.tight_layout()
    
    tsne_dir = os.path.join(config.OUTPUT_DIR, "t_sne")
    os.makedirs(tsne_dir, exist_ok=True)
    tsne_path = os.path.join(tsne_dir, f"tsne_run{experiment_num}.png")
    plt.savefig(tsne_path, dpi=300)
    plt.close()

    print(f"Saved t-SNE plot for run {experiment_num} to: {tsne_path}")

    
    # Save metrics and attention weights
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    y_probs = metrics['y_probs']
    attention_weights = metrics['attention_weights']
    
    # Save attention weights for interpretation
    attention_save_path = os.path.join(config.OUTPUT_DIR, f"deep_{experiment_num}_attention_weights.npy")
    np.save(attention_save_path, attention_weights)
    print(f"Attention weights saved to: {attention_save_path}")
    
    # Aggregate confusion matrix
    if cm_aggregator is not None:
        cm_aggregator.add_experiment(y_true, y_pred)
    
    # Aggregate ROC curves
    if roc_aggregator is not None:
        roc_aggregator.add_experiment(y_true, y_probs)
    
    # Save individual experiment metrics
    metrics_dict = {
        'correct': {int(k): v for k, v in metrics['correct'].items()},
        'top_3': {int(k): v for k, v in metrics['top_3'].items()},
        'accuracy': float(accuracy)
    }
    
    with open(os.path.join(config.OUTPUT_DIR, f"deep_{experiment_num}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    # =========================================================================
    # EXPLANATION ANALYSIS
    # =========================================================================
    explanation_results = None
    
    if run_explanation:
        print("\n" + "="*70)
        print(f"RUNNING EXPLANATION ANALYSIS FOR EXPERIMENT {experiment_num}")
        print("="*70)
        
        # Use experiment_<num> directory directly; saver will create subfolders
        explanation_output_dir = os.path.join(config.OUTPUT_DIR, f"experiment_{experiment_num}")
        
        try:
            explanation_results = run_explanation_analysis(
                model=model,
                edge_index=edge_index.to(config.DEVICE),
                X_test=X_test_scaled,
                y_test=y_test,
                X_train=X_train_scaled,
                feature_names=feature_names,
                class_labels=study_labels,
                device=config.DEVICE,
                output_dir=explanation_output_dir,
                experiment_num=experiment_num,
                verbose=True
            )
            
            # Add to aggregator
            if explanation_aggregator is not None:
                explanation_aggregator.add_experiment(explanation_results)
                print(f"Added experiment {experiment_num} to explanation aggregator")
                
        except Exception as e:
            print(f"Warning: Explanation analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    return accuracy, explanation_results, train_size, test_size, used_ec_graph, early_stop_epoch


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Load taxonomic data
    X, y, feature_names, label_encoder, named_labels, sample_ids = load_taxonomic_data(
        config.TAXONOMIC_DATA_PATH
    )
    
    # Map from sample name to label name (for reference output)
    labels_dict = {i: name for i, name in enumerate(label_encoder.classes_)}
    study_labels = list(label_encoder.classes_)
    
    # Load EC data
    if os.path.exists(config.EC_DATA_PATH):
        ec_matrix = load_picrust2_ec_data(config.EC_DATA_PATH)
    else:
        print(f"EC data file not found at {config.EC_DATA_PATH}; proceeding without EC-based graph.")
        ec_matrix = None

    # Initialize ConfusionMatrixAggregator
    cm_aggregator = ConfusionMatrixAggregator(class_labels=study_labels)

    # Initialize ROCAggregator
    roc_aggregator = ROCAggregator(class_labels=study_labels)
    
    # Initialize ExplanationAggregator
    explanation_aggregator = None
    if config.RUN_EXPLANATIONS:
        explanation_aggregator = ExplanationAggregator(
            feature_names=feature_names,
            class_labels=study_labels
        )
    
    exp_log = SimpleExperimentLog(config.OUTPUT_DIR)
    # Determine which experiments to run explanations for
    if config.RUN_EXPLANATIONS:
        if config.EXPLANATION_EXPERIMENTS == 'all':
            explanation_experiments = list(range(1, config.NUM_EXPERIMENTS + 1))
        else:
            explanation_experiments = config.EXPLANATION_EXPERIMENTS
    else:
        explanation_experiments = []
    
    # Run experiments
    all_accuracies = []
    all_explanation_results = {}
    
    for exp in range(1, config.NUM_EXPERIMENTS + 1):
        run_explanation = exp in explanation_experiments
        
        acc, exp_results, train_size, test_size, used_ec_graph, early_stop_epoch = run_experiment(
            experiment_num=exp,
            X=X,
            y=y,
            feature_names=feature_names,
            label_encoder=label_encoder,
            labels_dict=labels_dict,
            named_labels=named_labels,
            study_labels=study_labels,
            ec_matrix=ec_matrix,
            sample_ids=sample_ids,
            cm_aggregator=cm_aggregator,
            roc_aggregator=roc_aggregator,
            explanation_aggregator=explanation_aggregator,
            run_explanation=run_explanation
        )
        all_accuracies.append(acc)
        
        exp_log.add_experiment(exp, train_size, test_size, used_ec_graph, early_stop_epoch)

        if exp_results is not None:
            all_explanation_results[exp] = exp_results

    exp_log.save()
    # ========================================================================
    # GENERATE AGGREGATED CONFUSION MATRIX RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING AGGREGATED CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    # Print summary statistics with 95% CI
    cm_aggregator.print_summary()
    
    # Save all aggregated results
    cm_output_dir = os.path.join(config.OUTPUT_DIR, "confusion_matrices")
    cm_aggregator.save_results(cm_output_dir)
    
    # Generate confusion matrix plot
    cm_plot_path = os.path.join(cm_output_dir, "confusion_matrix_aggregated.png")
    cm_aggregator.plot_confusion_matrix(
        output_path=cm_plot_path,
        title=f'Mean Confusion Matrix with 95% CI (n={config.NUM_EXPERIMENTS} experiments)',
        normalize='true',
        annot_fontsize=14,
        label_fontsize=14,
        title_fontsize=16,
        tick_fontsize=12,
        figsize=(10, 8)
    )

    
    # ========================================================================
    # GENERATE ROC CURVE ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING ROC CURVE ANALYSIS")
    print("="*70)
    
    # Save all ROC results
    roc_output_dir = os.path.join(config.OUTPUT_DIR, "roc_curves")
    roc_aggregator.save_results(roc_output_dir)
    
    print(f"\nROC curve analysis complete!")
    
    # ========================================================================
    # GENERATE AGGREGATED EXPLANATION RESULTS
    # ========================================================================
    if config.RUN_EXPLANATIONS and explanation_aggregator is not None:
        print("\n" + "="*70)
        print("GENERATING AGGREGATED EXPLANATION ANALYSIS")
        print("="*70)
        
        # Compute aggregated results
        aggregated_explanations = explanation_aggregator.compute_aggregated_results()
        
        # Get robust rankings
        robust_rankings = explanation_aggregator.get_robust_rankings(aggregated_explanations)
        
        # Save aggregated data as pickle
        aggregated_data_dir = os.path.join(config.OUTPUT_DIR, "aggregated_results")
        os.makedirs(aggregated_data_dir, exist_ok=True)
        
        with open(os.path.join(aggregated_data_dir, "aggregated_results.pkl"), 'wb') as f:
            pickle.dump(aggregated_explanations, f)
        
        
        # ====================================================================
        # Save aggregated GNNExplainer and IntegratedGradients results
        # ====================================================================
        explanation_aggregator.save_aggregated_method_results(
            aggregated_explanations,
            config.OUTPUT_DIR
        )
        
        
        print(f"\nAggregated explanation results saved to: {aggregated_data_dir}")

    # ========================================================================
    # Save overall summary
    # ========================================================================
    summary = {
        'accuracies': all_accuracies,
        'mean_accuracy': float(np.mean([float(acc) for acc in all_accuracies])) if all_accuracies else 0.0,
        'std_accuracy': float(np.std([float(acc) for acc in all_accuracies])) if all_accuracies else 0.0,
        'seed': config.NUM_EXPERIMENTS,
        'config': {
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'batch_size': config.BATCH_SIZE,
            'conv_channels': config.CONV_CHANNELS,
            'hidden_width': config.HIDDEN_WIDTH,
            'k_neighbors': config.K_NEIGHBORS,
            'test_size': config.TEST_SIZE,
            'num_experiments': config.NUM_EXPERIMENTS,
            'run_explanations': config.RUN_EXPLANATIONS,
            'explanation_experiments': config.EXPLANATION_EXPERIMENTS if config.RUN_EXPLANATIONS else []
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to: {config.OUTPUT_DIR}/")
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nMean Accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
    print(f"Configuration: LR={config.LEARNING_RATE}, WD={config.WEIGHT_DECAY}, BS={config.BATCH_SIZE}")


if __name__ == "__main__":
    main()