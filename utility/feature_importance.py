"""
================================================================================
GNN EXPLAINER MODULE FOR MICROBIOME DISEASE PREDICTION
================================================================================

Two-tier Explanation Strategy:
1. GNNExplainer - For node (taxa) and edge (functional relationships) importance
2. Integrated Gradients - For attribution-based feature importance

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import GNNExplainer as TGGNNExplainer
from torch_geometric.explain.config import (
    ExplanationType,
    ExplainerConfig as TGExplainerConfig,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
    MaskType
)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import sem
from collections import defaultdict
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION FOR EXPLAINER
# ==============================================================================

class ExplainerConfig:
    """Configuration for explanation module"""
    
    # GNNExplainer parameters
    GNNEXPLAINER_EPOCHS = 200
    GNNEXPLAINER_LR = 0.01
    NODE_MASK_TYPE = "attributes"
    EDGE_MASK_TYPE = "object"
    
    # Integrated Gradients parameters
    IG_STEPS = 50
    IG_BASELINE = "zero"
    IG_BATCH_SIZE = 10  # Process one sample at a time to avoid OOM
    
    # Output settings
    TOP_K_NODES = 50
    TOP_K_EDGES = 100
    
    # Statistical thresholds
    SIGNIFICANCE_THRESHOLD = 0.05
    MIN_IMPORTANCE_THRESHOLD = 0.01


# ==============================================================================
# INTEGRATED GRADIENTS IMPLEMENTATION
# ==============================================================================

class IntegratedGradients:
    """
    Integrated Gradients for GNN explanation.
    """
    
    def __init__(self, model, device, baseline_type='zero', n_steps=50, batch_size=1):
        self.model = model
        self.device = device
        self.baseline_type = baseline_type
        self.n_steps = n_steps
        self.batch_size = batch_size  # configurable batch size
        
    def _get_baseline(self, x, X_train=None):
        if self.baseline_type == 'zero':
            baseline = torch.zeros_like(x, device=self.device)
        elif self.baseline_type == 'mean':
            if X_train is not None:
                base_vec = torch.tensor(X_train.mean(axis=0), dtype=torch.float32, device=self.device)
                if base_vec.ndim == 1 and x.ndim == 2:
                    baseline = base_vec.unsqueeze(0).expand(x.size(0), -1)
                else:
                    baseline = base_vec
            else:
                baseline = torch.zeros_like(x, device=self.device)
        else:
            baseline = torch.zeros_like(x, device=self.device)
        
        if baseline.shape != x.shape:
            baseline = baseline.expand_as(x)
        return baseline
    
    def _get_baseline_single(self, x_single, X_train=None):
        """Get baseline for a single sample [1, features]."""
        if self.baseline_type == 'zero':
            baseline = torch.zeros_like(x_single, device=self.device)
        elif self.baseline_type == 'mean':
            if X_train is not None:
                base_vec = torch.tensor(X_train.mean(axis=0), dtype=torch.float32, device=self.device)
                baseline = base_vec.unsqueeze(0)  # [1, features]
            else:
                baseline = torch.zeros_like(x_single, device=self.device)
        else:
            baseline = torch.zeros_like(x_single, device=self.device)
        return baseline
    
    def compute_gradients(self, x, target_class):
        x = x.clone().detach().requires_grad_(True)
        batch = {'examples': x}
        output = self.model(batch)
        probs = output['predictions']
        target_prob = probs[:, target_class].sum()
        self.model.zero_grad()
        target_prob.backward()
        return x.grad.detach()
    
    def _attribute_single(self, x_single, target_class, X_train=None):
        """
        Compute attributions for a single sample.
        
        Args:
            x_single: Single sample tensor [1, features]
            target_class: Target class index
            X_train: Training data for baseline computation (optional)
            
        Returns:
            Attribution tensor [1, features]
        """
        baseline = self._get_baseline_single(x_single, X_train)
        
        # Create scaled inputs: [n_steps+1, features]
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device).view(-1, 1)
        scaled_inputs = baseline + alphas * (x_single - baseline)
        scaled_inputs.requires_grad_()
        
        # Forward pass through model
        with torch.enable_grad():
            batch_dict = {'examples': scaled_inputs}
            outputs = self.model(batch_dict)
            probs = outputs['predictions']
            target_probs = probs[:, target_class]
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=target_probs.sum(),
                inputs=scaled_inputs,
                retain_graph=False,
                create_graph=False
            )[0]
        
        # Trapezoidal rule: average adjacent gradients, then mean over path
        avg_gradients = (gradients[:-1] + gradients[1:]).mean(dim=0) / 2
        
        # Attribution = (input - baseline) * average_gradients
        attribution = (x_single.squeeze(0) - baseline.squeeze(0)) * avg_gradients
        
        return attribution.detach().unsqueeze(0)  # [1, features]
    
    def attribute(self, x, target_class, X_train=None):
        """
        Compute Integrated Gradients attributions.
        
        Args:
            x: Input tensor [batch, features] or [features]
            target_class: Target class index
            X_train: Training data for baseline computation (optional)
            
        Returns:
            Attribution tensor [batch, features]
        """
        self.model.eval()
        
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        n_samples = x.size(0)
        all_attributions = []
        
        # Process samples one at a time to avoid OOM
        for i in range(n_samples):
            x_single = x[i:i+1]  # [1, features]
            
            attribution = self._attribute_single(x_single, target_class, X_train)
            all_attributions.append(attribution)
            
            # Clear CUDA cache periodically to prevent memory buildup
            if self.device.type == 'cuda' and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Stack all attributions: [n_samples, features]
        result = torch.cat(all_attributions, dim=0)
        
        # Final cache clear
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return result
    
    def attribute_batch(self, X, target_class, X_train=None):
        """
        Compute attributions for a batch of samples.
        
        Args:
            X: Input array [batch, features]
            target_class: Target class index
            X_train: Training data for baseline computation (optional)
            
        Returns:
            Attribution numpy array [batch, features]
        """
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        attributions = self.attribute(x, target_class, X_train)
        return attributions.cpu().numpy()


# ==============================================================================
# GNN EXPLAINER WRAPPER
# ==============================================================================

class GNNExplainerWrapper:
    """
    Thin adapter around torch_geometric.explain.GNNExplainer.
    """
    
    class _ModelWrapper(nn.Module):
        """Adapts the custom model signature to GNNExplainer expectations."""
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x, edge_index):
            # x: [num_nodes, feat_dim] → model expects [batch, num_features]
            sample = x.transpose(0, 1)  # assume feat_dim is small (usually 1)
            batch = {'examples': sample}
            if hasattr(self.base_model, 'conv_net'):
                self.base_model.conv_net.edge_index = edge_index
            outputs = self.base_model(batch)
            probs = outputs['predictions']
            # Convert probabilities to logits for stable CE loss
            logits = torch.logit(torch.clamp(probs, 1e-6, 1 - 1e-6))
            return logits
    
    def __init__(self, model, edge_index, device, num_classes, epochs=200, lr=0.01):
        self.model = model
        self.edge_index = edge_index.to(device)
        self.device = device
        
        self.model_wrapper = self._ModelWrapper(model).to(device)
        self.explainer = TGGNNExplainer(epochs=epochs, lr=lr)
        self.explainer.connect(
            explainer_config=TGExplainerConfig(
                explanation_type=ExplanationType.model,
                node_mask_type=MaskType.attributes,
                edge_mask_type=MaskType.object
            ),
            model_config=ModelConfig(
                mode=ModelMode.multiclass_classification,
                task_level=ModelTaskLevel.graph,
                return_type=ModelReturnType.raw
            )
        )
    
    def explain_sample(self, x, target_class):
        self.model.eval()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        node_features = x.squeeze(0).unsqueeze(-1).to(self.device)  # [num_features, 1]
        target = torch.tensor([target_class], device=self.device)
        
        explanation = self.explainer(
            self.model_wrapper,
            node_features,
            self.edge_index,
            target=target
        )
        
        node_mask = explanation.node_mask
        edge_mask = explanation.edge_mask
        
        node_mask_np = node_mask.detach().cpu().numpy().squeeze() if node_mask is not None else np.zeros(node_features.size(0))
        edge_mask_np = edge_mask.detach().cpu().numpy() if edge_mask is not None else np.zeros(self.edge_index.size(1))
        
        return node_mask_np, edge_mask_np
    
    def explain_batch(self, X, target_class):
        all_node_masks = []
        all_edge_masks = []
        
        for i in range(X.shape[0]):
            x = torch.tensor(X[i], dtype=torch.float32, device=self.device)
            node_mask, edge_mask = self.explain_sample(x, target_class)
            all_node_masks.append(node_mask)
            all_edge_masks.append(edge_mask)
        
        return np.array(all_node_masks), np.array(all_edge_masks)


# ==============================================================================
# COMPREHENSIVE EXPLAINER CLASS
# ==============================================================================

class MicrobiomeGCNExplainer:
    """
    Comprehensive explainer for microbiome GCN disease prediction.
    """
    
    def __init__(self, model, edge_index, feature_names, class_labels,
                 device, config=None):
        self.model = model
        self.edge_index = edge_index
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.device = device
        self.config = config or ExplainerConfig()
        
        self.num_features = len(feature_names)
        self.num_classes = len(class_labels)
        
        # Pass batch_size to IntegratedGradients
        self.ig_explainer = IntegratedGradients(
            model, device,
            baseline_type=self.config.IG_BASELINE,
            n_steps=self.config.IG_STEPS,
            batch_size=getattr(self.config, 'IG_BATCH_SIZE', 1)
        )
        
        self.gnn_explainer = GNNExplainerWrapper(
            model, edge_index, device, num_classes=self.num_classes,
            epochs=self.config.GNNEXPLAINER_EPOCHS,
            lr=self.config.GNNEXPLAINER_LR
        )
        
        self.results = {
            'integrated_gradients': {},
            'gnn_explainer': {},
            'per_class': {},
            'edges': {}
        }
        
    def explain_integrated_gradients(self, X, y, X_train=None, verbose=True):
        if verbose:
            print("\n" + "="*70)
            print("INTEGRATED GRADIENTS EXPLANATION")
            print("="*70)
        
        ig_results = {
            'per_sample': {},
            'per_class': {},
            'global': None
        }
        
        all_attributions = []
        
        for class_idx in range(self.num_classes):
            class_name = self.class_labels[class_idx]
            
            if verbose:
                print(f"\nComputing IG for class: {class_name}")
            
            class_mask = y == class_idx
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            attributions = self.ig_explainer.attribute_batch(X_class, class_idx, X_train)
            
            ig_results['per_class'][class_name] = {
                'attributions': attributions,
                'mean_attribution': attributions.mean(axis=0),
                'std_attribution': attributions.std(axis=0),
                'abs_mean_attribution': np.abs(attributions).mean(axis=0)
            }
            
            all_attributions.append(attributions)
        
        if all_attributions:
            all_attr = np.concatenate(all_attributions, axis=0)
            ig_results['global'] = {
                'mean_attribution': all_attr.mean(axis=0),
                'abs_mean_attribution': np.abs(all_attr).mean(axis=0),
                'std_attribution': all_attr.std(axis=0)
            }
        
        self.results['integrated_gradients'] = ig_results
        return ig_results
    
    def explain_gnn_explainer(self, X, y, verbose=True, max_samples_per_class=20):
        if verbose:
            print("\n" + "="*70)
            print("GNN EXPLAINER EXPLANATION")
            print("="*70)
        
        gnn_results = {
            'node_importance': {},
            'edge_importance': {},
            'per_class': {}
        }
        
        all_node_masks = []
        all_edge_masks = []
        
        for class_idx in range(self.num_classes):
            class_name = self.class_labels[class_idx]
            
            if verbose:
                print(f"\nComputing GNNExplainer for class: {class_name}")
            
            class_mask = y == class_idx
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            n_samples = min(len(X_class), max_samples_per_class)
            X_subset = X_class[:n_samples]
            
            if verbose:
                print(f"  Explaining {n_samples} samples...")
            
            node_masks, edge_masks = self.gnn_explainer.explain_batch(X_subset, class_idx)
            
            gnn_results['per_class'][class_name] = {
                'node_masks': node_masks,
                'edge_masks': edge_masks,
                'mean_node_importance': node_masks.mean(axis=0),
                'std_node_importance': node_masks.std(axis=0),
                'mean_edge_importance': edge_masks.mean(axis=0),
                'std_edge_importance': edge_masks.std(axis=0)
            }
            
            all_node_masks.append(node_masks)
            all_edge_masks.append(edge_masks)
        
        if all_node_masks:
            all_nodes = np.concatenate(all_node_masks, axis=0)
            all_edges = np.concatenate(all_edge_masks, axis=0)
            
            gnn_results['node_importance'] = {
                'mean': all_nodes.mean(axis=0),
                'std': all_nodes.std(axis=0)
            }
            gnn_results['edge_importance'] = {
                'mean': all_edges.mean(axis=0),
                'std': all_edges.std(axis=0)
            }
        
        self.results['gnn_explainer'] = gnn_results
        return gnn_results
    
    def analyze_edge_importance(self, verbose=True):
        if verbose:
            print("\n" + "="*70)
            print("EDGE (FUNCTIONAL RELATIONSHIP) ANALYSIS")
            print("="*70)
        
        gnn_results = self.results.get('gnn_explainer', {})
        edge_importance = gnn_results.get('edge_importance', {})
        
        if not edge_importance:
            return {}
        
        edge_results = {
            'global': {},
            'per_class': {},
            'top_edges': []
        }
        
        mean_importance = edge_importance.get('mean', [])
        if len(mean_importance) > 0:
            edge_index_np = self.edge_index.cpu().numpy()
            top_edge_indices = np.argsort(-mean_importance)[:self.config.TOP_K_EDGES]
            
            top_edges = []
            for idx in top_edge_indices:
                if idx < edge_index_np.shape[1]:
                    src, dst = edge_index_np[0, idx], edge_index_np[1, idx]
                    src_name = self.feature_names[src] if src < len(self.feature_names) else f"Node_{src}"
                    dst_name = self.feature_names[dst] if dst < len(self.feature_names) else f"Node_{dst}"
                    
                    top_edges.append({
                        'edge_index': int(idx),
                        'source_node': int(src),
                        'target_node': int(dst),
                        'source_taxon': src_name,
                        'target_taxon': dst_name,
                        'importance': float(mean_importance[idx])
                    })
            
            edge_results['top_edges'] = top_edges
            edge_results['global'] = {
                'mean_importance': mean_importance,
                'num_edges': len(mean_importance)
            }
        
        self.results['edges'] = edge_results
        return edge_results
    
    def run_full_explanation(self, X_test, y_test, X_train=None, max_samples=20, verbose=True):
        if verbose:
            print("\n" + "="*70)
            print("RUNNING FULL EXPLANATION PIPELINE")
            print("="*70)
        
        self.explain_integrated_gradients(X_test, y_test, X_train, verbose)
        self.explain_gnn_explainer(X_test, y_test, verbose, max_samples)
        self.analyze_edge_importance(verbose)
        
        return self.results


# ==============================================================================
# AGGREGATOR WITH STATISTICS
# ==============================================================================

class ExplanationAggregator:
    """
    Aggregates explanation results across multiple experiments.
    
    Computes:
    - Mean importance
    - Standard deviation
    - Standard error (SEM)
    - 95% Confidence intervals
    """
    
    def __init__(self, feature_names, class_labels):
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.num_features = len(feature_names)
        
        # Storage
        self.all_ig_global = []
        self.all_gnn_global = []
        self.all_per_class = defaultdict(list)
        
        self.experiment_count = 0
    
    def add_experiment(self, results):
        """Add results from a single experiment."""
        self.experiment_count += 1
        
        # Global IG importance
        ig_global = results.get('integrated_gradients', {}).get('global', {})
        if ig_global:
            abs_attr = ig_global.get('abs_mean_attribution', None)
            if abs_attr is not None:
                self.all_ig_global.append(abs_attr)
        
        # Global GNN importance
        gnn_node = results.get('gnn_explainer', {}).get('node_importance', {})
        if gnn_node:
            mean_imp = gnn_node.get('mean', None)
            if mean_imp is not None:
                self.all_gnn_global.append(mean_imp)
        
        # Per-class importance from GNN
        gnn_per_class = results.get('gnn_explainer', {}).get('per_class', {})
        for class_name, class_data in gnn_per_class.items():
            score = class_data.get('mean_node_importance', None)
            if score is not None:
                self.all_per_class[class_name].append(score)
    
    def compute_aggregated_results(self):
        """
        Compute complete aggregation statistics:
        - Mean
        - Standard deviation  
        - Standard error (SEM)
        - 95% CI (percentile-based)
        """
        aggregated = {
            'global': {},
            'per_class': {}
        }
        
        # =====================================================================
        # GLOBAL AGGREGATION
        # =====================================================================
        
        # Aggregate global IG
        if self.all_ig_global:
            ig_array = np.array(self.all_ig_global)
            ig_n = ig_array.shape[0]
            ig_ddof = 1 if ig_n > 1 else 0
            aggregated['global']['ig_mean'] = ig_array.mean(axis=0)
            aggregated['global']['ig_std'] = ig_array.std(axis=0, ddof=ig_ddof)
            aggregated['global']['ig_sem'] = sem(ig_array, axis=0, ddof=ig_ddof) if ig_n > 1 else np.zeros_like(aggregated['global']['ig_mean'])
            aggregated['global']['ig_ci_lower'] = np.percentile(ig_array, 2.5, axis=0) if ig_n > 1 else aggregated['global']['ig_mean']
            aggregated['global']['ig_ci_upper'] = np.percentile(ig_array, 97.5, axis=0) if ig_n > 1 else aggregated['global']['ig_mean']
            aggregated['global']['ig_median'] = np.median(ig_array, axis=0)
        
        # Aggregate global GNN
        if self.all_gnn_global:
            gnn_array = np.array(self.all_gnn_global)
            gnn_n = gnn_array.shape[0]
            gnn_ddof = 1 if gnn_n > 1 else 0
            aggregated['global']['gnn_mean'] = gnn_array.mean(axis=0)
            aggregated['global']['gnn_std'] = gnn_array.std(axis=0, ddof=gnn_ddof)
            aggregated['global']['gnn_sem'] = sem(gnn_array, axis=0, ddof=gnn_ddof) if gnn_n > 1 else np.zeros_like(aggregated['global']['gnn_mean'])
            aggregated['global']['gnn_ci_lower'] = np.percentile(gnn_array, 2.5, axis=0) if gnn_n > 1 else aggregated['global']['gnn_mean']
            aggregated['global']['gnn_ci_upper'] = np.percentile(gnn_array, 97.5, axis=0) if gnn_n > 1 else aggregated['global']['gnn_mean']
            aggregated['global']['gnn_median'] = np.median(gnn_array, axis=0)
        
        # =====================================================================
        # PER-CLASS AGGREGATION
        # =====================================================================
        
        for class_name, scores_list in self.all_per_class.items():
            if scores_list:
                scores_array = np.array(scores_list)
                scores_n = scores_array.shape[0]
                scores_ddof = 1 if scores_n > 1 else 0
                aggregated['per_class'][class_name] = {
                    'mean': scores_array.mean(axis=0),
                    'std': scores_array.std(axis=0, ddof=scores_ddof),
                    'sem': sem(scores_array, axis=0, ddof=scores_ddof) if scores_n > 1 else np.zeros_like(scores_array.mean(axis=0)),
                    'ci_lower': np.percentile(scores_array, 2.5, axis=0) if scores_n > 1 else scores_array.mean(axis=0),
                    'ci_upper': np.percentile(scores_array, 97.5, axis=0) if scores_n > 1 else scores_array.mean(axis=0)
                }
        
        return aggregated
    
    def get_robust_rankings(self, aggregated):
        """
        Get robust feature rankings with complete statistics.
        """
        global_data = aggregated.get('global', {})
        
        # Use GNN mean for rankings if available, otherwise IG
        if 'gnn_mean' in global_data:
            mean_scores = global_data['gnn_mean']
            std_scores = global_data['gnn_std']
            sem_scores = global_data['gnn_sem']
            ci_lower = global_data.get('gnn_ci_lower', mean_scores - 1.96 * sem_scores)
            ci_upper = global_data.get('gnn_ci_upper', mean_scores + 1.96 * sem_scores)
        elif 'ig_mean' in global_data:
            mean_scores = global_data['ig_mean']
            std_scores = global_data['ig_std']
            sem_scores = global_data['ig_sem']
            ci_lower = global_data.get('ig_ci_lower', mean_scores - 1.96 * sem_scores)
            ci_upper = global_data.get('ig_ci_upper', mean_scores + 1.96 * sem_scores)
        else:
            return None
        
        df = pd.DataFrame({
            'Taxon': self.feature_names,
            'Mean_Importance': mean_scores,
            'Std': std_scores,
            'SEM': sem_scores,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })
        
        df['Rank'] = df['Mean_Importance'].rank(ascending=False).astype(int)
        df = df.sort_values('Rank')
        
        return df
    
    def save_aggregated_method_results(self, aggregated, output_dir):
        """
        Save aggregated results for GNNExplainer and IntegratedGradients separately.
        
        Creates:
        - aggregated_results/GNNExplainer.csv
        - aggregated_results/IntegratedGradients.csv
        
        Each CSV contains per-feature statistics:
        - mean, median, std, sem, 95% CI
        """
        # Create aggregated_results folder
        agg_dir = os.path.join(output_dir, 'aggregated_results')
        os.makedirs(agg_dir, exist_ok=True)
        
        global_data = aggregated.get('global', {})
        
        # Save GNNExplainer aggregated results
        if 'gnn_mean' in global_data:
            gnn_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean': global_data['gnn_mean'],
                'Median': global_data.get('gnn_median', global_data['gnn_mean']),
                'Std': global_data['gnn_std'],
                'SEM': global_data['gnn_sem'],
                'CI_Lower_95': global_data['gnn_ci_lower'],
                'CI_Upper_95': global_data['gnn_ci_upper']
            })
            # Sort by mean importance (descending)
            gnn_df = gnn_df.sort_values('Mean', ascending=False).reset_index(drop=True)
            gnn_df['Rank'] = range(1, len(gnn_df) + 1)
            # Reorder columns
            gnn_df = gnn_df[['Rank', 'Feature', 'Mean', 'Median', 'Std', 'SEM', 'CI_Lower_95', 'CI_Upper_95']]
            
            gnn_path = os.path.join(agg_dir, 'GNNExplainer.csv')
            gnn_df.to_csv(gnn_path, index=False)
            print(f"Saved aggregated GNNExplainer results to: {gnn_path}")
        
        # Save IntegratedGradients aggregated results
        if 'ig_mean' in global_data:
            ig_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean': global_data['ig_mean'],
                'Median': global_data.get('ig_median', global_data['ig_mean']),
                'Std': global_data['ig_std'],
                'SEM': global_data['ig_sem'],
                'CI_Lower_95': global_data['ig_ci_lower'],
                'CI_Upper_95': global_data['ig_ci_upper']
            })
            # Sort by mean importance (descending)
            ig_df = ig_df.sort_values('Mean', ascending=False).reset_index(drop=True)
            ig_df['Rank'] = range(1, len(ig_df) + 1)
            # Reorder columns
            ig_df = ig_df[['Rank', 'Feature', 'Mean', 'Median', 'Std', 'SEM', 'CI_Lower_95', 'CI_Upper_95']]
            
            ig_path = os.path.join(agg_dir, 'IntegratedGradients.csv')
            ig_df.to_csv(ig_path, index=False)
            print(f"Saved aggregated IntegratedGradients results to: {ig_path}")


# ==============================================================================
# RESULT SAVER
# ==============================================================================

class ExplanationResultSaver:
    """Saves explanation results."""
    
    def __init__(self, output_dir, feature_names, class_labels):
        self.output_dir = output_dir
        self.feature_names = feature_names
        self.class_labels = class_labels
        
        self.dirs = {
            'data': os.path.join(output_dir, 'explanation_data'),
            'tables': os.path.join(output_dir, 'explanation_tables')
        }
        
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
    
    def _convert_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(item) for item in obj]
        elif obj is None:
            return None
        else:
            try:
                if np.isnan(obj):
                    return None
            except Exception:
                pass
            return obj
    
    def save_all_results(self, results, experiment_num=None):
        """Save all explanation results to appropriate directories."""
        prefix = f"exp{experiment_num}_" if experiment_num else ""
        
        # Save raw pickle
        filepath = os.path.join(self.dirs['data'], f'{prefix}explanation_results.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Save node importance
        self._save_node_importance_tables(results, prefix)
        
        # Save edge importance
        self._save_edge_importance_tables(results, prefix)
        
        print(f"Results saved to: {self.output_dir}")
    
    def _save_node_importance_tables(self, results, prefix):
        # Save GNN node importance
        gnn_node = results.get('gnn_explainer', {}).get('node_importance', {})
        if gnn_node:
            gnn_mean = gnn_node.get('mean', np.zeros(len(self.feature_names)))
            gnn_std = gnn_node.get('std', np.zeros(len(self.feature_names)))
            
            df = pd.DataFrame({
                'Taxon': self.feature_names,
                'GNN_Importance': gnn_mean,
                'GNN_Std': gnn_std
            })
            df['Rank'] = df['GNN_Importance'].rank(ascending=False).astype(int)
            df = df.sort_values('Rank')
            
            filepath = os.path.join(self.dirs['tables'], f'{prefix}global_node_importance.csv')
            df.to_csv(filepath, index=False)
        
        # Save IG global importance
        ig_global = results.get('integrated_gradients', {}).get('global', {})
        if ig_global:
            ig_mean = ig_global.get('abs_mean_attribution', np.zeros(len(self.feature_names)))
            ig_std = ig_global.get('std_attribution', np.zeros(len(self.feature_names)))
            
            df = pd.DataFrame({
                'Taxon': self.feature_names,
                'IG_Importance': ig_mean,
                'IG_Std': ig_std
            })
            df['Rank'] = df['IG_Importance'].rank(ascending=False).astype(int)
            df = df.sort_values('Rank')
            
            filepath = os.path.join(self.dirs['tables'], f'{prefix}ig_global_importance.csv')
            df.to_csv(filepath, index=False)
    
    def _save_edge_importance_tables(self, results, prefix):
        edge_results = results.get('edges', {})
        top_edges = edge_results.get('top_edges', [])
        
        if top_edges:
            df = pd.DataFrame(top_edges)
            filepath = os.path.join(self.dirs['tables'], f'{prefix}top_edge_importance.csv')
            df.to_csv(filepath, index=False)


# ==============================================================================
# PER-EXPERIMENT FEATURE IMPORTANCE SAVER
# ==============================================================================

def save_per_experiment_feature_importance(results, feature_names, output_base_dir, experiment_num):
    """
    Save per-experiment feature importance for GNNExplainer and IntegratedGradients.
    
    Creates:
    - experiment_<experiment_number>/GNNExplainer/node_importance.csv
    - experiment_<experiment_number>/GNNExplainer/edge_importance.csv
    - experiment_<experiment_number>/IntegratedGradients/feature_importance.csv
    
    Args:
        results: Results dictionary from run_full_explanation
        feature_names: List of feature names
        output_base_dir: Base output directory
        experiment_num: Experiment number
    """
    # Create experiment directory structure
    exp_dir = os.path.join(output_base_dir, f'experiment_{experiment_num}')
    gnn_dir = os.path.join(exp_dir, 'GNNExplainer')
    ig_dir = os.path.join(exp_dir, 'IntegratedGradients')
    
    os.makedirs(gnn_dir, exist_ok=True)
    os.makedirs(ig_dir, exist_ok=True)
    
    # Save GNNExplainer node importance
    gnn_node = results.get('gnn_explainer', {}).get('node_importance', {})
    if gnn_node:
        gnn_mean = gnn_node.get('mean', np.zeros(len(feature_names)))
        gnn_std = gnn_node.get('std', np.zeros(len(feature_names)))
        
        gnn_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': gnn_mean,
            'Std': gnn_std
        })
        # Sort by importance (descending)
        gnn_df = gnn_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        gnn_df['Rank'] = range(1, len(gnn_df) + 1)
        gnn_df = gnn_df[['Rank', 'Feature', 'Importance', 'Std']]
        
        gnn_node_path = os.path.join(gnn_dir, 'node_importance.csv')
        gnn_df.to_csv(gnn_node_path, index=False)
        print(f"Saved GNNExplainer node importance to: {gnn_node_path}")
    
    # Save GNNExplainer edge importance
    gnn_edge = results.get('gnn_explainer', {}).get('edge_importance', {})
    edge_results = results.get('edges', {})
    top_edges = edge_results.get('top_edges', [])
    
    if gnn_edge or top_edges:
        # Save edge importance mean/std if available
        if gnn_edge:
            edge_mean = gnn_edge.get('mean', [])
            edge_std = gnn_edge.get('std', [])
            
            if len(edge_mean) > 0:
                edge_df = pd.DataFrame({
                    'Edge_Index': range(len(edge_mean)),
                    'Importance': edge_mean,
                    'Std': edge_std
                })
                edge_df = edge_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                edge_df['Rank'] = range(1, len(edge_df) + 1)
                edge_df = edge_df[['Rank', 'Edge_Index', 'Importance', 'Std']]
                
                gnn_edge_path = os.path.join(gnn_dir, 'edge_importance.csv')
                edge_df.to_csv(gnn_edge_path, index=False)
                print(f"Saved GNNExplainer edge importance to: {gnn_edge_path}")
        
        # Save top edges with taxon names
        if top_edges:
            top_edges_df = pd.DataFrame(top_edges)
            top_edges_path = os.path.join(gnn_dir, 'top_edges.csv')
            top_edges_df.to_csv(top_edges_path, index=False)
            print(f"Saved GNNExplainer top edges to: {top_edges_path}")
    
    # Save IntegratedGradients results
    ig_global = results.get('integrated_gradients', {}).get('global', {})
    if ig_global:
        ig_mean = ig_global.get('abs_mean_attribution', np.zeros(len(feature_names)))
        ig_std = ig_global.get('std_attribution', np.zeros(len(feature_names)))
        
        ig_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': ig_mean,
            'Std': ig_std
        })
        # Sort by importance (descending)
        ig_df = ig_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        ig_df['Rank'] = range(1, len(ig_df) + 1)
        ig_df = ig_df[['Rank', 'Feature', 'Importance', 'Std']]
        
        ig_path = os.path.join(ig_dir, 'feature_importance.csv')
        ig_df.to_csv(ig_path, index=False)
        print(f"Saved IntegratedGradients results to: {ig_path}")


# ==============================================================================
# MAIN INTEGRATION FUNCTION
# ==============================================================================

def run_explanation_analysis(model, edge_index, X_test, y_test, X_train,
                            feature_names, class_labels, device, output_dir,
                            experiment_num=None, verbose=True):
    """
    Run complete explanation analysis for a trained model.
    """
    config = ExplainerConfig()
    
    explainer = MicrobiomeGCNExplainer(
        model=model,
        edge_index=edge_index,
        feature_names=feature_names,
        class_labels=class_labels,
        device=device,
        config=config
    )
    
    results = explainer.run_full_explanation(
        X_test, y_test, X_train,
        max_samples=20,
        verbose=verbose
    )
    
    saver = ExplanationResultSaver(output_dir, feature_names, class_labels)
    saver.save_all_results(results, experiment_num=experiment_num)
    
    # Save per-experiment feature importance to experiment_<num>/GNNExplainer/ and experiment_<num>/IntegratedGradients/
    if experiment_num is not None:
        # Use the parent of output_dir for per-experiment results
        parent_dir = os.path.dirname(output_dir)
        save_per_experiment_feature_importance(results, feature_names, parent_dir, experiment_num)
    
    return results


if __name__ == "__main__":
    print("GCN Explainer Module loaded successfully!")
    print("\nFeatures:")
    print("  - GNNExplainer for node and edge importance")
    print("  - Integrated Gradients for attribution-based importance")
    print("  - Complete statistics: Mean, Std, SEM, 95% CI")
    print("\nOutput Structure:")
    print("  - experiment_<num>/GNNExplainer/node_importance.csv")
    print("  - experiment_<num>/GNNExplainer/edge_importance.csv")
    print("  - experiment_<num>/IntegratedGradients/feature_importance.csv")
    print("  - aggregated_results/GNNExplainer.csv")
    print("  - aggregated_results/IntegratedGradients.csv")