
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import sem
import json
import os
from collections import defaultdict


class ROCAggregator:
    """
    Aggregate and visualize ROC curves across multiple experiments.
    
    Designed
    - Mean ROC curves with confidence bands
    - Per-class and macro-average ROC
    - Statistical rigor (bootstrapped CI or standard error)
    
    Usage:
        # Initialize
        roc_agg = ROCAggregator(class_labels=['Healthy', 'Early', 'Intermediate', 'Advanced'])
        
        # Add experiments
        for exp in range(20):
            roc_agg.add_experiment(y_true, y_probs)
        
        # Generate plots
        roc_agg.plot_mean_roc_per_class(output_path='roc_per_class.png')
        roc_agg.plot_macro_average_roc(output_path='roc_macro.png')
        roc_agg.save_results(output_dir='plots/')
    """
    
    def __init__(self, class_labels, n_bootstrap=None):
        """
        Initialize ROC aggregator.
        
        Args:
            class_labels (list): Names of classes
            n_bootstrap (int): Number of bootstrap samples for CI (None=use SEM)
        """
        self.class_labels = class_labels
        self.n_classes = len(class_labels)
        self.n_bootstrap = n_bootstrap
        
        # Storage for all experiments
        self.all_tprs = defaultdict(list)  # Per-class true positive rates
        self.all_aucs = defaultdict(list)  # Per-class AUC scores
        self.mean_fpr = np.linspace(0, 1, 100)  # Common FPR points for interpolation
        
        # Storage for macro-average
        self.all_macro_tprs = []
        self.all_macro_aucs = []
        
    def add_experiment(self, y_true, y_probs):
        """
        Add results from one experiment.
        
        Args:
            y_true (array): True labels [n_samples] (integer class indices)
            y_probs (array): Predicted probabilities [n_samples, n_classes]
        """
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        
        # Compute per-class ROC
        tprs_this_exp = []
        for i in range(self.n_classes):
            # One-vs-rest: class i vs all others
            y_true_binary = (y_true == i).astype(int)
            y_score = y_probs[:, i]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Interpolate TPR at common FPR points
            tpr_interp = np.interp(self.mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0  # Ensure starts at 0
            
            # Store
            self.all_tprs[i].append(tpr_interp)
            self.all_aucs[i].append(roc_auc)
            tprs_this_exp.append(tpr_interp)
        
        # Compute macro-average for this experiment
        macro_tpr = np.mean(tprs_this_exp, axis=0)
        macro_auc = np.mean([self.all_aucs[i][-1] for i in range(self.n_classes)])
        
        self.all_macro_tprs.append(macro_tpr)
        self.all_macro_aucs.append(macro_auc)
    
    def get_mean_roc(self, class_idx):
        """
        Get mean ROC curve and confidence bands for a specific class.
        
        Args:
            class_idx (int): Class index
            
        Returns:
            dict with 'fpr', 'mean_tpr', 'std_tpr', 'ci_lower', 'ci_upper', 'mean_auc', 'std_auc'
        """
        tprs = np.array(self.all_tprs[class_idx])
        aucs = np.array(self.all_aucs[class_idx])
        
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        
        # Confidence interval (95%)
        n = len(tprs)
        se_tpr = sem(tprs, axis=0)
        ci_lower = mean_tpr - 1.96 * se_tpr
        ci_upper = mean_tpr + 1.96 * se_tpr
        
        # Clip to [0, 1]
        ci_lower = np.clip(ci_lower, 0, 1)
        ci_upper = np.clip(ci_upper, 0, 1)
        
        return {
            'fpr': self.mean_fpr,
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'all_aucs': aucs
        }
    
    def get_macro_average_roc(self):
        """Get mean macro-average ROC curve."""
        tprs = np.array(self.all_macro_tprs)
        aucs = np.array(self.all_macro_aucs)
        
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        
        n = len(tprs)
        se_tpr = sem(tprs, axis=0)
        ci_lower = mean_tpr - 1.96 * se_tpr
        ci_upper = mean_tpr + 1.96 * se_tpr
        
        ci_lower = np.clip(ci_lower, 0, 1)
        ci_upper = np.clip(ci_upper, 0, 1)
        
        return {
            'fpr': self.mean_fpr,
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'all_aucs': aucs
        }
    
    def plot_mean_roc_per_class(self, output_path, figsize=(14, 10), 
                                 show_ci=True, show_individual=False,
                                 dpi=300):
        """
        Plot mean ROC curve for each class.
        
        Args:
            output_path (str): Path to save figure
            figsize (tuple): Figure size
            show_ci (bool): Show 95% confidence bands
            show_individual (bool): Show individual experiment curves
            dpi (int): Resolution for saving
        """
        n_cols = 3
        n_rows = int(np.ceil(self.n_classes / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if self.n_classes > 1 else [axes]
        
        for class_idx in range(self.n_classes):
            ax = axes[class_idx]
            roc_data = self.get_mean_roc(class_idx)
            
            # Show individual experiments
            if show_individual:
                for tpr in self.all_tprs[class_idx]:
                    ax.plot(self.mean_fpr, tpr, color='gray', alpha=0.1, lw=0.5)
            
            # Plot mean ROC
            ax.plot(roc_data['fpr'], roc_data['mean_tpr'], 
                   color='#2E86AB', lw=2.5, 
                   label=f"Mean ROC (AUC = {roc_data['mean_auc']:.3f} ± {roc_data['std_auc']:.3f})")
            
            # Plot confidence band
            if show_ci:
                ax.fill_between(roc_data['fpr'], 
                               roc_data['ci_lower'], 
                               roc_data['ci_upper'],
                               color='#2E86AB', alpha=0.2, 
                               label='95% CI')
            
            # Plot diagonal (chance level)
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Chance')
            
            # Formatting
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title(f'{self.class_labels[class_idx]}', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for idx in range(self.n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Per-class ROC curves saved to: {output_path}")
    
    def plot_macro_average_roc(self, output_path, figsize=(8, 7), 
                                show_ci=True, show_individual=False, dpi=300):
        """
        Plot macro-average ROC curve (single plot for all classes).
        """
        roc_data = self.get_macro_average_roc()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Show individual experiments
        if show_individual:
            for tpr in self.all_macro_tprs:
                ax.plot(self.mean_fpr, tpr, color='gray', alpha=0.1, lw=0.5)
        
        # Plot mean ROC
        ax.plot(roc_data['fpr'], roc_data['mean_tpr'], 
               color='#A23B72', lw=3, 
               label=f"Mean ROC (AUC = {roc_data['mean_auc']:.3f} ± {roc_data['std_auc']:.3f})")
        
        # Plot confidence band
        if show_ci:
            ax.fill_between(roc_data['fpr'], 
                           roc_data['ci_lower'], 
                           roc_data['ci_upper'],
                           color='#A23B72', alpha=0.2, 
                           label='95% CI')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Chance (AUC = 0.5)')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('Macro-Average ROC Curve', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.tick_params(labelsize=12)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Macro-average ROC curve saved to: {output_path}")
    
    def plot_all_classes_combined(self, output_path, figsize=(10, 8), 
                                   show_ci=True, dpi=300):
        """
        Plot all classes on single plot with macro-average.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        # Plot each class
        for class_idx in range(self.n_classes):
            roc_data = self.get_mean_roc(class_idx)
            
            ax.plot(roc_data['fpr'], roc_data['mean_tpr'], 
                   color=colors[class_idx], lw=2, alpha=0.8,
                   label=f"{self.class_labels[class_idx]} (AUC = {roc_data['mean_auc']:.3f})")
            
            if show_ci:
                ax.fill_between(roc_data['fpr'], 
                               roc_data['ci_lower'], 
                               roc_data['ci_upper'],
                               color=colors[class_idx], alpha=0.1)
        
        # Plot macro-average
        macro_data = self.get_macro_average_roc()
        ax.plot(macro_data['fpr'], macro_data['mean_tpr'], 
               color='black', lw=3, linestyle='--',
               label=f"Macro-avg (AUC = {macro_data['mean_auc']:.3f})")
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.4)
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title('ROC Curves - All Classes', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, ncol=2 if self.n_classes > 8 else 1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Combined ROC curves saved to: {output_path}")
    
    def generate_publication_table(self, output_path=None):
        """
        Generate table of AUC scores.
        Returns formatted table as string and optionally saves to file.
        """
        lines = []
        lines.append("="*70)
        lines.append("ROC-AUC SCORES (Mean ± SD over experiments)")
        lines.append("="*70)
        lines.append("")
        lines.append(f"{'Class':<25} {'AUC (Mean ± SD)':<20} {'95% CI':<20}")
        lines.append("-"*70)
        
        # Per-class AUC
        for class_idx in range(self.n_classes):
            roc_data = self.get_mean_roc(class_idx)
            mean_auc = roc_data['mean_auc']
            std_auc = roc_data['std_auc']
            
            # Compute CI for AUC
            n = len(roc_data['all_aucs'])
            se_auc = sem(roc_data['all_aucs'])
            ci_lower = mean_auc - 1.96 * se_auc
            ci_upper = mean_auc + 1.96 * se_auc
            
            lines.append(
                f"{self.class_labels[class_idx]:<25} "
                f"{mean_auc:.4f} ± {std_auc:.4f}    "
                f"({ci_lower:.4f}-{ci_upper:.4f})"
            )
        
        # Macro-average
        lines.append("-"*70)
        macro_data = self.get_macro_average_roc()
        mean_auc = macro_data['mean_auc']
        std_auc = macro_data['std_auc']
        
        n = len(macro_data['all_aucs'])
        se_auc = sem(macro_data['all_aucs'])
        ci_lower = mean_auc - 1.96 * se_auc
        ci_upper = mean_auc + 1.96 * se_auc
        
        lines.append(
            f"{'Macro-Average':<25} "
            f"{mean_auc:.4f} ± {std_auc:.4f}    "
            f"({ci_lower:.4f}-{ci_upper:.4f})"
        )
        lines.append("="*70)
        
        table_text = "\n".join(lines)
        
        # Save to file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table_text)
            print(f"Table saved to: {output_path}")
        
        # Print to console
        print("\n" + table_text)
        
        return table_text
    
    def save_results(self, output_dir):
        """
        Save all results (plots, tables, raw data).
        
        Creates:
        - roc_per_class.png
        - roc_macro_average.png
        - roc_all_classes.png
        - auc_scores.txt
        - roc_curves.json (raw data)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self.plot_mean_roc_per_class(
            os.path.join(output_dir, 'roc_per_class.png'),
            show_ci=True, show_individual=False
        )
        
        self.plot_macro_average_roc(
            os.path.join(output_dir, 'roc_macro_average.png'),
            show_ci=True, show_individual=False
        )
        
        if self.n_classes <= 10:  # Only for moderate number of classes
            self.plot_all_classes_combined(
                os.path.join(output_dir, 'roc_all_classes.png'),
                show_ci=True
            )
        
        # Generate table
        self.generate_publication_table(
            os.path.join(output_dir, 'auc_scores.txt')
        )
        
        # Save raw data
        raw_data = {
            'class_labels': self.class_labels,
            'n_experiments': len(self.all_macro_aucs),
            'per_class': {},
            'macro_average': self.get_macro_average_roc()
        }
        
        for class_idx in range(self.n_classes):
            roc_data = self.get_mean_roc(class_idx)
            raw_data['per_class'][self.class_labels[class_idx]] = {
                'fpr': roc_data['fpr'].tolist(),
                'mean_tpr': roc_data['mean_tpr'].tolist(),
                'ci_lower': roc_data['ci_lower'].tolist(),
                'ci_upper': roc_data['ci_upper'].tolist(),
                'mean_auc': float(roc_data['mean_auc']),
                'std_auc': float(roc_data['std_auc']),
                'all_aucs': roc_data['all_aucs'].tolist()
            }
        
        # Convert numpy arrays to lists for JSON
        macro = raw_data['macro_average']
        for key in ['fpr', 'mean_tpr', 'ci_lower', 'ci_upper', 'all_aucs']:
            if isinstance(macro[key], np.ndarray):
                macro[key] = macro[key].tolist()
        
        # Convert numpy scalars to Python types
        for key in ['mean_auc', 'std_auc']:
            if isinstance(macro[key], (np.floating, np.integer)):
                macro[key] = float(macro[key])

        # Also convert std_tpr if present
        if 'std_tpr' in macro and isinstance(macro['std_tpr'], np.ndarray):
            macro['std_tpr'] = macro['std_tpr'].tolist()

        with open(os.path.join(output_dir, 'roc_curves.json'), 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        print(f"\nAll ROC results saved to: {output_dir}/")