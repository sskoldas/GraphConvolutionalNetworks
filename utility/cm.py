import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# ============================================================================
# CONFUSION MATRIX AGGREGATOR MODULE
# ============================================================================
class ConfusionMatrixAggregator:
    """
    Aggregates confusion matrices across multiple experiments.
    
    Since test set has same size class labels in each experiment,
    we sum the confusion matrices directly.
    """
    
    def __init__(self, class_labels):
        self.class_labels = class_labels
        self.n_classes = len(class_labels)
        self.all_cms = []
        self.test_distributions = []
        self.all_metrics = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_weighted': [],
            'recall_weighted': [],
            'f1_weighted': []
        }
    
    def add_experiment(self, y_true, y_pred):
        """Add results from one experiment."""
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.n_classes))
        self.all_cms.append(cm)
        
        # Macro: treats all classes equally
        self.all_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        self.all_metrics['precision_macro'].append(
            precision_score(y_true, y_pred, average='macro', zero_division=0))
        self.all_metrics['recall_macro'].append(
            recall_score(y_true, y_pred, average='macro', zero_division=0))
        self.all_metrics['f1_macro'].append(
            f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Weighted: accounts for class frequency
        self.all_metrics['precision_weighted'].append(
            precision_score(y_true, y_pred, average='weighted', zero_division=0))
        self.all_metrics['recall_weighted'].append(
            recall_score(y_true, y_pred, average='weighted', zero_division=0))
        self.all_metrics['f1_weighted'].append(
            f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Store class distribution of test set
        unique, counts = np.unique(y_true, return_counts=True)
        self.test_distributions.append(dict(zip(unique.tolist(), counts.tolist())))

    def get_summed_cm(self):
        """Return sum of all confusion matrices."""
        return np.sum(self.all_cms, axis=0)
    
    def _calculate_ci(self, values, confidence=0.95, clip_bounds=(0, 1)):
        """
        Calculate mean and confidence interval.
        
        Args:
            values: List of values
            confidence: Confidence level (default 0.95 for 95% CI)
            clip_bounds: Tuple (min, max) to clip CI bounds. Use None for no clipping.
                        Default (0, 1) for proportions/percentages.
        """
        n = len(values)
        mean = np.mean(values)
        
        # Handle edge cases
        if n < 2:
            return mean, mean, mean
        
        se = stats.sem(values)
        
        # If standard error is 0 (all values identical), CI equals mean
        if se == 0 or np.isnan(se):
            return mean, mean, mean
        
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        ci_lower, ci_upper = ci[0], ci[1]
        
        # Clip to valid bounds if specified
        if clip_bounds is not None:
            ci_lower = max(clip_bounds[0], ci_lower)
            ci_upper = min(clip_bounds[1], ci_upper)
        
        return mean, ci_lower, ci_upper
    
    def get_mean_ci_cm(self, normalize='true', confidence=0.95):
        """
        Return mean and confidence interval of confusion matrices.
        
        normalize options:
        - None: raw counts
        - 'true': normalize by row (recall)
        - 'pred': normalize by column (precision)
        """
        if normalize is None:
            cms_to_use = [cm.astype(float) for cm in self.all_cms]
        elif normalize == 'true':
            # Row normalization - shows RECALL per class
            cms_to_use = []
            for cm in self.all_cms:
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cms_to_use.append(cm.astype(float) / row_sums)
        elif normalize == 'pred':
            # Column normalization - shows PRECISION per class
            cms_to_use = []
            for cm in self.all_cms:
                col_sums = cm.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1
                cms_to_use.append(cm.astype(float) / col_sums)
        
        # Calculate mean and CI for each cell
        mean_cm = np.zeros((self.n_classes, self.n_classes))
        ci_lower_cm = np.zeros((self.n_classes, self.n_classes))
        ci_upper_cm = np.zeros((self.n_classes, self.n_classes))
        
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                cell_values = [cm[i, j] for cm in cms_to_use]
                mean, ci_low, ci_high = self._calculate_ci(cell_values, confidence)
                mean_cm[i, j] = mean
                ci_lower_cm[i, j] = ci_low
                ci_upper_cm[i, j] = ci_high
        
        return mean_cm, ci_lower_cm, ci_upper_cm
    
    def get_aggregated_metrics(self, confidence=0.95):
        """Return mean with 95% CI for all metrics."""
        results = {}
        for metric, values in self.all_metrics.items():
            if len(values) > 0:
                mean, ci_low, ci_high = self._calculate_ci(values, confidence)
                results[metric] = {
                    'mean': mean,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'std': np.std(values, ddof=1) if len(values) > 1 else 0
                }
        return results
    
    def print_summary(self):
        """Print aggregated results summary with 95% CI."""
        metrics = self.get_aggregated_metrics()
        n_exp = len(self.all_cms)
        
        print("\n" + "="*70)
        print(f"AGGREGATED RESULTS OVER {n_exp} EXPERIMENTS")
        print("="*70)
        
        # Print class distribution
        self.print_class_distribution()
        
        # Print metrics table
        print("\n" + "-"*70)
        print(f"{'Metric':<25} {'Mean':>10} {'95% CI':>25}")
        print("-"*70)
        
        # Group metrics by type
        print("\nMacro-averaged (all classes weighted equally):")
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            if metric in metrics:
                vals = metrics[metric]
                ci_str = f"({vals['ci_lower']*100:.1f} - {vals['ci_upper']*100:.1f})"
                display_name = metric.replace('_macro', '').capitalize()
                print(f"  {display_name:<23} {vals['mean']*100:>9.1f}% {ci_str:>25}")
        
        print("\nWeighted-averaged (accounts for class frequency):")
        for metric in ['precision_weighted', 'recall_weighted', 'f1_weighted']:
            if metric in metrics:
                vals = metrics[metric]
                ci_str = f"({vals['ci_lower']*100:.1f} - {vals['ci_upper']*100:.1f})"
                display_name = metric.replace('_weighted', '').capitalize()
                print(f"  {display_name:<23} {vals['mean']*100:>9.1f}% {ci_str:>25}")
        
        print("\n" + "-"*70)
        print("\nSummed Confusion Matrix:")
        print(self.get_summed_cm())
    
    def print_class_distribution(self):
        """Show class imbalance in test sets."""
        if not self.test_distributions:
            print("\nNo test distribution data available.")
            return
        
        print("\nTest Set Class Distribution:")
        avg_dist = defaultdict(list)
        for dist in self.test_distributions:
            for cls, count in dist.items():
                avg_dist[cls].append(count)
        
        # Calculate total samples
        total_samples = sum(np.mean(avg_dist[cls]) for cls in range(self.n_classes))
        
        for cls in range(self.n_classes):
            counts = avg_dist.get(cls, [0])
            mean_count = np.mean(counts)
            percentage = (mean_count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  {self.class_labels[cls]:<15}: {mean_count:>6.1f} samples ({percentage:>5.1f}%)")
    
    def plot_confusion_matrix(self, output_path, title=None, normalize='true', 
                              annot_fontsize=14, label_fontsize=14, title_fontsize=16,
                              tick_fontsize=14, figsize=(10, 8)):
        """
        Plot and save the aggregated confusion matrix with mean and 95% CI.
        
        Args:
            output_path: Path to save the plot
            title: Plot title (auto-generated if None)
            normalize: 'true' (row), 'pred' (column), or None (raw counts)
            annot_fontsize: Font size for cell annotations (default 14)
            label_fontsize: Font size for axis labels (default 14)
            title_fontsize: Font size for title (default 16)
            tick_fontsize: Font size for tick labels (default 12)
            figsize: Figure size tuple (default (10, 8))
        """
        mean_cm, ci_lower_cm, ci_upper_cm = self.get_mean_ci_cm(normalize=normalize)
        n_exp = len(self.all_cms)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create annotation labels showing mean with 95% CI as percentages
        labels = np.array([
            [f"{mean_cm[i,j]*100:.1f}%\n({ci_lower_cm[i,j]*100:.1f}-{ci_upper_cm[i,j]*100:.1f})" 
             for j in range(self.n_classes)] 
            for i in range(self.n_classes)
        ])
        
        # Plot heatmap with bigger annotation font
        sns.heatmap(mean_cm, annot=labels, fmt='', cmap='Blues',
                   xticklabels=self.class_labels, 
                   yticklabels=self.class_labels, 
                   ax=ax, vmin=0, vmax=1,
                   annot_kws={'size': annot_fontsize, 'weight': 'bold'})
        
        # Set axis label sizes
        ax.set_xlabel('Predicted Label', fontsize=label_fontsize)
        ax.set_ylabel('True Label', fontsize=label_fontsize)
        
        # Set tick label sizes
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        if title is None:
            title = f'Mean Confusion Matrix with 95% CI (n={n_exp} experiments)'
        ax.set_title(title, fontsize=title_fontsize)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Confusion matrix plot saved to: {output_path}")
    
    def save_results(self, output_dir):
        """Save all aggregated results to files."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summed confusion matrix
        summed_cm = self.get_summed_cm()
        np.savetxt(os.path.join(output_dir, 'aggregated_cm_summed.csv'), 
                   summed_cm, delimiter=',', fmt='%d',
                   header=','.join(self.class_labels), comments='')
        
        # Save mean and CI confusion matrices
        mean_cm, ci_lower_cm, ci_upper_cm = self.get_mean_ci_cm(normalize='true')
        np.savetxt(os.path.join(output_dir, 'aggregated_cm_mean.csv'), 
                   mean_cm, delimiter=',', fmt='%.4f',
                   header=','.join(self.class_labels), comments='')
        np.savetxt(os.path.join(output_dir, 'aggregated_cm_ci_lower.csv'), 
                   ci_lower_cm, delimiter=',', fmt='%.4f',
                   header=','.join(self.class_labels), comments='')
        np.savetxt(os.path.join(output_dir, 'aggregated_cm_ci_upper.csv'), 
                   ci_upper_cm, delimiter=',', fmt='%.4f',
                   header=','.join(self.class_labels), comments='')
        
        # Save aggregated metrics
        metrics = self.get_aggregated_metrics()
        with open(os.path.join(output_dir, 'aggregated_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save all individual confusion matrices
        all_cms_list = [cm.tolist() for cm in self.all_cms]
        with open(os.path.join(output_dir, 'all_confusion_matrices.json'), 'w') as f:
            json.dump(all_cms_list, f, indent=2)
        
        # Save class distribution
        with open(os.path.join(output_dir, 'class_distribution.json'), 'w') as f:
            json.dump(self.test_distributions, f, indent=2)
        
        # Save publication table
        self.generate_publication_table(os.path.join(output_dir, 'publication_table.txt'))
        
        print(f"Aggregated results saved to: {output_dir}")
    
    def generate_publication_table(self, output_path=None):
        """Generate a publication-ready table as text."""
        metrics = self.get_aggregated_metrics()
        n_exp = len(self.all_cms)
        
        lines = []
        lines.append(f"Table X: Classification Performance (n={n_exp} iterations)")
        lines.append("")
        lines.append("Test Set Class Distribution:")
        
        # Class distribution
        avg_dist = defaultdict(list)
        for dist in self.test_distributions:
            for cls, count in dist.items():
                avg_dist[cls].append(count)
        
        total_samples = sum(np.mean(avg_dist[cls]) for cls in range(self.n_classes))
        for cls in range(self.n_classes):
            counts = avg_dist.get(cls, [0])
            mean_count = np.mean(counts)
            pct = (mean_count / total_samples) * 100 if total_samples > 0 else 0
            lines.append(f"  {self.class_labels[cls]:<15}: {mean_count:>6.1f} samples ({pct:>5.1f}%)")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"{'Metric':<20} {'Macro Mean (95% CI)':<25} {'Weighted Mean (95% CI)':<25}")
        lines.append("-" * 70)
        
        # Metrics
        for base_metric in ['precision', 'recall', 'f1']:
            macro_key = f"{base_metric}_macro"
            weighted_key = f"{base_metric}_weighted"
            
            macro_vals = metrics.get(macro_key, {})
            weighted_vals = metrics.get(weighted_key, {})
            
            macro_str = f"{macro_vals.get('mean', 0)*100:.1f}% ({macro_vals.get('ci_lower', 0)*100:.1f}-{macro_vals.get('ci_upper', 0)*100:.1f})"
            weighted_str = f"{weighted_vals.get('mean', 0)*100:.1f}% ({weighted_vals.get('ci_lower', 0)*100:.1f}-{weighted_vals.get('ci_upper', 0)*100:.1f})"
            
            lines.append(f"{base_metric.capitalize():<20} {macro_str:<25} {weighted_str:<25}")
        
        # Accuracy
        acc_vals = metrics.get('accuracy', {})
        acc_str = f"{acc_vals.get('mean', 0)*100:.1f}% ({acc_vals.get('ci_lower', 0)*100:.1f}-{acc_vals.get('ci_upper', 0)*100:.1f})"
        lines.append(f"{'Accuracy':<20} {acc_str:<25} {acc_str:<25}")
        
        lines.append("-" * 70)
        lines.append("")
        lines.append("      Evaluation performed on original imbalanced test distribution.")
        lines.append("      95% CI calculated using t-distribution.")
        
        table_text = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            # Create directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(table_text)
            print(f"Publication table saved to: {output_path}")
        
        # Also print to console
        print("\n" + table_text)
        
        return table_text