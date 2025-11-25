"""
K-means clustering on PC1 subset (right side compounds).

Focuses on clustering compounds that appear on the right side of PC1,
looking for finer distinctions within that group.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_classifier.data_loader import parse_s1p_file


def filter_frequency_range(freq_hz, s11_db, phase_deg, min_freq_hz=100e6):
    """Filter data to only include frequencies >= min_freq_hz."""
    mask = freq_hz >= min_freq_hz
    return freq_hz[mask], s11_db[mask], phase_deg[mask]


def extract_mixed_polynomial_features(freq_hz, s11_db, phase_deg, 
                                     magnitude_degree=1, phase_degree=6):
    """Extract polynomial features with different degrees for magnitude and phase."""
    freq_ghz = freq_hz / 1e9
    mag_coeffs = np.polyfit(freq_ghz, s11_db, magnitude_degree)
    phase_coeffs = np.polyfit(freq_ghz, phase_deg, phase_degree)
    return np.concatenate([mag_coeffs, phase_coeffs])


def select_purest_sample(compound_dir):
    """Select the purest sample using compound.yaml metadata."""
    yaml_path = compound_dir / 'compound.yaml'
    
    if not yaml_path.exists():
        # Fallback: use filename heuristics
        s1p_files = list(compound_dir.glob('*.s1p'))
        if not s1p_files:
            return None
        
        for f in s1p_files:
            if '100%' in f.name or '100' in f.stem:
                return f
        
        pure_candidates = []
        for f in s1p_files:
            name_lower = f.name.lower()
            if any(indicator in name_lower for indicator in ['%', 'ratio', '-50-', 'propanol', 'butoxy', 'mixture']):
                continue
            pure_candidates.append(f)
        
        if pure_candidates:
            return sorted(pure_candidates)[0]
        
        return sorted(s1p_files)[0]
    
    # Load YAML and find purest sample
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'measurements' not in data:
            return None
        
        measurements = data['measurements']
        
        # Priority 1: is_pure = true
        pure_measurements = [m for m in measurements if m.get('is_pure', False)]
        if pure_measurements:
            pure_measurements.sort(key=lambda m: m.get('purity_percent', 100), reverse=True)
            return compound_dir / pure_measurements[0]['data_file']
        
        # Priority 2: Highest purity_percent
        purity_measurements = [m for m in measurements if m.get('purity_percent') is not None]
        if purity_measurements:
            purity_measurements.sort(key=lambda m: m['purity_percent'], reverse=True)
            return compound_dir / purity_measurements[0]['data_file']
        
        # Priority 3: concentration with highest percent
        conc_measurements = [m for m in measurements if m.get('concentration') is not None]
        if conc_measurements:
            for m in conc_measurements:
                conc = m['concentration']
                if isinstance(conc, dict):
                    if 'volume_percent' in conc:
                        m['_conc_value'] = conc['volume_percent']
                    elif 'percent' in conc:
                        m['_conc_value'] = conc['percent']
                    else:
                        m['_conc_value'] = 0
                else:
                    m['_conc_value'] = 0
            
            conc_measurements.sort(key=lambda m: m.get('_conc_value', 0), reverse=True)
            if conc_measurements[0].get('_conc_value', 0) > 0:
                return compound_dir / conc_measurements[0]['data_file']
        
        # Fallback: first measurement
        if measurements:
            return compound_dir / measurements[0]['data_file']
        
        return None
        
    except Exception:
        s1p_files = list(compound_dir.glob('*.s1p'))
        return sorted(s1p_files)[0] if s1p_files else None


def load_database_samples(database_dir, min_freq_hz=100e6, 
                         magnitude_degree=1, phase_degree=6, verbose=True):
    """Load pure compound samples and extract features."""
    database_path = Path(database_dir)
    
    if not database_path.exists():
        raise FileNotFoundError(f"Database directory not found: {database_dir}")
    
    X_list = []
    labels = []
    
    for compound_dir in sorted(database_path.iterdir()):
        if not compound_dir.is_dir():
            continue
            
        compound_name = compound_dir.name
        selected_file = select_purest_sample(compound_dir)
        
        if selected_file is None:
            if verbose:
                print(f"⊗ No suitable file in {compound_name}")
            continue
        
        try:
            freq_hz, s11_db, phase_deg = parse_s1p_file(selected_file)
            freq_hz, s11_db, phase_deg = filter_frequency_range(
                freq_hz, s11_db, phase_deg, min_freq_hz
            )
            
            features = extract_mixed_polynomial_features(
                freq_hz, s11_db, phase_deg,
                magnitude_degree=magnitude_degree,
                phase_degree=phase_degree
            )
            
            X_list.append(features)
            labels.append(compound_name)
            
            if verbose:
                print(f"✓ {compound_name}")
                
        except Exception as e:
            if verbose:
                print(f"✗ Failed: {compound_name} - {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError("No samples loaded")
    
    X = np.vstack(X_list)
    
    return X, labels


def filter_pc1_subset(X, labels, threshold=0.0, side='right'):
    """
    Filter samples based on PC1 position.
    
    Args:
        X: Feature matrix
        labels: Compound labels
        threshold: PC1 threshold value
        side: 'right' or 'left' - which side to keep
        
    Returns:
        X_subset, labels_subset, mask, pca, X_pca
    """
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Filter based on PC1
    pc1_values = X_pca[:, 0]
    
    if side == 'right':
        mask = pc1_values > threshold
    else:
        mask = pc1_values < threshold
    
    X_subset = X[mask]
    labels_subset = [labels[i] for i in range(len(labels)) if mask[i]]
    
    return X_subset, labels_subset, mask, pca, X_pca, scaler


def cluster_subset(X_subset, n_clusters=2, random_state=42):
    """Cluster the subset of data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, cluster_labels)
    
    return kmeans, X_scaled, cluster_labels, silhouette, scaler


def visualize_full_and_subset(X_full, labels_full, X_pca_full, mask_subset, 
                               cluster_labels_subset, n_clusters, 
                               mag_deg, phase_deg, silhouette, save_path=None):
    """
    Visualize both the full dataset and the subset clustering.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left plot: Full dataset with subset highlighted
    pc1 = X_pca_full[:, 0]
    pc2 = X_pca_full[:, 1]
    
    # Plot non-subset in gray
    ax1.scatter(pc1[~mask_subset], pc2[~mask_subset], 
               c='lightgray', s=200, alpha=0.5, edgecolors='black', 
               linewidth=1, label='Not in subset')
    
    # Plot subset in blue
    ax1.scatter(pc1[mask_subset], pc2[mask_subset], 
               c='blue', s=200, alpha=0.7, edgecolors='black', 
               linewidth=1.5, label='PC1 > 0 (subset)')
    
    # Add vertical line at PC1=0
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Label all points
    for i, name in enumerate(labels_full):
        ax1.annotate(name, (pc1[i], pc2[i]), 
                    fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('Full Dataset - Subset Selection (PC1 > 0)', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Subset with clusters
    labels_subset = [labels_full[i] for i in range(len(labels_full)) if mask_subset[i]]
    X_subset = X_pca_full[mask_subset]
    
    # Re-compute PCA on subset only for better visualization
    pca_subset = PCA(n_components=2)
    X_subset_pca = pca_subset.fit_transform(X_subset)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask_cluster = cluster_labels_subset == i
        ax2.scatter(X_subset_pca[mask_cluster, 0], X_subset_pca[mask_cluster, 1], 
                   c=[colors[i]], s=250, alpha=0.8, edgecolors='black', 
                   linewidth=2, label=f'Cluster {i}')
    
    # Label subset points
    for i, name in enumerate(labels_subset):
        ax2.annotate(name, (X_subset_pca[i, 0], X_subset_pca[i, 1]), 
                    fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', alpha=0.8, edgecolor='gray'))
    
    var1 = pca_subset.explained_variance_ratio_[0]
    var2 = pca_subset.explained_variance_ratio_[1]
    ax2.set_xlabel(f'PC1 ({var1:.1%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({var2:.1%} variance)', fontsize=12)
    ax2.set_title(f'Subset Clustering (k={n_clusters})\n' + 
                 f'Mag Deg {mag_deg}, Phase Deg {phase_deg} | Silhouette: {silhouette:.3f}',
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    
    plt.show()


def print_cluster_composition(labels_data, cluster_labels, n_clusters):
    """Print which compounds are in each cluster."""
    print("\n" + "="*70)
    print("SUBSET CLUSTER COMPOSITION")
    print("="*70)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_compounds = [labels_data[j] for j in range(len(labels_data)) if mask[j]]
        
        print(f"\nCluster {i} ({len(cluster_compounds)} compounds):")
        print("-" * 70)
        for compound in sorted(cluster_compounds):
            print(f"  • {compound}")


def main():
    print("="*70)
    print("K-MEANS CLUSTERING - PC1 SUBSET ANALYSIS")
    print("Clustering compounds on the right side of PC1")
    print("="*70)
    
    database_dir = Path('Database')
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    
    magnitude_degree = 1
    phase_degree = 6
    
    print(f"\nStep 1: Loading all samples...")
    print("-" * 70)
    
    X, labels = load_database_samples(
        database_dir,
        magnitude_degree=magnitude_degree,
        phase_degree=phase_degree,
        verbose=True
    )
    
    print(f"\n✓ Loaded {len(labels)} compounds")
    
    print(f"\nStep 2: Filtering to PC1 > 0 subset...")
    print("-" * 70)
    
    X_subset, labels_subset, mask, pca_full, X_pca_full, scaler_full = filter_pc1_subset(
        X, labels, threshold=0.0, side='right'
    )
    
    print(f"✓ Full dataset: {len(labels)} compounds")
    print(f"✓ Subset (PC1 > 0): {len(labels_subset)} compounds")
    print(f"\nCompounds in subset:")
    for label in labels_subset:
        print(f"  • {label}")
    
    # Test multiple k values on subset
    print(f"\nStep 3: Testing different k values on subset...")
    print("=" * 70)
    
    k_values = [2, 3, 4]
    results = []
    
    for k in k_values:
        print(f"\nTrying k={k}...")
        print("-" * 70)
        
        kmeans, X_subset_scaled, cluster_labels, silhouette, scaler_subset = cluster_subset(
            X_subset, n_clusters=k
        )
        
        print(f"✓ Silhouette score: {silhouette:.4f}")
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'kmeans': kmeans,
            'X_scaled': X_subset_scaled,
            'cluster_labels': cluster_labels,
            'scaler': scaler_subset
        })
        
        # Print composition for this k
        print_cluster_composition(labels_subset, cluster_labels, k)
    
    # Use all k values for visualization
    print(f"\nStep 4: Generating visualizations for all k values...")
    print("=" * 70)
    
    for result in results:
        k = result['k']
        cluster_labels = result['cluster_labels']
        silhouette = result['silhouette']
        
        print(f"\nVisualizing k={k} (silhouette: {silhouette:.4f})...")
        
        visualize_full_and_subset(
            X, labels, X_pca_full, mask, 
            cluster_labels, k,
            magnitude_degree, phase_degree, silhouette,
            save_path=output_dir / f'kmeans_pc1_subset_k{k}.png'
        )
    
    print("\n" + "="*70)
    print("SUBSET ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print()


if __name__ == '__main__':
    main()
