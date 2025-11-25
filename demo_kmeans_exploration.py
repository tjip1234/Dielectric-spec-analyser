"""
K-means clustering exploration with different polynomial degrees and 3D visualization.

Tests various combinations of magnitude and phase polynomial degrees
to find the best feature representation for clustering pure compounds.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
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


def select_purest_sample(s1p_files):
    """Select the purest sample from a list of S1P files."""
    if not s1p_files:
        return None
    
    # Priority 1: Files containing '100%'
    for f in s1p_files:
        if '100%' in f.name or '100' in f.stem:
            return f
    
    # Priority 2: Files without percentages, ratios, or mixture indicators
    pure_candidates = []
    for f in s1p_files:
        name_lower = f.name.lower()
        if any(indicator in name_lower for indicator in ['%', 'ratio', '-50-', 'propanol', 'butoxy', 'mixture']):
            continue
        pure_candidates.append(f)
    
    if pure_candidates:
        return sorted(pure_candidates)[0]
    
    return sorted(s1p_files)[0]


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
        s1p_files = list(compound_dir.glob('*.s1p'))
        selected_file = select_purest_sample(s1p_files)
        
        if selected_file is None:
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
    mag_names = [f"mag_c{i}" for i in range(magnitude_degree, -1, -1)]
    phase_names = [f"phase_c{i}" for i in range(phase_degree, -1, -1)]
    feature_names = mag_names + phase_names
    
    return X, labels, feature_names


def cluster_and_evaluate(X, n_clusters=5, random_state=42):
    """Perform K-means clustering and calculate silhouette score."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, cluster_labels)
    
    return kmeans, X_scaled, cluster_labels, silhouette, scaler


def visualize_3d(X_scaled, labels_data, cluster_labels, n_clusters, 
                 mag_deg, phase_deg, silhouette, save_path=None):
    """Create 3D PCA visualization of clusters."""
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                  c=[colors[i]], label=f'Cluster {i}',
                  alpha=0.8, s=300, edgecolors='black', linewidth=2)
        
        # Label each point
        for j in np.where(mask)[0]:
            ax.text(X_3d[j, 0], X_3d[j, 1], X_3d[j, 2],
                   labels_data[j], fontsize=9, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, labelpad=10)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12, labelpad=10)
    
    total_var = pca.explained_variance_ratio_[:3].sum()
    ax.set_title(f'3D K-means Clustering (k={n_clusters})\n' + 
                 f'Mag Deg {mag_deg}, Phase Deg {phase_deg} | ' +
                 f'Silhouette: {silhouette:.3f} | Total Variance: {total_var:.1%}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    
    # Rotate for better view
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 3D plot saved: {save_path}")
    
    plt.show()
    
    return pca, X_3d


def visualize_2d(X_scaled, labels_data, cluster_labels, n_clusters,
                 mag_deg, phase_deg, silhouette, save_path=None):
    """Create 2D PCA visualization with labels."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[colors[i]], label=f'Cluster {i}',
                  alpha=0.7, s=250, edgecolors='black', linewidth=1.5)
    
    # Label points
    for i, name in enumerate(labels_data):
        ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]), 
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'2D K-means Clustering (k={n_clusters})\n' + 
                 f'Mag Deg {mag_deg}, Phase Deg {phase_deg} | Silhouette: {silhouette:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 2D plot saved: {save_path}")
    
    plt.show()
    
    return pca, X_2d


def print_cluster_composition(labels_data, cluster_labels, n_clusters):
    """Print which compounds are in each cluster."""
    print("\n" + "="*70)
    print("CLUSTER COMPOSITION")
    print("="*70)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_compounds = [labels_data[j] for j in range(len(labels_data)) if mask[j]]
        
        print(f"\nCluster {i} ({len(cluster_compounds)} compounds):")
        print("-" * 70)
        for compound in sorted(cluster_compounds):
            print(f"  • {compound}")


def test_polynomial_combinations(database_dir, output_dir, n_clusters=5):
    """Test different polynomial degree combinations."""
    
    degree_combos = [
        (1, 1),   # Both linear
        (1, 2),   # Linear mag, quadratic phase
        (1, 3),   # Linear mag, cubic phase
        (1, 4),   # Linear mag, quartic phase
        (1, 6),   # Linear mag, 6th order phase
        (2, 3),   # Quadratic mag, cubic phase
        (2, 4),   # Quadratic mag, quartic phase
        (2, 6),   # Quadratic mag, 6th order phase
        (1, 8),   # Higher phase complexity
        (3, 6),   # Both higher
    ]
    
    print("\n" + "="*70)
    print("TESTING POLYNOMIAL DEGREE COMBINATIONS")
    print("="*70)
    
    results = []
    all_data = []
    
    for mag_deg, phase_deg in degree_combos:
        print(f"\nTesting: Magnitude degree {mag_deg}, Phase degree {phase_deg}")
        print("-" * 70)
        
        try:
            # Load with new degrees
            X, labels_new, feature_names = load_database_samples(
                database_dir, 
                magnitude_degree=mag_deg,
                phase_degree=phase_deg,
                verbose=False
            )
            
            print(f"  Loaded {X.shape[0]} samples with {X.shape[1]} features")
            
            # Cluster and evaluate
            _, X_scaled, cluster_labels, silhouette, _ = cluster_and_evaluate(
                X, n_clusters=n_clusters
            )
            
            results.append({
                'mag_deg': mag_deg,
                'phase_deg': phase_deg,
                'silhouette': silhouette,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            })
            
            all_data.append({
                'mag_deg': mag_deg,
                'phase_deg': phase_deg,
                'X_scaled': X_scaled,
                'labels': labels_new,
                'cluster_labels': cluster_labels,
                'silhouette': silhouette
            })
            
            print(f"  ✓ Silhouette score: {silhouette:.4f}")
            print(f"  ✓ Features: {X.shape[1]} ({mag_deg+1} mag + {phase_deg+1} phase)")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - SORTED BY SILHOUETTE SCORE")
    print("="*70)
    print(f"{'Rank':<6}{'Mag Deg':<10}{'Phase Deg':<12}{'Features':<12}{'Silhouette':<12}")
    print("-" * 70)
    
    results_sorted = sorted(results, key=lambda x: x['silhouette'], reverse=True)
    
    for i, res in enumerate(results_sorted, 1):
        print(f"{i:<6}{res['mag_deg']:<10}{res['phase_deg']:<12}{res['n_features']:<12}{res['silhouette']:.4f}")
    
    return results_sorted, all_data


def main():
    print("="*70)
    print("K-MEANS CLUSTERING EXPLORATION")
    print("Pure Compounds with Different Polynomial Degrees")
    print("="*70)
    
    database_dir = Path('Database')
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    
    n_clusters = 5
    
    print(f"\n=== PART 1: Test Multiple Polynomial Combinations ===")
    results_sorted, all_data = test_polynomial_combinations(
        database_dir, output_dir, n_clusters=n_clusters
    )
    
    # Use the best combination for detailed visualization
    if results_sorted:
        best = results_sorted[0]
        magnitude_degree = best['mag_deg']
        phase_degree = best['phase_deg']
        
        print(f"\n\n=== PART 2: Detailed Analysis of Best Configuration ===")
        print(f"Using: Magnitude degree {magnitude_degree}, Phase degree {phase_degree}")
        print(f"Silhouette score: {best['silhouette']:.4f}")
        print("-" * 70)
        
        # Find the data for this configuration
        best_data = None
        for data in all_data:
            if data['mag_deg'] == magnitude_degree and data['phase_deg'] == phase_degree:
                best_data = data
                break
        
        if best_data:
            X_scaled = best_data['X_scaled']
            labels = best_data['labels']
            cluster_labels = best_data['cluster_labels']
            silhouette = best_data['silhouette']
            
            # Print composition
            print_cluster_composition(labels, cluster_labels, n_clusters)
            
            print(f"\n=== PART 3: Visualizations for Best Configuration ===")
            print("-" * 70)
            
            # 3D visualization
            print("\nGenerating 3D visualization...")
            visualize_3d(
                X_scaled, labels, cluster_labels, n_clusters,
                magnitude_degree, phase_degree, silhouette,
                save_path=output_dir / f'kmeans_3d_best_mag{magnitude_degree}_phase{phase_degree}.png'
            )
            
            # 2D visualization
            print("\nGenerating 2D visualization...")
            visualize_2d(
                X_scaled, labels, cluster_labels, n_clusters,
                magnitude_degree, phase_degree, silhouette,
                save_path=output_dir / f'kmeans_2d_best_mag{magnitude_degree}_phase{phase_degree}.png'
            )
    
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print()


if __name__ == '__main__':
    main()
