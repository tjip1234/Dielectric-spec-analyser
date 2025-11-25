"""
K-means clustering demo for dielectric spectroscopy data.

Uses polynomial features with:
- Degree 6 for phase (captures complex frequency-dependent behavior)
- Degree 1 for magnitude (simple linear trend)
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
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
    """
    Extract polynomial features with different degrees for magnitude and phase.
    
    Args:
        freq_hz: Frequency array in Hz
        s11_db: S11 magnitude in dB
        phase_deg: S11 phase in degrees
        magnitude_degree: Polynomial degree for magnitude (default 1)
        phase_degree: Polynomial degree for phase (default 6)
        
    Returns:
        Combined feature vector [mag_coeffs, phase_coeffs]
    """
    # Normalize frequency to GHz for numerical stability
    freq_ghz = freq_hz / 1e9
    
    # Fit polynomials with different degrees
    mag_coeffs = np.polyfit(freq_ghz, s11_db, magnitude_degree)
    phase_coeffs = np.polyfit(freq_ghz, phase_deg, phase_degree)
    
    # Combine into single feature vector
    return np.concatenate([mag_coeffs, phase_coeffs])


def select_purest_sample(s1p_files):
    """
    Select the purest sample from a list of S1P files.
    Prioritizes files with '100%' or the compound name itself.
    Avoids files with percentages, 'ratio', or mixture indicators.
    
    Args:
        s1p_files: List of Path objects to S1P files
        
    Returns:
        Selected S1P file Path or None
    """
    if not s1p_files:
        return None
    
    # Priority 1: Files containing '100%'
    for f in s1p_files:
        if '100%' in f.name or '100' in f.stem:
            return f
    
    # Priority 2: Files that don't contain percentages, ratios, or mixture indicators
    pure_candidates = []
    for f in s1p_files:
        name_lower = f.name.lower()
        # Skip if contains percentage, ratio, or mixture indicators
        if any(indicator in name_lower for indicator in ['%', 'ratio', '-50-', 'propanol', 'butoxy', 'mixture']):
            continue
        pure_candidates.append(f)
    
    # From pure candidates, prefer files matching the directory name
    if pure_candidates:
        # If only one candidate, use it
        if len(pure_candidates) == 1:
            return pure_candidates[0]
        # Otherwise pick the first one (alphabetically sorted)
        return sorted(pure_candidates)[0]
    
    # Fallback: return the first file alphabetically
    return sorted(s1p_files)[0]


def load_database_samples(database_dir, min_freq_hz=100e6, 
                         magnitude_degree=1, phase_degree=6, verbose=True):
    """
    Load pure compound samples from database and extract features.
    Only loads one sample per compound (the purest available).
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        labels: List of compound names
        feature_names: List of feature names
    """
    database_path = Path(database_dir)
    
    if not database_path.exists():
        raise FileNotFoundError(f"Database directory not found: {database_dir}")
    
    X_list = []
    labels = []
    
    # Iterate through compound directories
    for compound_dir in sorted(database_path.iterdir()):
        if not compound_dir.is_dir():
            continue
            
        compound_name = compound_dir.name
        
        # Find all .s1p files in this directory
        s1p_files = list(compound_dir.glob('*.s1p'))
        
        # Select only the purest sample
        selected_file = select_purest_sample(s1p_files)
        
        if selected_file is None:
            if verbose:
                print(f"⊗ No suitable file found in {compound_name}")
            continue
        
        try:
            # Load and filter data
            freq_hz, s11_db, phase_deg = parse_s1p_file(selected_file)
            freq_hz, s11_db, phase_deg = filter_frequency_range(
                freq_hz, s11_db, phase_deg, min_freq_hz
            )
            
            # Extract features
            features = extract_mixed_polynomial_features(
                freq_hz, s11_db, phase_deg,
                magnitude_degree=magnitude_degree,
                phase_degree=phase_degree
            )
            
            X_list.append(features)
            labels.append(compound_name)
            
            if verbose:
                print(f"✓ Loaded: {compound_name}/{selected_file.name}")
                
        except Exception as e:
            if verbose:
                print(f"✗ Failed to load {selected_file}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError("No samples loaded from database")
    
    X = np.vstack(X_list)
    
    # Generate feature names
    mag_names = [f"mag_coeff_{i}" for i in range(magnitude_degree, -1, -1)]
    phase_names = [f"phase_coeff_{i}" for i in range(phase_degree, -1, -1)]
    feature_names = mag_names + phase_names
    
    return X, labels, feature_names


def perform_kmeans_clustering(X, labels, n_clusters=5, random_state=42):
    """
    Perform K-means clustering on feature matrix.
    
    Returns:
        kmeans: Fitted KMeans object
        X_scaled: Scaled feature matrix
        scaler: Fitted StandardScaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    
    return kmeans, X_scaled, scaler


def visualize_clusters_2d(X_scaled, labels_data, cluster_labels, n_clusters, 
                          save_path=None):
    """
    Visualize clusters in 2D using first two principal components.
    Includes compound name labels on each point.
    """
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[colors[i]], label=f'Cluster {i}',
                  alpha=0.7, s=200, edgecolors='black', linewidth=1.0)
    
    # Label each point with compound name
    for i, name in enumerate(labels_data):
        ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]), 
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Pure Compounds - K-means Clustering\n' + 
                 f'Phase Poly Degree 6, Magnitude Poly Degree 1',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster visualization saved to {save_path}")
    
    plt.show()
    
    return pca, X_2d


def visualize_feature_space(X_scaled, labels_data, cluster_labels, feature_names,
                           save_path=None):
    """
    Visualize using magnitude slope vs phase curvature.
    """
    # For degree 1 magnitude: [a1, a0] - slope is a1
    # For degree 6 phase: [a6, a5, a4, a3, a2, a1, a0] - quadratic term is a4
    
    mag_slope = X_scaled[:, 0]  # First coefficient (slope) of magnitude
    phase_quadratic = X_scaled[:, 3]  # a4 term of phase (quadratic curvature)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(phase_quadratic[mask], mag_slope[mask],
                  c=[colors[i]], label=f'Cluster {i}',
                  alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Phase Quadratic Coefficient (a₄) [normalized]', fontsize=12)
    ax.set_ylabel('Magnitude Slope (a₁) [normalized]', fontsize=12)
    ax.set_title('K-means Clustering in Feature Space\n' +
                 'Phase Curvature vs Magnitude Slope',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Feature space visualization saved to {save_path}")
    
    plt.show()


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """
    Find optimal number of clusters using silhouette score.
    
    Returns:
        scores: List of silhouette scores for each k
        best_k: Optimal k value
    """
    scores = []
    k_values = list(k_range)
    
    print("\nEvaluating different k values:")
    print("-" * 70)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f"  k={k}: silhouette score = {score:.4f}")
    
    best_idx = np.argmax(scores)
    best_k = k_values[best_idx]
    best_score = scores[best_idx]
    
    print(f"\n✓ Best k: {best_k} (silhouette score: {best_score:.4f})")
    
    return scores, best_k, k_values


def plot_silhouette_analysis(k_values, scores, save_path=None):
    """
    Plot silhouette scores for different k values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, scores, marker='o', linewidth=2, markersize=8)
    
    # Mark the best k
    best_idx = np.argmax(scores)
    best_k = k_values[best_idx]
    best_score = scores[best_idx]
    
    ax.plot(best_k, best_score, marker='*', markersize=20, 
            color='red', label=f'Best k={best_k}')
    
    ax.set_xlabel('Number of clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette score', fontsize=12)
    ax.set_title('Optimal k Selection for Pure Compounds\n' +
                 'Higher silhouette score = better separated clusters',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Silhouette analysis saved to {save_path}")
    
    plt.show()


def print_cluster_composition(labels_data, cluster_labels, n_clusters):
    """
    Print which compounds are in each cluster.
    """
    print("\n" + "="*70)
    print("CLUSTER COMPOSITION")
    print("="*70)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_samples = [labels_data[j] for j in range(len(labels_data)) if mask[j]]
        
        print(f"\nCluster {i} ({len(cluster_samples)} compounds):")
        print("-" * 70)
        
        for compound in sorted(cluster_samples):
            print(f"  - {compound}")


def main():
    print("="*70)
    print("K-MEANS CLUSTERING FOR DIELECTRIC SPECTROSCOPY")
    print("="*70)
    print("Feature extraction:")
    print("  - Magnitude: Polynomial degree 1 (linear trend)")
    print("  - Phase: Polynomial degree 6 (complex behavior)")
    print("="*70)
    
    # Configuration
    database_dir = Path('Database')
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    
    magnitude_degree = 1
    phase_degree = 6
    min_freq_hz = 100e6
    
    print(f"\nStep 1: Loading pure compound samples from {database_dir}...")
    print("-" * 70)
    
    try:
        X, labels, feature_names = load_database_samples(
            database_dir,
            min_freq_hz=min_freq_hz,
            magnitude_degree=magnitude_degree,
            phase_degree=phase_degree,
            verbose=True
        )
        
        print(f"\n✓ Loaded {X.shape[0]} samples")
        print(f"✓ Feature vector size: {X.shape[1]}")
        print(f"  - Magnitude coefficients: {magnitude_degree + 1}")
        print(f"  - Phase coefficients: {phase_degree + 1}")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nStep 2: Standardizing features...")
    print("-" * 70)
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"✓ Features standardized")
        
    except Exception as e:
        print(f"\n✗ Error standardizing features: {e}")
        return
    
    print(f"\nStep 3: Finding optimal number of clusters...")
    print("-" * 70)
    
    try:
        # Find optimal k
        scores, best_k, k_values = find_optimal_k(X_scaled, k_range=range(2, 11))
        
        # Plot silhouette analysis
        plot_silhouette_analysis(
            k_values, scores,
            save_path=output_dir / 'silhouette_analysis.png'
        )
        
    except Exception as e:
        print(f"\n⚠ Warning: Could not perform k selection: {e}")
        best_k = 5  # fallback
    
    print(f"\nStep 4: Performing K-means clustering (k={best_k})...")
    print("-" * 70)
    
    try:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, cluster_labels)
        
        print(f"✓ Clustering complete")
        print(f"✓ Silhouette score: {sil_score:.4f}")
        print(f"✓ Inertia (sum of squared distances): {kmeans.inertia_:.2f}")
        
        # Print cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} compounds")
            
    except Exception as e:
        print(f"\n✗ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nStep 5: Generating visualizations...")
    print("-" * 70)
    
    try:
        # PCA visualization
        pca, X_2d = visualize_clusters_2d(
            X_scaled, labels, cluster_labels, best_k,
            save_path=output_dir / 'kmeans_pca_projection.png'
        )
        
        # Feature space visualization
        visualize_feature_space(
            X_scaled, labels, cluster_labels, feature_names,
            save_path=output_dir / 'kmeans_feature_space.png'
        )
        
    except Exception as e:
        print(f"\n⚠ Warning: Could not generate all plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Print cluster composition
    print_cluster_composition(labels, cluster_labels, best_k)
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print()


if __name__ == '__main__':
    main()
