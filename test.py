import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
import time

# ==========================================
# HAND-WRITTEN K-MEANS IMPLEMENTATION
# ==========================================


class KMeansHandwritten:
    """
    Hand-written implementation of K-means clustering algorithm
    """

    def __init__(self, n_clusters=3, max_iters=100, tolerance=1e-4, random_state=None):
        """
        Initialize K-means parameters

        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        max_iters : int
            Maximum number of iterations
        tolerance : float
            Convergence threshold
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X):
        """
        Compute K-means clustering

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Step 1: Initialize centroids randomly from data points
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()

        # Iterate until convergence or max iterations
        for iteration in range(self.max_iters):
            # Step 2: Assign each point to nearest centroid
            labels = self._assign_clusters(X)

            # Step 3: Update centroids based on cluster means
            new_centroids = np.array(
                [X[labels == k].mean(axis=0) for k in range(self.n_clusters)]
            )

            # Step 4: Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.n_iter_ = iteration + 1

            if centroid_shift < self.tolerance:
                break

        # Final cluster assignment
        self.labels_ = self._assign_clusters(X)

        # Calculate inertia (within-cluster sum of squares)
        self.inertia_ = self._calculate_inertia(X)

        return self

    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid

        Parameters:
        -----------
        X : array-like
            Data points

        Returns:
        --------
        labels : array
            Cluster assignment for each point
        """
        # Calculate Euclidean distance from each point to each centroid
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))

        # Assign to nearest centroid
        labels = np.argmin(distances, axis=1)

        return labels

    def _calculate_inertia(self, X):
        """
        Calculate within-cluster sum of squares

        Parameters:
        -----------
        X : array-like
            Data points

        Returns:
        --------
        inertia : float
            Sum of squared distances to nearest centroid
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                inertia += ((cluster_points - self.centroids[k]) ** 2).sum()
        return inertia

    def predict(self, X):
        """
        Predict cluster labels for new data

        Parameters:
        -----------
        X : array-like
            New data points

        Returns:
        --------
        labels : array
            Predicted cluster labels
        """
        return self._assign_clusters(X)


# ==========================================
# GENERATE 2D DATASET
# ==========================================


def generate_2d_dataset(n_samples=300, n_centers=4, random_state=42):
    """
    Generate random 2D dataset with multiple clusters

    Parameters:
    -----------
    n_samples : int
        Total number of points
    n_centers : int
        Number of cluster centers
    random_state : int
        Random seed

    Returns:
    --------
    X : array, shape (n_samples, 2)
        Generated data points
    y : array, shape (n_samples,)
        True cluster labels
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_centers,
        cluster_std=0.6,
        random_state=random_state,
    )
    return X, y


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================


def plot_clusters(X, labels, centroids, title):
    """
    Visualize clustering results

    Parameters:
    -----------
    X : array
        Data points
    labels : array
        Cluster labels
    centroids : array
        Cluster centroids
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot data points colored by cluster
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6, edgecolors="k"
    )

    # Plot centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=300,
        edgecolors="black",
        linewidths=2,
        label="Centroids",
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.colorbar(scatter, label="Cluster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# TEST HAND-WRITTEN K-MEANS ON 2D DATA
# ==========================================


def test_handwritten_kmeans():
    """
    Test hand-written K-means on 2D dataset
    """
    print("=" * 60)
    print("EXPERIMENT A: Hand-written K-means on 2D Dataset")
    print("=" * 60)

    # Generate data
    X, y_true = generate_2d_dataset(n_samples=300, n_centers=4, random_state=42)

    # Apply hand-written K-means
    kmeans_hw = KMeansHandwritten(n_clusters=4, random_state=42)

    start_time = time.time()
    kmeans_hw.fit(X)
    hw_time = time.time() - start_time

    # Print results
    print(f"\nConverged in {kmeans_hw.n_iter_} iterations")
    print(f"Execution time: {hw_time:.4f} seconds")
    print(f"Inertia: {kmeans_hw.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette_score(X, kmeans_hw.labels_):.4f}")

    # Visualize
    plot_clusters(
        X,
        kmeans_hw.labels_,
        kmeans_hw.centroids,
        "Hand-written K-means Clustering (K=4)",
    )

    return X, kmeans_hw


# ==========================================
# IMAGE COMPRESSION USING K-MEANS
# ==========================================


def compress_image_kmeans(image_array, n_colors):
    """
    Compress image using K-means clustering

    Parameters:
    -----------
    image_array : array, shape (height, width, 3)
        Original image as RGB array
    n_colors : int
        Number of colors to use in compressed image

    Returns:
    --------
    compressed_image : array
        Compressed image with reduced color palette
    """
    # Get image dimensions
    height, width, channels = image_array.shape

    # Reshape image to 2D array (pixels x RGB channels)
    pixels = image_array.reshape(-1, channels)

    # Apply K-means to cluster colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Replace each pixel with its cluster centroid color
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to original image dimensions
    compressed_image = compressed_pixels.reshape(height, width, channels)

    # Convert to uint8 for display
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    return compressed_image


def test_image_compression():
    """
    Test K-means for image compression
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Image Compression using K-means")
    print("=" * 60)

    # Create a sample colorful image (or load your own)
    # For demonstration, creating a synthetic gradient image
    height, width = 200, 300
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Create RGB channels with patterns
    R = (255 * X_grid).astype(np.uint8)
    G = (255 * Y_grid).astype(np.uint8)
    B = (255 * (1 - X_grid) * Y_grid).astype(np.uint8)

    original_image = np.stack([R, G, B], axis=2)

    # Test different K values
    k_values = [4, 8, 16, 32, 64]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Show original
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    # Compress with different K values
    for idx, k in enumerate(k_values):
        print(f"\nCompressing with K={k} colors...")

        compressed = compress_image_kmeans(original_image, k)

        axes[idx + 1].imshow(compressed)
        axes[idx + 1].set_title(f"K={k} colors", fontweight="bold")
        axes[idx + 1].axis("off")

        # Calculate compression ratio
        original_colors = len(np.unique(original_image.reshape(-1, 3), axis=0))
        compression_ratio = original_colors / k
        print(f"Compression ratio: {compression_ratio:.1f}: 1")

    plt.tight_layout()
    plt.show()


# ==========================================
# COMPARE HAND-WRITTEN VS SKLEARN
# ==========================================


def compare_implementations():
    """
    Compare hand-written K-means with sklearn implementation
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Hand-written vs sklearn Comparison")
    print("=" * 60)

    # Generate data
    X, y_true = generate_2d_dataset(n_samples=300, n_centers=4, random_state=42)

    # Hand-written K-means
    print("\n--- Hand-written K-means ---")
    kmeans_hw = KMeansHandwritten(n_clusters=4, random_state=42)
    start_time = time.time()
    kmeans_hw.fit(X)
    hw_time = time.time() - start_time
    hw_silhouette = silhouette_score(X, kmeans_hw.labels_)

    print(f"Iterations: {kmeans_hw.n_iter_}")
    print(f"Time: {hw_time:.6f} seconds")
    print(f"Inertia: {kmeans_hw.inertia_:.2f}")
    print(f"Silhouette Score: {hw_silhouette:.4f}")

    # sklearn K-means
    print("\n--- sklearn K-means ---")
    kmeans_sk = KMeans(n_clusters=4, random_state=42, n_init=10)
    start_time = time.time()
    kmeans_sk.fit(X)
    sk_time = time.time() - start_time
    sk_silhouette = silhouette_score(X, kmeans_sk.labels_)

    print(f"Iterations: {kmeans_sk.n_iter_}")
    print(f"Time: {sk_time:.6f} seconds")
    print(f"Inertia: {kmeans_sk.inertia_:.2f}")
    print(f"Silhouette Score: {sk_silhouette:.4f}")

    # Comparison
    print("\n--- Comparison ---")
    print(f"Speed improvement: {hw_time/sk_time:.2f}x (sklearn is faster)")
    print(f"Inertia difference: {abs(kmeans_hw.inertia_ - kmeans_sk. inertia_):.2f}")
    print(f"Silhouette difference: {abs(hw_silhouette - sk_silhouette):.4f}")

    # Visualize both
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Hand-written
    scatter1 = axes[0].scatter(
        X[:, 0],
        X[:, 1],
        c=kmeans_hw.labels_,
        cmap="viridis",
        s=50,
        alpha=0.6,
        edgecolors="k",
    )
    axes[0].scatter(
        kmeans_hw.centroids[:, 0],
        kmeans_hw.centroids[:, 1],
        c="red",
        marker="X",
        s=300,
        edgecolors="black",
        linewidths=2,
    )
    axes[0].set_title("Hand-written K-means", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, alpha=0.3)

    # sklearn
    scatter2 = axes[1].scatter(
        X[:, 0],
        X[:, 1],
        c=kmeans_sk.labels_,
        cmap="viridis",
        s=50,
        alpha=0.6,
        edgecolors="k",
    )
    axes[1].scatter(
        kmeans_sk.cluster_centers_[:, 0],
        kmeans_sk.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=300,
        edgecolors="black",
        linewidths=2,
    )
    axes[1].set_title("sklearn K-means", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==========================================
# PARAMETER COMPARISON (DIFFERENT K VALUES)
# ==========================================


def compare_different_k():
    """
    Compare results with different K values using elbow method
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT D: Different K Values Comparison")
    print("=" * 60)

    # Generate data
    X, y_true = generate_2d_dataset(n_samples=300, n_centers=4, random_state=42)

    k_values = range(2, 9)
    inertias = []
    silhouettes = []

    print("\nTesting different K values...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, kmeans.labels_)

        inertias.append(inertia)
        silhouettes.append(silhouette)

        print(f"K={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}")

    # Plot elbow curve and silhouette scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow curve
    axes[0].plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[0].set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
    axes[0].set_title("Elbow Method", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=4, color="r", linestyle="--", label="Optimal K=4")
    axes[0].legend()

    # Silhouette scores
    axes[1].plot(k_values, silhouettes, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].set_title("Silhouette Score vs K", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=4, color="r", linestyle="--", label="Optimal K=4")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Visualize clustering for different K values
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for idx, k in enumerate([2, 3, 4, 5, 6, 7]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        axes[idx].scatter(
            X[:, 0],
            X[:, 1],
            c=kmeans.labels_,
            cmap="viridis",
            s=50,
            alpha=0.6,
            edgecolors="k",
        )
        axes[idx].scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
        )
        axes[idx].set_title(f"K={k}", fontsize=12, fontweight="bold")
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(
        "Clustering Results for Different K Values", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" K-MEANS CLUSTERING EXPERIMENTS")
    print("=" * 60)

    # Run all experiments
    test_handwritten_kmeans()
    test_image_compression()
    compare_implementations()
    compare_different_k()

    print("\n" + "=" * 60)
    print(" ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
