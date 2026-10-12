"""Analysis of trained runs: latent-space quality, t-SNE projections and loss curves."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def latent_scores(z, y, k=5):
    """Quality of an embedding space with respect to the ground-truth classes.

    silhouette: silhouette coefficient with cosine distance (compactness and separation of the classes);
    knn: leave-one-out accuracy of a k-nearest-neighbour majority vote with cosine distance
         (class purity of local neighbourhoods). Embeddings are L2-normalized first."""
    y = np.asarray(y)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    silhouette = float(silhouette_score(z, y, metric='cosine'))
    neighbours = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(z).kneighbors(z, return_distance=False)[:, 1:]
    pred = np.array([np.bincount(votes).argmax() for votes in y[neighbours]])
    return dict(silhouette=silhouette, knn=float((pred == y).mean()))


def tsne_2d(z, seed=42):
    """2-D t-SNE of standardized embeddings (perplexity min(30, n - 1))."""
    z = StandardScaler().fit_transform(z)
    return TSNE(n_components=2, perplexity=min(30, len(z) - 1), learning_rate=200, max_iter=1000,
                random_state=seed).fit_transform(z)


def plot_projection(xy, colors, path, label='Class', dpi=300):
    """Scatter plot of a 2-D projection with one discrete color per class or cluster."""
    classes = np.unique(colors)
    cmap = plt.get_cmap('viridis', len(classes))
    norm = matplotlib.colors.BoundaryNorm(boundaries=np.arange(len(classes) + 1) - 0.5, ncolors=len(classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=np.searchsorted(classes, colors), cmap=cmap, norm=norm,
                         s=100, edgecolors='black', linewidths=1.0, alpha=0.9)
    ax.set_xlabel('t-SNE 1', fontweight='bold', fontsize=18)
    ax.set_ylabel('t-SNE 2', fontweight='bold', fontsize=18)
    ax.tick_params(width=2.0, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(16)
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(classes)), fraction=0.05, pad=0.03)
    cbar.ax.set_yticklabels([str(c) for c in classes])
    cbar.set_label(label, fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16, width=1.8, length=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')
    cbar.outline.set_linewidth(2.0)
    plt.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def plot_losses(pre_history, fine_history, path, dpi=300):
    """Loss curves: L_hier during pretraining; L_fine, L_hier and L_proto during fine-tuning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(pre_history['epoch'], pre_history['total_loss'], label=r'$\mathcal{L}_{hier}$')
    axes[0].set_title('Pretraining')
    axes[1].plot(fine_history['epoch'], fine_history['total_loss'], label=r'$\mathcal{L}_{fine}$')
    axes[1].plot(fine_history['epoch'], fine_history['ssl_loss'], label=r'$\mathcal{L}_{hier}$')
    axes[1].plot(fine_history['epoch'], fine_history['prototype_loss'], label=r'$\mathcal{L}_{proto}$')
    axes[1].set_title('Fine-tuning')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(alpha=.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
