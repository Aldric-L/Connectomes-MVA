import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_correlogram(*correlation_matrices, titles=None, cmap='viridis'):
    """
    Plots multiple correlograms (NumPy arrays) in a row as subplots.

    Args:
        *correlation_matrices: Variable number of NumPy arrays (correlation matrices).
        titles: Optional list of titles for each correlogram. If None, default titles are used.
        cmap: Optional colormap for the heatmaps.
    """

    num_matrices = len(correlation_matrices)

    if num_matrices == 0:
        print("No correlation matrices provided.")
        return

    if titles is None:
        titles = [f"Correlogram {i+1}" for i in range(num_matrices)]
    elif len(titles) != num_matrices:
        print("Number of titles does not match number of matrices.")
        titles = [f"Correlogram {i+1}" for i in range(num_matrices)]

    if num_matrices == 1:
        plt.figure(figsize=(6,6))
    else:
        plt.figure(figsize=(12 * num_matrices, 10))

    for i, corr_matrix in enumerate(correlation_matrices):
        plt.subplot(1, num_matrices, i + 1)
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap=cmap,
            fmt='.2f',
            linewidths=0,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 12},
            square=True,
            linecolor='white'
        )
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()

def plot_connectome(*graphs, titles=None, layout=nx.circular_layout):
    """
    Plots multiple connectome graphs side by side as subplots.

    Args:
        *graphs: Variable number of NetworkX graph objects.
        titles: Optional list of titles for each connectome plot.
        layout: Optional layout function (e.g., nx.circular_layout, nx.spring_layout).
    """

    num_graphs = len(graphs)

    if num_graphs == 0:
        print("No graphs provided.")
        return

    if titles is None:
        titles = [f"Connectome {i+1}" for i in range(num_graphs)]
    elif len(titles) != num_graphs:
        print("Number of titles does not match number of graphs.")
        titles = [f"Connectome {i+1}" for i in range(num_graphs)]

    plt.figure(figsize=(10 * num_graphs, 8))

    for i, G in enumerate(graphs):
        plt.subplot(1, num_graphs, i + 1)

        pos = layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=10, node_color='black', alpha=0.6)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        if edge_weights: #check if there are any edges.
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            cmap = cm.viridis
            nx.draw_networkx_edges(G, pos, width=0.5, edge_color=edge_weights, edge_cmap=cmap, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights))

        nx.draw_networkx_labels(G, pos, font_size=5, font_color='black')

        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()