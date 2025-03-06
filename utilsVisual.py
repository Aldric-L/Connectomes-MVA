import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def plot_correlogram(*correlation_matrices, titles=None, cmap='viridis', organ_names_dict=None):
    """
    Plots multiple correlograms (NumPy arrays) in a row as subplots with a shared color scale and adjusted axis labels.

    Args:
        *correlation_matrices: Variable number of NumPy arrays (correlation matrices).
        titles: Optional list of titles for each correlogram. If None, default titles are used.
        cmap: Optional colormap for the heatmaps.
        organ_names_dict: Optional dictionary mapping indices to organ names. If provided, axis labels are adjusted.
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

    # Find the global min and max for the color scale
    global_min = min(np.nanmin(corr_matrix) for corr_matrix in correlation_matrices)
    global_max = max(np.nanmax(corr_matrix) for corr_matrix in correlation_matrices)

    if num_matrices == 1:
        plt.figure(figsize=(6, 6))
    else:
        plt.figure(figsize=(12 * num_matrices, 10))

    for i, corr_matrix in enumerate(correlation_matrices):
        plt.subplot(1, num_matrices, i + 1)
        ax = sns.heatmap(
            corr_matrix,
            annot=False,
            cmap=cmap,
            fmt='.2f',
            linewidths=0,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 12},
            square=True,
            linecolor='white',
            vmin=global_min,  # Set global min
            vmax=global_max   # Set global max
        )
        plt.title(titles[i])

        if organ_names_dict is not None:
            num_organs = corr_matrix.shape[0]
            if len(organ_names_dict) == num_organs:
                organ_names = [organ_names_dict.get(j, f"Index {j}") for j in range(num_organs)]
                ax.set_xticklabels(organ_names, rotation=90)
                ax.set_yticklabels(organ_names, rotation=0)
            else:
                print(f"Warning: Length of organ_names_dict ({len(organ_names_dict)}) does not match matrix size ({num_organs}).")

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


def plot_density(*vectors, titles=None):
    """
    Plots multiple density plots in a row as subplots.

    Args:
        *vectors: Variable number of lists or NumPy arrays representing the data.
        titles: Optional list of titles for each density plot. If None, default titles are used.
    """

    num_vectors = len(vectors)

    if num_vectors == 0:
        print("No vectors provided.")
        return

    if titles is None:
        titles = [f"Density Plot {i+1}" for i in range(num_vectors)]
    elif len(titles) != num_vectors:
        print("Number of titles does not match number of vectors.")
        titles = [f"Density Plot {i+1}" for i in range(num_vectors)]

    plt.figure(figsize=(10 * num_vectors, 5))

    for i, vector in enumerate(vectors):
        if not isinstance(vector, (list, np.ndarray)):
            raise TypeError(f"Input vector {i+1} must be a list or numpy array.")

        if isinstance(vector, list):
            vector = np.array(vector)

        plt.subplot(1, num_vectors, i + 1)
        sns.kdeplot(vector, fill=True)
        plt.title(titles[i])
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
  
def plot_distribution_vector(vector_of_distributions, title="Distribution Vector"):
    """
    Plots a vector of distributions using overlapping density plots.

    Args:
      vector_of_distributions: A list of lists or numpy arrays, where each inner 
                                list/array represents a distribution.
      title: The title of the plot.
    """
    plt.figure(figsize=(10, 5))
    for distribution in vector_of_distributions:
        sns.kdeplot(distribution, fill=True)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
