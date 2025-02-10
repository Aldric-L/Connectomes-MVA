import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_correlogram(correlation_matrix, title="Correlogram"):
    """ 
    Function that plots a correlogram from a pandas correlation matrix
    """
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap='viridis',  # You can change this to another colormap like 'viridis' or 'RdBu_r'
        fmt='.2f',
        linewidths=0,  # Increase linewidth for better separation between cells
        cbar_kws={"shrink": 0.8},  # Shrink colorbar for better aesthetics
        annot_kws={"size": 12},  # Set font size of annotations
        square=True,  # Make the heatmap square-shaped
        linecolor='white'  # Add white lines between cells
    )
    plt.title(title)
    plt.show()

def plot_connectome(G):
    plt.figure(figsize=(10, 8))

    # Get positions of nodes in a circular layout
    pos = nx.circular_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='black', alpha=0.6)

    # Get edge weights for color mapping
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Normalize edge weights for color mapping
    norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))  # Normalize edge weights
    cmap = cm.viridis  # Use a colormap (can change to other colormaps like 'coolwarm')

    # Draw edges with colors based on weight
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color=edge_weights, edge_cmap=cmap, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights))

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black')

    plt.title('Connectome Graph on Circular Layout with Edge Intensity')
    plt.axis('off') 
    plt.show()