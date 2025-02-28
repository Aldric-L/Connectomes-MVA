import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigvalsh
from scipy.spatial.distance import mahalanobis


def custom_spectral_clustering_on_similarity_matrix(similarity_matrix, num_clusters = 2, viz=True):
    spectral_clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',  # Use the similarity matrix directly
        random_state=42
    )
    labels = spectral_clustering.fit_predict(similarity_matrix)

    # Optional: Visualize the similarity matrix
    if viz is True:
        plt.imshow(similarity_matrix, cmap='viridis')
        plt.colorbar()
        plt.title("Similarity Matrix")
        plt.show()
    return labels

def compute_frobenius_similarity(adj_matrices):
    """
    Compute pairwise similarity between adjacency matrices using the Frobenius norm.
    The similarity is computed as the negative distance (i.e., -Frobenius norm).

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise similarity matrix.
    """
    n = len(adj_matrices)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(adj_matrices[i] - adj_matrices[j], ord='fro')
            similarity_matrix[i, j] = -dist  # Use negative distance as similarity
            similarity_matrix[j, i] = -dist

    return similarity_matrix

def compute_cosine_similarity(adj_matrices):
    """
    Compute pairwise similarity between adjacency matrices using cosine similarity.

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise similarity matrix.
    """
    # Flatten each adjacency matrix into a vector
    flattened_matrices = [adj.flatten() for adj in adj_matrices]
    similarity_matrix = cosine_similarity(flattened_matrices)
    return similarity_matrix

def compute_spectral_distance(adj_matrices):
    """
    Compute pairwise distances between adjacency matrices based on spectral distance.
    The spectral distance is the L2 norm between the sorted eigenvalues of the graph Laplacians.

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise distance matrix.
    """
    n = len(adj_matrices)
    eigenvalue_list = []

    for adj in adj_matrices:
        degree_matrix = np.diag(adj.sum(axis=1))
        laplacian = degree_matrix - adj
        eigenvalues = eigvalsh(laplacian)
        eigenvalue_list.append(np.sort(eigenvalues))

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(eigenvalue_list[i] - eigenvalue_list[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def compute_neighborhood_similarity(graph1, graph2):
    """
    Compute node-to-node similarity between two graphs based on neighborhood similarity.
    Uses cosine similarity of the adjacency matrices as a simple measure of similarity.
    """
    # Flatten the adjacency matrices of both graphs
    graph1_flat = np.array([graph1[i].flatten() for i in range(len(graph1))])
    graph2_flat = np.array([graph2[i].flatten() for i in range(len(graph2))])
    
    # Compute cosine similarity between nodes of the two graphs
    similarity_matrix = cosine_similarity(graph1_flat, graph2_flat)
    return similarity_matrix

def deltaCon(graph1, graph2):
    """
    Compute the deltaCon (dynamic consensus) similarity score between two graphs.
    DeltaCon is based on node similarity and consensus across the two graphs.
    """
    # Step 1: Compute the neighborhood similarity matrix
    neighborhood_similarity = compute_neighborhood_similarity(graph1, graph2)
    
    # Step 2: Create the consensus matrix by comparing graph1 and graph2
    consensus_matrix = (neighborhood_similarity + neighborhood_similarity.T) / 2
    
    # Step 3: Normalize the consensus matrix
    row_norms = np.linalg.norm(consensus_matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    normalized_consensus_matrix = consensus_matrix / row_norms
    
    # Step 4: Calculate deltaCon score as the mean of the normalized consensus matrix
    deltaCon_score = np.mean(normalized_consensus_matrix)
    
    return deltaCon_score

def compute_deltaCon_distance(adj_matrices):
    """
    Compute pairwise distances between adjacency matrices based on the DeltaCon algorithm

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise distance matrix.
    """
    n = len(adj_matrices)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = deltaCon(adj_matrices[i], adj_matrices[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def graph_to_edges(graph):
    """
    Convert an adjacency matrix to a set of edges.
    Each edge is represented as a tuple (i, j) where i and j are node indices.
    """
    edges = set()
    n = graph.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # Only consider upper triangle to avoid duplicate edges
            if graph[i, j] != 0:
                edges.add((i, j))
    return edges

def cut_distance(graph1, graph2):
    """
    Compute the cut distance between two graphs.
    The cut distance is defined as the size of the symmetric difference of edges
    divided by the size of the union of edges.
    """
    # Convert adjacency matrices to edge sets
    edges1 = graph_to_edges(graph1)
    edges2 = graph_to_edges(graph2)
    
    # Compute symmetric difference and union of edge sets
    symmetric_difference = edges1.symmetric_difference(edges2)
    union_edges = edges1.union(edges2)
    # Check if union of edges is empty to avoid division by zero
    if len(union_edges) == 0:
        return 0.0  # No edges in either graph, cut distance is 0
    
    # Cut distance is the ratio of the symmetric difference to the union size
    cut_dist = len(symmetric_difference) / len(union_edges)
    return cut_dist

def compute_cut_distance(adj_matrices):
    """
    Compute pairwise distances between adjacency matrices based on the Cut distance algorithm

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise distance matrix.
    """
    n = len(adj_matrices)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = cut_distance(adj_matrices[i], adj_matrices[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def compute_mahalanobis_similarity(adj_matrices):
    """
    Compute pairwise similarity between adjacency matrices using the Mahalanobis distance.
    The similarity is computed as the negative distance.

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise similarity matrix.
    """
    n = len(adj_matrices)
    similarity_matrix = np.zeros((n, n))

    # Flatten the matrices to use with mahalanobis
    flattened_matrices = [adj.flatten() for adj in adj_matrices]
    
    # Calculate the inverse covariance matrix.
    covariance_matrix = np.cov(flattened_matrices, rowvar=False) #rowvar=False is very important
    inv_covariance = np.linalg.pinv(covariance_matrix)

    for i in range(n):
        for j in range(i, n):
            dist = mahalanobis(flattened_matrices[i], flattened_matrices[j], inv_covariance)
            similarity_matrix[i, j] = -dist  # Use negative distance as similarity
            similarity_matrix[j, i] = -dist

    return similarity_matrix