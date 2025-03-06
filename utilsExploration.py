import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 

from scipy.stats import ttest_ind
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy import stats

import random

from utilsStats import resample_vectors

# def build_organ_dict(organ_SUV_values):
#     """
#     Constructs an organ dictionary with (mean, variance) for each organ based on SUV values.

#     Parameters:
#         patients_data (dict): Dictionary where keys are patient IDs and values are pandas DataFrames.
#         patients_list (list): List of patient IDs to aggregate data from.

#     Returns:
#         dict: {organ_name: (mean_SUV, variance_SUV)}
#     """

#     organ_dict = {}

#     for i, arr in enumerate(organ_SUV_values):
#         mean = np.mean(arr)
#         variance = np.var(arr)
#         organ_dict[i] = (mean, variance)

#     return organ_dict

def normalize_matrix_no_diagonal(matrix):
    """
    Normalizes a matrix (min-max scaling) while ignoring the diagonal.

    Args:
        matrix: A NumPy array (matrix) to normalize.

    Returns:
        A normalized NumPy array.
    """
    matrix = matrix.astype(float) #Ensure it's a float matrix.
    diagonal = np.diag(matrix).copy() #Store the diagonal values.
    np.fill_diagonal(matrix, np.nan) #Temporarily set diagonal to NaN.

    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)

    if max_val == min_val:
        # Handle the case where all off-diagonal elements are the same
        normalized_matrix = np.zeros_like(matrix)
    else:
        normalized_matrix = (matrix - min_val) / (max_val - min_val)

    np.fill_diagonal(normalized_matrix, diagonal) #Restore the original diagonal.
    return normalized_matrix


# def print_adjacency_matrices(data_patient):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
#     im1 = axes[0].imshow(data_patient['bhattacharyya'], cmap='viridis', interpolation='nearest')
#     axes[0].set_title('Adjacency Matrix (bhattacharyya)')
#     axes[0].set_xlabel('Node')
#     axes[0].set_ylabel('Node')
#     fig.colorbar(im1, ax=axes[0])

#     # Plot the second matrix
#     im2 = axes[1].imshow(data_patient['kl_divergence'], cmap='viridis', interpolation='nearest')
#     axes[1].set_title('Adjacency Matrix (kl_divergence)')
#     axes[1].set_xlabel('Node')
#     axes[1].set_ylabel('Node')
#     fig.colorbar(im2, ax=axes[1])

#     # Plot the third matrix
#     im3 = axes[2].imshow(data_patient['euclidean-on-log'], cmap='viridis', interpolation='nearest')
#     axes[2].set_title('Adjacency Matrix (euclidean-on-log)')
#     axes[2].set_xlabel('Node')
#     axes[2].set_ylabel('Node')
#     fig.colorbar(im3, ax=axes[2])

#     plt.tight_layout() #Prevents overlapping titles/labels.
#     plt.show()


def upper_triangle_to_vector(matrix):
    """
    Converts the upper triangle of a matrix (excluding the diagonal) to a vector.

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        A NumPy array containing the upper triangle elements as a vector.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input 'matrix' must be a NumPy array.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Input 'matrix' must be a square matrix.")

    upper_triangle_elements = []
    for i in range(rows):
        for j in range(i + 1, cols):
            upper_triangle_elements.append(matrix[i, j])

    return np.array(upper_triangle_elements)

def pairwise_covariance_matrices(matrix_vector):
    """
    Computes the pairwise covariance between a vector of same-size matrices.

    Args:
        matrix_vector: A list or NumPy array of NumPy matrices, all with the same shape.

    Returns:
        A NumPy array representing the pairwise covariance matrix.
        The shape of the returned matrix will be (n, n), where n is the number of matrices.
    """

    if not isinstance(matrix_vector, (list, np.ndarray)):
        raise TypeError("Input 'matrix_vector' must be a list or NumPy array.")

    if not matrix_vector:
        return np.array([])  # Return empty array if input is empty

    num_matrices = len(matrix_vector)
    matrix_shape = matrix_vector[0].shape

    # Check if all elements are matrices and have the same shape
    for matrix in matrix_vector:
        if not isinstance(matrix, np.ndarray) or matrix.shape != matrix_shape:
            raise ValueError("All elements in 'matrix_vector' must be NumPy matrices with the same shape.")

    # Flatten each matrix into a 1D vector
    flattened_matrices = [matrix.flatten() for matrix in matrix_vector]

    # Compute the covariance matrix of the flattened vectors
    covariance_matrix = np.cov(flattened_matrices)

    return covariance_matrix

def compute_mahalanobis_similarity_optimized(adj_matrices):
    """
    Compute pairwise similarity between adjacency matrices using Mahalanobis distance,
    considering only the upper triangle (excluding the diagonal).

    Parameters:
        adj_matrices (list of np.ndarray): List of adjacency matrices.

    Returns:
        np.ndarray: Pairwise similarity matrix.
    """
    n = len(adj_matrices)
    similarity_matrix = np.zeros((n, n))

    if n == 0:
        return similarity_matrix

    # Extract upper triangles and flatten
    flattened_matrices = []
    for adj in adj_matrices:
        rows, cols = adj.shape
        upper_indices = np.triu_indices(rows, k=1)
        flattened_matrices.append(adj[upper_indices])
    flattened_matrices = np.array(flattened_matrices)

    # Calculate inverse covariance matrix (once)
    covariance_matrix = np.cov(flattened_matrices, rowvar=False)
    inv_covariance = np.linalg.pinv(covariance_matrix)

    # Calculate all pairwise Mahalanobis distances
    for i in range(n):
        for j in range(i, n):
            dist = mahalanobis(flattened_matrices[i], flattened_matrices[j], inv_covariance)
            similarity_matrix[i, j] = -dist
            similarity_matrix[j, i] = -dist

    return similarity_matrix


from joblib import Parallel, delayed

def compute_mahalanobis_similarity_optimized(adj_matrices):
    n = len(adj_matrices)
    similarity_matrix = np.zeros((n, n))

    if n == 0:
        return similarity_matrix

    print("Flattening")
    flattened_matrices = []
    for adj in adj_matrices:
        rows, cols = adj.shape
        upper_indices = np.triu_indices(rows, k=1)
        flattened_matrices.append(adj[upper_indices])
    flattened_matrices = np.array(flattened_matrices)

    covariance_matrix = np.cov(flattened_matrices, rowvar=False)
    inv_covariance = np.linalg.pinv(covariance_matrix)

    def calculate_distance(i, j):
        print("Distance ", i, " ", j)
        dist = mahalanobis(flattened_matrices[i], flattened_matrices[j], inv_covariance)
        return i, j, dist

    results = Parallel(n_jobs=-1)(delayed(calculate_distance)(i, j)
                                   for i in range(n) for j in range(i, n))

    for i, j, dist in results:
        similarity_matrix[i, j] = -dist
        similarity_matrix[j, i] = -dist

    return similarity_matrix

def remove_last_n_elements_and_slice(dictionary, n):
    """
    Removes the last n elements from a dictionary (maintaining order) and returns both the new dictionary and the removed slice.

    Args:
        dictionary: The input dictionary.
        n: The number of elements to remove from the end.

    Returns:
        A tuple containing:
            - A new dictionary with the last n elements removed.
            - A dictionary containing the removed slice.
    """

    def shuffle_dictionary(dic):
        """
        Shuffles a dictionary and returns a new shuffled dictionary.

        Args:
            dictionary: The input dictionary.

        Returns:
            A new dictionary with the same key-value pairs but in a shuffled order.
        """
        if not isinstance(dic, dict):
            raise TypeError("Input must be a dictionary.")

        items = list(dic.items())  # Get key-value pairs as a list of tuples
        random.shuffle(items)  # Shuffle the list of tuples
        return dict(items)  # Create a new dictionary from the shuffled list

    if not isinstance(dictionary, dict):
        raise TypeError("Input must be a dictionary.")

    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer.")

    if not dictionary:  # Handle empty dictionary
        return {}, {}

    if n >= len(dictionary):
        return {}, dictionary.copy()
    
    dictionary = shuffle_dictionary(dictionary)
    items = list(dictionary.items())
    new_items = items[:-n]
    removed_items = items[-n:]

    return dict(new_items), dict(removed_items)

def compute_column_sum_probabilities(test_matrix, set_of_matrices, epsilon=0.01):
    """
    For each matrix in set_of_matrices (each n x n), compute the sum of each column.
    Learn the distribution of these sums using kernel density estimation.
    Then, for each column in the test_matrix, compute the sum and approximate the 
    probability that a sample from the learned distribution falls within [x-epsilon, x+epsilon].
    
    Parameters:
      test_matrix (np.ndarray): An n x n matrix.
      set_of_matrices (list of np.ndarray): A list of n x n matrices.
      epsilon (float): Half the width of the interval around the observed column sum.
    
    Returns:
      probabilities (np.ndarray): An array of probabilities (one per column of test_matrix).
    """
    # Gather all column sums from the dataset
    dataset_column_sums = []
    for mat in set_of_matrices:
        # Compute column sums for each matrix (resulting in an array of shape (n,))
        dataset_column_sums.extend(np.sum(mat, axis=0))
    dataset_column_sums = np.array(dataset_column_sums)
    
    # Learn the distribution using kernel density estimation (non-parametric)
    kde = gaussian_kde(dataset_column_sums)
    
    # Compute column sums for the test matrix
    test_column_sums = np.sum(test_matrix, axis=0)
    
    # For each test column sum, compute the probability mass in [x - epsilon, x + epsilon]
    probabilities = np.array([
        max(0, kde.integrate_box_1d(x - epsilon, x + epsilon))
        for x in test_column_sums
    ])
    
    return probabilities

def test_normality_of_matrices(matrices):
    """
    Returns a matrix of p-values from Shapiro-Wilk tests for each coefficient.

    Args:
        matrices: A list or NumPy array of squared NumPy matrices.

    Returns:
        A NumPy matrix of means, with the same shape as the input matrices.
        A NumPy matrix of variances, with the same shape as the input matrices.
        A NumPy matrix of p-values, with the same shape as the input matrices.
        A number of p-values that are under 0.05
    """

    if not isinstance(matrices, (list, np.ndarray)):
        raise TypeError("Input must be a list or NumPy array of matrices.")

    if not matrices:
        return np.array([])

    matrix_shape = matrices[0].shape
    if len(matrix_shape) != 2 or matrix_shape[0] != matrix_shape[1]:
        raise ValueError("Matrices must be square.")

    num_matrices = len(matrices)
    p_values_matrix = np.zeros(matrix_shape)
    n_pvalues = 0

    for row in range(matrix_shape[0]):
        for col in range(matrix_shape[1]):
            coefficient_values = [matrix[row, col] for matrix in matrices]
            _, p_value = stats.shapiro(coefficient_values)  # Get p-value only
            p_values_matrix[row, col] = p_value
            n_pvalues += (p_value <= 0.05)

    return np.mean(matrices, axis=0), np.var(matrices, axis=0), p_values_matrix, n_pvalues

def compute_ncs(correlation_matrix):
    """
    Compute the Nodal Connectivity Strength (NCS) for each node in a given correlation matrix.

    NCS is defined as the sum of the absolute values of correlation coefficients
    for each node (excluding self-correlations).

    Args:
        correlation_matrix (pd.DataFrame or np.array): A symmetric matrix of Pearson correlation coefficients.

    Returns:
        pd.Series or np.array: A series containing the NCS for each node, indexed by the node names.
    """
    # Ensure input is a DataFrame
    if not isinstance(correlation_matrix, pd.DataFrame):
        return np.sum(np.abs(correlation_matrix), axis=1) - np.diag(correlation_matrix)
    else:
        # Compute NCS by summing the absolute values of correlations (excluding self-correlations)
        ncs_values = correlation_matrix.abs().sum(axis=1) - np.diag(correlation_matrix)

        return pd.Series(ncs_values, index=correlation_matrix.index)
    

def set_diagonal_to_zero(matrix):
    """
    Sets the diagonal elements of a NumPy matrix to zero.

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        A NumPy array with the diagonal elements set to zero.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix must be square.")

    modified_matrix = matrix.copy() #Create a copy to avoid modifying the original.
    np.fill_diagonal(modified_matrix, 0)

    return modified_matrix