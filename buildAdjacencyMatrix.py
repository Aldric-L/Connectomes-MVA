import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import cdist

from utilsStats import resample_vectors

def energy_distance(samples_p, samples_q):
    """
    Compute the Energy Distance between two distributions.

    Args:
        samples_p (np.ndarray or list): Samples from distribution P.
        samples_q (np.ndarray or list): Samples from distribution Q.

    Returns:
        float: Energy Distance.
    """
    # Ensure input is a NumPy array
    samples_p = np.asarray(samples_p).reshape(-1, 1)
    samples_q = np.asarray(samples_q).reshape(-1, 1)

    # Compute pairwise distances
    term1 = np.mean(cdist(samples_p, samples_q, metric='euclidean'))
    term2 = np.mean(cdist(samples_p, samples_p, metric='euclidean'))
    term3 = np.mean(cdist(samples_q, samples_q, metric='euclidean'))

    return 2 * term1 - term2 - term3

def kl_divergence_histogram(samples_p, samples_q, num_bins=100):
    """
    Compute the empirical KL divergence between two distributions using histogram-based probability densities.

    Args:
        samples_p (np.ndarray): Sampled values from distribution P.
        samples_q (np.ndarray): Sampled values from distribution Q.
        num_bins (int): Number of histogram bins.

    Returns:
        float: Symmetric KL divergence (D_KL(P || Q) + D_KL(Q || P)).
    """
    min_val, max_val = min(np.min(samples_p), np.min(samples_q)), max(np.max(samples_p), np.max(samples_q))
    bins = np.linspace(min_val, max_val, num_bins)

    p_hist, _ = np.histogram(samples_p, bins=bins, density=True)
    q_hist, _ = np.histogram(samples_q, bins=bins, density=True)

    p_hist += 1e-10  # Avoid division by zero
    q_hist += 1e-10

    kl_pq = np.sum(p_hist * np.log(p_hist / q_hist)) * (max_val - min_val) / num_bins
    kl_qp = np.sum(q_hist * np.log(q_hist / p_hist)) * (max_val - min_val) / num_bins

    return kl_pq + kl_qp  # Symmetric KL divergence


def wasserstein_approximation(samples_p, samples_q, method="quantile", num_bins=100, num_quantiles=100):
    """
    Compute an approximated Wasserstein distance between two large distributions.
    
    Args:
        samples_p (np.ndarray): Sampled values from distribution P.
        samples_q (np.ndarray): Sampled values from distribution Q.
        method (str): Approximation method ("histogram" or "quantile").
        num_bins (int): Number of bins for histogram approximation.
        num_quantiles (int): Number of quantiles for quantile approximation.
    
    Returns:
        float: Approximated Wasserstein distance.
    """
    if method == "histogram":
        # Define a common bin range
        min_val = min(np.min(samples_p), np.min(samples_q))
        max_val = max(np.max(samples_p), np.max(samples_q))
        bins = np.linspace(min_val, max_val, num_bins)

        # Compute histogram bin centers and densities
        p_hist, _ = np.histogram(samples_p, bins=bins, density=True)
        q_hist, _ = np.histogram(samples_q, bins=bins, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Compute Wasserstein distance using histograms
        dist = wasserstein_distance(bin_centers, bin_centers, p_hist, q_hist)

    elif method == "quantile":
        # Generate quantiles for both distributions
        quantiles = np.linspace(0, 1, num_quantiles)
        p_quantiles = np.quantile(samples_p, quantiles)
        q_quantiles = np.quantile(samples_q, quantiles)

        # Wasserstein distance between quantile approximations
        dist = np.mean(np.abs(p_quantiles - q_quantiles))

    else:
        raise ValueError("Unsupported approximation method. Choose 'histogram' or 'quantile'.")

    return dist


def bhattacharyya_quantile(samples_p, samples_q, num_quantiles=100, epsilon=1e-10):
    """
    Compute the Bhattacharyya distance between two distributions using quantile approximation.

    Args:
        samples_p (np.ndarray): Sampled values from distribution P.
        samples_q (np.ndarray): Sampled values from distribution Q.
        num_quantiles (int): Number of quantiles to use for approximation.
        epsilon (float): small value to prevent division by zero, and log(0).

    Returns:
        float: Bhattacharyya distance.
    """
    # Compute quantiles for both distributions
    quantiles = np.linspace(0, 1, num_quantiles)
    p_quantiles = np.quantile(samples_p, quantiles)
    q_quantiles = np.quantile(samples_q, quantiles)

    # Compute probability densities from quantiles (finite difference approximation)
    p_densities = np.gradient(p_quantiles)
    q_densities = np.gradient(q_quantiles)

    # Normalize densities to sum to 1, with handling for zero sums
    p_sum = np.sum(p_densities)
    q_sum = np.sum(q_densities)

    if p_sum > epsilon:
        p_densities /= p_sum
    else:
        p_densities = np.ones_like(p_densities) / len(p_densities) #If all zero, make uniform distribution.

    if q_sum > epsilon:
        q_densities /= q_sum
    else:
        q_densities = np.ones_like(q_densities) / len(q_densities) #If all zero, make uniform distribution.

    # Compute Bhattacharyya coefficient (sum of square-rooted probability densities)
    bc = np.sum(np.sqrt(p_densities * q_densities))

    # Convert to Bhattacharyya distance
    return -np.log(bc + epsilon)  # Avoid log(0)

def compute_pairwise_covariance(organs, mode="data"):
    """
    Compute the pairwise covariance matrix between organs

    Parameters:
        organs (dict): Dictionary where keys are organ names and values are (mean, variance) tuples.
        mode (string): "data" or "parameters" : if data we compute the actual covariance, if parameters we compute the cov based on the log-transformed 
    parameters of their log-normal distributions.

    Returns:
        np.ndarray: Pairwise covariance matrix between organs.
        list: Ordered list of organ names corresponding to matrix indices.
    """
    if mode == "data":
        if type(organs) != np.ndarray:
            organ_names = list(organs.keys())
            # Stack organ data into a matrix (rows = organs, columns = samples)
            data_matrix = np.vstack([organs[name] for name in organ_names])
        else:
            organ_names = []
            data_matrix = resample_vectors(organs)
            
        # Compute the covariance matrix (num_organs x num_organs)
        covariance_matrix = np.cov(data_matrix)
    else: 
        organ_names = list(organs.keys())
        # Convert organ dictionary to a NumPy array for easier computation
        organ_vectors = np.array([organs[name] for name in organ_names])
        
        # Convert (mean, variance) to underlying normal parameters (log_mu, log_sigma_sq)
        log_mu = np.log(organ_vectors[:, 0]**2 / np.sqrt(organ_vectors[:, 1] + organ_vectors[:, 0]**2))
        log_sigma_sq = np.log(organ_vectors[:, 1] / organ_vectors[:, 0]**2 + 1)
        
        # Stack transformed parameters into a (num_organs, 2) matrix
        log_params = np.vstack((log_mu, log_sigma_sq)).T  

        # Compute the covariance matrix between organs (num_organs x num_organs)
        covariance_matrix = np.cov(log_params, rowvar=False)  

    return covariance_matrix, organ_names


def build_adjacency_matrix(organs, metric="kl_divergence", covariance_matrix=None, cov_sensitivity=1):
    """
    Build the adjacency matrix for a graph of organs based on empirical distributions.

    Parameters:
        organs (dict): Dictionary where keys are organ names and values are sample vectors (np.ndarray).
        metric (str): Distance metric to use ("kl_divergence", "wasserstein", "jensen_shannon", "euclidean", "bhattacharyya").
        covariance_matrix (np.ndarray, optional): Covariance matrix between organ variables.
        cov_sensitivity (float, optional): A coefficient to ponder the covariance correction

    Returns:
        np.ndarray: Adjusted adjacency matrix (symmetric).
        list: Ordered list of organ names corresponding to matrix indices.
    """
    organ_names = list(organs.keys())
    num_organs = len(organ_names)
    distance_matrix = np.zeros((num_organs, num_organs))

    # Compute distance matrix
    if metric == "kl_divergence":
        for i in range(num_organs):
            for j in range(i + 1, num_organs):
                dist = kl_divergence_histogram(organs[organ_names[i]], organs[organ_names[j]])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
    elif metric == "energy":
        for i in range(num_organs):
            for j in range(i + 1, num_organs):
                dist = energy_distance(organs[organ_names[i]], organs[organ_names[j]])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
    elif metric == "wasserstein":
        for i in range(num_organs):
            for j in range(i + 1, num_organs):
                dist = wasserstein_approximation(organs[organ_names[i]], organs[organ_names[j]])
                distance_matrix[i, j] = distance_matrix[j, i] = dist

    elif metric == "jensen_shannon":
        def js_divergence(i, j):
            samples_p, samples_q = organs[organ_names[i]], organs[organ_names[j]]
            min_val, max_val = min(np.min(samples_p), np.min(samples_q)), max(np.max(samples_p), np.max(samples_q))
            bins = np.linspace(min_val, max_val, 100)

            p_hist, _ = np.histogram(samples_p, bins=bins, density=True)
            q_hist, _ = np.histogram(samples_q, bins=bins, density=True)

            return jensenshannon(p_hist, q_hist)

        for i in range(num_organs):
            for j in range(i + 1, num_organs):
                dist = js_divergence(i, j)
                distance_matrix[i, j] = distance_matrix[j, i] = dist

    elif metric == "euclidean" or metric == "euclidean-on-log":
        organ_vectors = resample_vectors(organs)
        for i in range(len(organ_vectors)):
            for j in range(i + 1, len(organ_vectors)):
                dist = np.linalg.norm(organ_vectors[i] - organ_vectors[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist

    elif metric == "bhattacharyya":
        for i in range(num_organs):
            for j in range(i + 1, num_organs):
                dist = bhattacharyya_quantile(organs[organ_names[i]], organs[organ_names[j]])
                distance_matrix[i, j] = distance_matrix[j, i] = dist

    else:
        raise ValueError("Unsupported metric. Choose 'kl_divergence', 'energy', 'wasserstein', 'jensen_shannon', 'euclidean', or 'bhattacharyya'.")

    # Adjust distance matrix with mutual information from covariance_matrix
    distance_matrix_corrected = distance_matrix
    if covariance_matrix is not None:
        sigma = np.sqrt(np.diag(covariance_matrix))
        corr_matrix = covariance_matrix / np.outer(sigma, sigma)
        np.fill_diagonal(corr_matrix, 1.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            mi_matrix = -0.5 * np.log(1 - np.square(corr_matrix))
        mi_matrix = np.nan_to_num(mi_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(mi_matrix, 0)

        mi_max = np.max(mi_matrix)
        if mi_max > 0:
            mi_matrix /= mi_max
        distance_matrix_corrected *= (1 - cov_sensitivity * mi_matrix)

    # Build adjacency matrix
    adjacency_matrix = np.zeros((num_organs, num_organs))
    adjacency_matrix = np.exp(-distance_matrix_corrected)
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix, distance_matrix, organ_names
