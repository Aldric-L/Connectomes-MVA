import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_density(vector, title="Density Plot"):
  """
  Plots the density of a 1D vector.

  Args:
    vector: A list or numpy array representing the data.
    title: The title of the plot.
  """
  if not isinstance(vector, (list, np.ndarray)):
    raise TypeError("Input 'vector' must be a list or numpy array.")

  if isinstance(vector, list):
    vector = np.array(vector)

  plt.figure(figsize=(10, 5))
  sns.kdeplot(vector, fill=True)  # Use seaborn for a smooth density plot
  plt.title(title)
  plt.xlabel("Value")
  plt.ylabel("Density")
  plt.grid(True)
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



def build_organ_dict(organ_SUV_values):
    """
    Constructs an organ dictionary with (mean, variance) for each organ based on SUV values.

    Parameters:
        patients_data (dict): Dictionary where keys are patient IDs and values are pandas DataFrames.
        patients_list (list): List of patient IDs to aggregate data from.

    Returns:
        dict: {organ_name: (mean_SUV, variance_SUV)}
    """

    organ_dict = {}

    for i, arr in enumerate(organ_SUV_values):
        mean = np.mean(arr)
        variance = np.var(arr)
        organ_dict[i] = (mean, variance)

    return organ_dict

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


def print_adjacency_matrices(data_patient):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    im1 = axes[0].imshow(data_patient['bhattacharyya'], cmap='viridis', interpolation='nearest')
    axes[0].set_title('Adjacency Matrix (bhattacharyya)')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    fig.colorbar(im1, ax=axes[0])

    # Plot the second matrix
    im2 = axes[1].imshow(data_patient['kl_divergence'], cmap='viridis', interpolation='nearest')
    axes[1].set_title('Adjacency Matrix (kl_divergence)')
    axes[1].set_xlabel('Node')
    axes[1].set_ylabel('Node')
    fig.colorbar(im2, ax=axes[1])

    # Plot the third matrix
    im3 = axes[2].imshow(data_patient['euclidean-on-log'], cmap='viridis', interpolation='nearest')
    axes[2].set_title('Adjacency Matrix (euclidean-on-log)')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('Node')
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout() #Prevents overlapping titles/labels.
    plt.show()

def compute_pairwise_covariance(organs):
    """
    Compute the pairwise covariance matrix between organs based on the log-transformed 
    parameters of their log-normal distributions.

    Parameters:
        organs (dict): Dictionary where keys are organ names and values are (mean, variance) tuples.

    Returns:
        np.ndarray: Pairwise covariance matrix between organs.
        list: Ordered list of organ names corresponding to matrix indices.
    """
    organ_names = list(organs.keys())
    num_organs = len(organ_names)
    
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