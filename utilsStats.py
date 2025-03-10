import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
import networkx as nx

def pearson_correlation(x, y):
    """ Pearson correlation formula: r = cov(X, Y) / (std(X) * std(Y)) """
    covariance = ((x - x.mean()) * (y - y.mean())).mean()
    correlation = covariance / (x.std() * y.std())
    return correlation

def pearson_p_value(r, n):
    """Calculate the p-value for Pearson correlation."""
    t_stat = r * ((n - 2) ** 0.5) / ((1 - r ** 2) ** 0.5)
    p_value = 2 * stats.t.sf(abs(t_stat), n - 2)
    return p_value

def build_correlation_matrix(patients_data: dict, patients_keys, names, graph_threshold=0):
    """ 
    Function that builds a correlation matrix for a dictionnary of pandas datasets.
    It assumes that the dictionnary maps patients' id to patients' dataframes. Each dataframe is assumed to have the mean_SUV row.

    graph_threshold: float : if not -1 it constructs a sparse graph by removing connections that are too week in abs, if -1, it does not build the graph
    """
    mean_SUV_by_names = {name: [patients_data[patient].loc[:, patients_data[patient].iloc[0] == name].loc["mean_SUV"].to_numpy()[0] for patient in patients_keys] for name in names}
    mean_SUV_by_names_df = pd.DataFrame(mean_SUV_by_names)
    #correlation_matrix = mean_SUV_by_names_df.corr()

    correlation_matrix = pd.DataFrame(index=mean_SUV_by_names_df.columns, columns=mean_SUV_by_names_df.columns)
    pearson_pval_matrix = pd.DataFrame(index=mean_SUV_by_names_df.columns, columns=mean_SUV_by_names_df.columns)
    spearman_corr_matrix = pd.DataFrame(index=mean_SUV_by_names_df.columns, columns=mean_SUV_by_names_df.columns)
    spearman_pval_matrix = pd.DataFrame(index=mean_SUV_by_names_df.columns, columns=mean_SUV_by_names_df.columns)

    # Compute the matrices for Pearson and Spearman correlations and p-values
    for col1 in mean_SUV_by_names_df.columns:
        for col2 in mean_SUV_by_names_df.columns:
            # Pearson correlation and p-value
            pearson_corr = pearson_correlation(mean_SUV_by_names_df[col1], mean_SUV_by_names_df[col2])
            pearson_pval = pearson_p_value(pearson_corr, len(mean_SUV_by_names_df))
            
            # Spearman correlation and p-value
            spearman_corr, spearman_pval_value = stats.spearmanr(mean_SUV_by_names_df[col1], mean_SUV_by_names_df[col2])
            
            correlation_matrix.loc[col1, col2] = pearson_corr
            pearson_pval_matrix.loc[col1, col2] = pearson_pval
            spearman_corr_matrix.loc[col1, col2] = spearman_corr
            spearman_pval_matrix.loc[col1, col2] = spearman_pval_value

    # Convert matrices to numeric type
    correlation_matrix = correlation_matrix.apply(pd.to_numeric)
    pearson_pval_matrix = pearson_pval_matrix.apply(pd.to_numeric)
    spearman_corr_matrix = spearman_corr_matrix.apply(pd.to_numeric)
    spearman_pval_matrix = spearman_pval_matrix.apply(pd.to_numeric)

    correlation_matrix_np = correlation_matrix.to_numpy()

    if graph_threshold != -1:
        G = build_graph_from_correlation_df(correlation_matrix, names, graph_threshold)
    else:
        G = None

    return correlation_matrix, correlation_matrix_np, pearson_pval_matrix, spearman_corr_matrix, spearman_pval_matrix, G

def build_graph_from_correlation_df(correlation_matrix, names, graph_threshold=0):
    """ 
    Function that builds a correlation graph from a correlation matrix
    """
    G = nx.Graph()
    G.add_nodes_from(names)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):  # Only upper triangle (undirected graph)
            if type(correlation_matrix) == np.ndarray:
                if abs(correlation_matrix[i, j]) >= graph_threshold:  # Threshold to determine if an edge exists
                    G.add_edge(names[i], names[j], weight=correlation_matrix[i, j])
            else:
                if abs(correlation_matrix.loc[names[i], names[j]]) >= graph_threshold:  # Threshold to determine if an edge exists
                    G.add_edge(names[i], names[j], weight=correlation_matrix.loc[names[i], names[j]])
    return G

def fisher_transform(r):
    """
    Apply Fisher's r-to-z transformation.
    Clamps r to avoid division errors.
    """
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def z_test_for_correlation_diff(r_N, r_N_plus_1, N):
    """
    Compute a z-test for the difference between two Pearson correlation coefficients
    using Fisher's z-transformation. The two correlations are computed from nearly identical 
    datasets (N and N+1 samples). Note that this test assumes independence between the two 
    estimates, which is not strictly true here, so the p-values are only approximate.
    
    Args:
        r_N (float): Pearson correlation from the N-sample matrix.
        r_N_plus_1 (float): Pearson correlation from the N+1-sample matrix.
        N (int): Number of samples in the original (N-sample) matrix.
    
    Returns:
        z (float): z-value for the difference.
        p_value (float): Two-tailed p-value corresponding to the z-value.
    """
    # Transform correlations to z-scale
    z_N = fisher_transform(r_N)
    z_N_plus_1 = fisher_transform(r_N_plus_1)
    
    # Standard error estimates for independent samples (for N and N+1 subjects)
    se_N = 1 / np.sqrt(N - 3)
    se_Np1 = 1 / np.sqrt((N + 1) - 3)
    
    # Under the (incorrect but common) assumption of independence, the variance of the difference is the sum:
    se_diff = np.sqrt(se_N**2 + se_Np1**2)
    
    # Compute the z-statistic for the difference
    z = (z_N - z_N_plus_1) / se_diff
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p_value

def t_test_for_correlation_diff(r_N, r_N_plus_1, N):
    """
    An alternative (approximate) test using t-scores.
    Here we simply convert the z-score into a t-statistic using an effective degree 
    of freedom (here approximated as N-3). This is provided as an option.
    
    Args:
        r_N (float): Pearson correlation from the N-sample matrix.
        r_N_plus_1 (float): Pearson correlation from the N+1-sample matrix.
        N (int): Number of samples in the original matrix.
    
    Returns:
        t (float): t-statistic for the difference.
        p_value (float): two-tailed p-value corresponding to the t-statistic.
    """
    z, _ = z_test_for_correlation_diff(r_N, r_N_plus_1, N)
    df = N - 3  # approximate degrees of freedom
    t_stat = z * np.sqrt(df)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    return t_stat, p_value

def build_significance_matrix_for_diff(pearson_corr_matrix_N, pearson_corr_matrix_N_plus_1, N, Fisher=True):
    """
    Compute a matrix of p-values (and corresponding test statistics) for the differences 
    in Pearson correlation coefficients between two correlation matrices: one computed 
    from N samples and the other from N+1 samples.
    
    The procedure is:
        - For each pair of variables, retrieve the correlation from both matrices.
        - Transform the correlations using Fisher's z-transformation.
        - Compute the difference and the corresponding standard error (assuming independence).
        - Compute the z-value (or t-value) and then the two-tailed p-value.
    
    Args:
        pearson_corr_matrix_N (pd.DataFrame): Square matrix of Pearson correlations for N samples.
        pearson_corr_matrix_N_plus_1 (pd.DataFrame): Square matrix of Pearson correlations for N+1 samples.
        N (int): Number of samples in the original matrix.
        Fisher (bool): If True, use the Fisher z-test (default). Otherwise, use the t-test.
    
    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Two DataFrames of the same shape as the input matrices.
            - The first contains the test statistics (z-values if Fisher=True, t-values otherwise).
            - The second contains the corresponding two-tailed p-values.
    """
    # Initialize DataFrames to store test statistics and p-values
    z_matrix = pd.DataFrame(index=pearson_corr_matrix_N.index, columns=pearson_corr_matrix_N.columns)
    p_values_matrix = pd.DataFrame(index=pearson_corr_matrix_N.index, columns=pearson_corr_matrix_N.columns)
    
    # Loop over every pair of variables (edges)
    for var1 in pearson_corr_matrix_N.index:
        for var2 in pearson_corr_matrix_N.columns:
            r_N = pearson_corr_matrix_N.loc[var1, var2]
            r_N_plus_1 = pearson_corr_matrix_N_plus_1.loc[var1, var2]
            
            if Fisher:
                stat_val, p_val = z_test_for_correlation_diff(r_N, r_N_plus_1, N)
            else:
                stat_val, p_val = t_test_for_correlation_diff(r_N, r_N_plus_1, N)
            
            z_matrix.loc[var1, var2] = stat_val
            p_values_matrix.loc[var1, var2] = p_val

    # Ensure that all entries are numeric
    z_matrix = z_matrix.astype(float)
    p_values_matrix = p_values_matrix.astype(float)
    
    return z_matrix, p_values_matrix


# def z_test_for_correlation_diff(r_N, r_N_plus_1, N):
#     """
#     Compute the z-test for the difference between two correlation coefficients
#     from matrices with N and N+1 samples.
    
#     Args:
#     r_N (float): Pearson correlation coefficient from the N-sample matrix.
#     r_N_plus_1 (float): Pearson correlation coefficient from the N+1-sample matrix.
#     N (int): Sample size for the N-sample matrix.
    
#     Returns:
#     z (float): z-value for the difference.
#     p_value (float): p-value corresponding to the z-value.
#     """
#     # Fisher z-transformation for both correlation coefficients
#     z_N = fisher_transform(r_N)
#     z_N_plus_1 = fisher_transform(r_N_plus_1)
    
#     # Compute the standard error of the difference
#     SE_diff = np.sqrt(1 / (N - 3) + 1 / (N + 1 - 3))
    
#     # Compute the z-value for the difference
#     z = (z_N - z_N_plus_1) / SE_diff
    
#     # Compute the two-tailed p-value
#     p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
#     return z, p_value

# def build_significance_matrix_for_diff(pearson_corr_matrix_N, pearson_corr_matrix_N_plus_1, N, Fisher=True):
#     """
#     Compute a matrix of p-values for the difference in Pearson correlation coefficients 
#     between two correlation matrices, one for N samples and the other for N+1 samples.
    
#     The function compares the correlation coefficients from two matrices where the first 
#     matrix corresponds to a sample size of N, and the second corresponds to N+1 samples. 
#     It computes the significance of the difference between each pair of correlation coefficients 
#     using the following procedure:
#         - Fisher Transformation: Both correlation coefficients (from N and N+1) are transformed using Fisher’s z-transformation. (The Fisher z-transformation converts correlation coefficients into a variable that follows approximately a normal distribution, even when the correlation is extreme)
#         - Difference of z-scores and std. error: it computes the difference between the z-scores for each pair of correlations.
#         - Compute the z-value: it uses the z-scores and the standard error to compute the z-value for each pair.
#         - Compute p-values: Finally, it computes the p-value corresponding to the z-value.

#     Args:
#         N: the number of patients in the original matrix
#         pearson_corr_matrix_N (pd.DataFrame): A square matrix of Pearson correlation coefficients 
#                                                for N samples. The matrix should be symmetric with 
#                                                columns and rows corresponding to variables.
#         pearson_corr_matrix_N_plus_1 (pd.DataFrame): A square matrix of Pearson correlation coefficients 
#                                                      for N+1 samples. It should have the same shape as 
#                                                      pearson_corr_matrix_N, with columns and rows corresponding 
#                                                      to the same variables.
#         Fisher: boolean (whether to compute z scores or t scores)
    
#     Returns:
#         pd.DataFrame: A matrix of p-values corresponding to the significance of the differences 
#                       between the correlation coefficients of pearson_corr_matrix_N and pearson_corr_matrix_N_plus_1.
#                       The matrix has the same shape as the input correlation matrices, with rows and columns representing 
#                       the variables being compared.
#     """
#     # Initialize a DataFrame for z-values and p-values
#     z_matrix = pd.DataFrame(index=pearson_corr_matrix_N.columns, columns=pearson_corr_matrix_N.columns)
#     p_values_matrix = pd.DataFrame(index=pearson_corr_matrix_N.columns, columns=pearson_corr_matrix_N.columns)

#     # Compute the p-value for the difference for each pair of coefficients
#     for col1 in pearson_corr_matrix_N.columns:
#         for col2 in pearson_corr_matrix_N.columns:
#             # Get the correlation coefficients from both matrices (for N and N+1 samples)
#             r_N = pearson_corr_matrix_N.loc[col1, col2]  # Correlation from the N-sample matrix
#             r_N_plus_1 = pearson_corr_matrix_N_plus_1.loc[col1, col2]  # Correlation from the N+1-sample matrix
            
#             # Compute the z-value and p-value for the difference
#             z, p_value = z_test_for_correlation_diff(r_N, r_N_plus_1, N) if Fisher is True else t_test_for_correlation_diff(r_N, r_N_plus_1, N)
            
#             p_values_matrix.loc[col1, col2] = p_value
#             z_matrix.loc[col1, col2] = z

#     p_values_matrix = p_values_matrix.apply(pd.to_numeric)
#     z_matrix = z_matrix.apply(pd.to_numeric)
#     return z_matrix, p_values_matrix

# def t_test_for_correlation_diff(r1, r2, N):
#     """
#     Compute the t-test for the difference between two correlation coefficients
#     without using Fisher transformation.
    
#     Args:
#         r1 (float): Pearson correlation coefficient from the N-sample matrix.
#         r2 (float): Pearson correlation coefficient from the N+1-sample matrix.
#         N (int): Sample size for the N-sample matrix.

#     Returns:
#         t (float): t-statistic for the difference.
#         p_value (float): p-value corresponding to the t-statistic.
#     """
#     # Compute the standard error of the difference
#     se_diff = np.sqrt((1 - r1**2) / (N - 1) + (1 - r2**2) / (N+1 - 1))
    
#     # Compute the t-statistic for the difference
#     t = (r1 - r2) / se_diff
    
#     # Compute the two-tailed p-value from the t-distribution
#     degrees_of_freedom = min(N - 1, N + 1 - 1)  # degrees of freedom for the t-distribution
#     p_value = 2 * (1 - stats.t.cdf(abs(t), df=degrees_of_freedom))
    
#     return t, p_value

def dfs(adj_matrix, node, visited):
    visited[node] = True
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
            dfs(adj_matrix, neighbor, visited)

def is_connected(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    
    # Start DFS from node 0 (or any arbitrary node)
    dfs(adj_matrix, 0, visited)
    
    # If all nodes are visited, the graph is connected
    return all(visited)

def resample_vectors(vectors):
    """
    Resamples longer vectors to match the length of the shortest vector,
    preserving the approximate distribution.

    Args:
        vectors: A list of NumPy arrays (vectors) of potentially different lengths.

    Returns:
        A list of resampled NumPy arrays, all with the length of the shortest vector.
    """

    if not isinstance(vectors, (list, np.ndarray)) or len(vectors) == 0: 
        return []

    lengths = [len(v) for v in vectors]
    min_length = min(lengths)

    resampled_vectors = []
    for vector in vectors:
        if len(vector) == min_length:
            resampled_vectors.append(vector)
        else:
            # Interpolate the vector
            x_original = np.linspace(0, 1, len(vector))
            x_resampled = np.linspace(0, 1, min_length)
            f = interp1d(x_original, vector, kind='linear', fill_value="extrapolate") #use linear interpolation
            resampled_vectors.append(f(x_resampled))

    return resampled_vectors

def do_t_test(distribA, distribB):
    t_stat, p_value = ttest_ind(distribA, distribB, equal_var=False)  # Welch's t-test

    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    print(f"Mean distribA : {np.mean(distribA):.3} | Mean distribB : {np.mean(distribB):.3}")

    # Interpret the results
    alpha = 0.05  # Common significance level
    if p_value < alpha:
        print("There is a statistically significant difference between distribA and distribB (p < 0.05).")
    else:
        print("No statistically significant difference found (p >= 0.05).")
    return t_stat, p_value