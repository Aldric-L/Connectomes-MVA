import os
import gzip
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def load_data_from_patients_folder(base_folder):
    """ Import the patients data from the archive given by the MD. Returns a dictionary of Pandas dataframes"""
    data_dict = {}

    # Walk through each subfolder in the base folder
    for patient_id in os.listdir(base_folder):
        patient_path = os.path.join(base_folder, patient_id)
        
        # Ensure it's a directory
        if os.path.isdir(patient_path):
            results_path = os.path.join(patient_path, "results", "dico.pickle.gz")
            
            # Check if the results folder and dico.pickle.gz file exist
            if os.path.isfile(results_path):
                try:
                    # Load the pickle.gz file
                    with gzip.open(results_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Ensure the data is in a DataFrame format
                    if isinstance(data, pd.DataFrame):
                        data_dict[patient_id] = data
                    else:
                        try:
                            data_dict[patient_id] = pd.DataFrame(data)
                        except Exception as e:
                            print(f"Failed to convert data to DataFrame for patient {patient_id}: {e}")
                
                except Exception as e:
                    print(f"Error loading data for patient {patient_id}: {e}")
    
    # Return the Python dictionary (instead of NumPy) for better flexibility
    return data_dict


def add_mean_SUV_index(data_dict):
    """ Append one row to each patient dataframe which contains the mean of the SUV activation for each of its ROE """
    for patient_id, df in data_dict.items():
        if "SUV" in df.index:
            df.loc["mean_SUV"] =  df.loc["SUV"].apply(lambda cell: np.mean(cell))
        else:
            print(f"'SUV' index not found for patient {patient_id}. Skipping...")
    return data_dict


def plot_connectome(G):
    # Step 4: Visualization - Circular layout with colored edges
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


    # Display the plot
    plt.title('Connectome Graph on Circular Layout with Edge Intensity')
    plt.axis('off')  # Hide axes for better aesthetics
    plt.show()