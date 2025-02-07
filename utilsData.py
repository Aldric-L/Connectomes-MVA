import os
import gzip
import pickle
import pandas as pd
import numpy as np

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