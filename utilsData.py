import os
import gzip
import pickle
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import gc

def load_data_from_patients_folder(base_folder, resample_max_length=None):
    """
    Imports the patients data from the archive given by the MD. Returns a dictionary of Pandas dataframes.
    Optionally resamples the SUV lines to a maximum length. If None, no resampling is performed.

    Args:
        base_folder (str): Path to the base folder containing patient data.
        resample_max_length (int, optional): Maximum length to resample SUV lines. If None, no resampling is performed.

    Returns:
        dict: A dictionary of Pandas dataframes.
    """
    data_dict = {}

    for patient_id in os.listdir(base_folder):
        patient_path = os.path.join(base_folder, patient_id)
        
        if os.path.isdir(patient_path):
            results_path = os.path.join(patient_path, "results", "dico.pickle.gz")
            
            if os.path.isfile(results_path):
                try:
                    with gzip.open(results_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, pd.DataFrame):
                        patient_df = data.copy()
                    else:
                        try:
                            patient_df = pd.DataFrame(data)
                        except Exception as e:
                            print(f"Failed to convert data to DataFrame for patient {patient_id}: {e}")
                            continue

                    if resample_max_length is not None:
                        try:
                            if 'SUV' in patient_df.index:
                                suv_series = patient_df.loc['SUV']

                                resampled_suv = {}
                                for col in suv_series.index: #Now iterate on the series index.
                                    try:
                                        original_vector = np.array(suv_series[col]) #Convert list to numpy array.
                                        original_length = len(original_vector)

                                        if original_length > resample_max_length:
                                            x_original = np.linspace(0, 1, original_length)
                                            x_resampled = np.linspace(0, 1, resample_max_length)

                                            f = interp1d(x_original, original_vector, kind='linear', fill_value="extrapolate")
                                            resampled_suv[col] = f(x_resampled).tolist() #Convert back to list.
                                        else:
                                            resampled_suv[col] = original_vector.tolist() #Convert back to list.

                                    except ValueError:
                                        print(f"Error processing SUV data for column {col} in patient {patient_id}. Skipping.")
                                        continue

                                patient_df.loc['SUV'] = pd.Series(resampled_suv) #Create series from resampled dict.

                        except Exception as resample_error:
                            print(f"Resampling error for patient {patient_id}: {resample_error}")
                    
                    data_dict[patient_id] = patient_df
                    gc.collect()
                
                except Exception as e:
                    print(f"Error loading data for patient {patient_id}: {e}")
    
    return data_dict

def add_mean_SUV_index(data_dict):
    """ Append one row to each patient dataframe which contains the mean of the SUV activation for each of its ROE """
    for patient_id, df in data_dict.items():
        if "SUV" in df.index:
            df.loc["mean_SUV"] =  df.loc["SUV"].apply(lambda cell: np.mean(cell))
        else:
            print(f"'SUV' index not found for patient {patient_id}. Skipping...")
    return data_dict


