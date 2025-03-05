import os
import gzip
import pickle
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import gc
import itertools

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

def create_meta_organs(patient_data, meta_organs=None, resample_max_length=None):
    """
    Creates meta-organ dataframes from patient-specific organ data.

    This function aggregates organ-level data into meta-organ data based on a provided
    mapping of meta-organs to their constituent organs. It calculates the total number
    of voxels and combines the Standardized Uptake Value (SUV) data for each meta-organ.
    Optionally, it resamples the combined SUV data to a specified maximum length using
    linear interpolation.

    Args:
        patient_data (dict): A dictionary where keys are patient IDs and values are
            Pandas DataFrames. Each DataFrame should contain organ-level data,
            including 'name', 'nb_voxels', and 'SUV' columns. The 'name' column
            should be in the first row, and the dataframe should be transposed
            before being passed to this function.
        meta_organs (dict): A dictionary defining meta-organs and their constituent
            organs. Keys are meta-organ names, and values are sets of organ names.
        resample_max_length (int, optional): If provided, the combined SUV data for
            each meta-organ will be resampled to this maximum length using linear
            interpolation. Defaults to None, which means no resampling is performed.

    Returns:
        dict: A dictionary where keys are patient IDs and values are Pandas DataFrames.
            Each DataFrame represents the meta-organ data for the corresponding patient,
            containing 'name', 'nb_voxels', 'SUV', and 'mean_SUV' columns. The returned
            DataFrames are transposed, with meta-organ names as columns.

    Raises:
        ValueError: If an error occurs during SUV data processing, such as invalid
            data types or issues with resampling. A warning message is printed, and
            the problematic meta-organ is skipped.
    """
    if meta_organs is None:
        meta_organs = {
            "axial_skeleton": {"sacrum", "vertebrae_S1",
            'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2',
            'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10',
            'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6',
            'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2',
            'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5',
            'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1',
                "skull", 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4',
            'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8',
            'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12',
            'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4',
            'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8',
            'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12', "sternum", 'costal_cartilages'},
            "appendicular_skeleton": { 'humerus_left', 'humerus_right',
            'scapula_left', 'scapula_right', 'clavicula_left',
            'clavicula_right', 'femur_left', 'femur_right', 'hip_left',
            'hip_right'},
            "skeleton_muscles": {'iliopsoas_left', 'iliopsoas_right', 'skeletal_muscle', 'autochthon_left', 'autochthon_right', 'gluteus_maximus_left',
            'gluteus_maximus_right', 'gluteus_medius_left',
            'gluteus_medius_right', 'gluteus_minimus_left',
            'gluteus_minimus_right',},
            "lung": {"lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"},
            "VAT_fat": {"torso_fat"},
            "SAT_fat": {"subcutaneous_fat"},
            "liver": {"liver"},
            "spleen": {"spleen"},
            "kidney": {"kidney_right", "kidney_left", "adrenal_gland_right", "adrenal_gland_left", "kidney_cyst_left", "kidney_cyst_right"},
            "heart_left": {"atrial_appendage_left", 'heart_atrium_left', 'heart_ventricle_left',},
            "heart_right": {'heart_atrium_right', 'heart_ventricle_right'},
            "heart": {"heart", "heart_myocardium",'aorta'},
            "brain": {'brain'},
            "other": {"gallbladder", "stomach", "pancreas","esophagus", "trachea", "thyroid_gland", "duodenum", "colon", "urinary_bladder", "prostate",
                    "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right", "common_carotid_artery_right",
                    "common_carotid_artery_left", "brachiocephalic_vein_left", "superior_vena_cava", "inferior_vena_cava", 
                    "portal_vein_and_splenic_vein", 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left',
                    'iliac_vena_right', "spinal_cord", 'pulmonary_artery', "small_bowel",}
        }
    meta_data = {}
    
    for patient, organs_df in patient_data.items():
        meta_df = pd.DataFrame()
        
        for meta_name, organ_set in meta_organs.items():
            organs_in_meta = organs_df.loc[:, organs_df.loc["name"].isin(organ_set)]
            
            if not organs_in_meta.empty:
                total_voxels = organs_in_meta.loc["nb_voxels"].sum()         
                combined_SUV = list(itertools.chain.from_iterable(organs_in_meta.loc["SUV"].dropna().tolist()))
                if resample_max_length is not None:
                    try:
                        original_vector = np.array(combined_SUV) 
                        original_length = len(original_vector)

                        if original_length > resample_max_length:
                            x_original = np.linspace(0, 1, original_length)
                            x_resampled = np.linspace(0, 1, resample_max_length)

                            f = interp1d(x_original, original_vector, kind='linear', fill_value="extrapolate")
                            combined_SUV = f(x_resampled).tolist() #Convert back to list.
                        else:
                            combined_SUV = original_vector.tolist() #Convert back to list.
                        
                    except ValueError:
                        print(f"Error processing SUV data for column {meta_name} in patient {patient}. Skipping.")

                meta_df = pd.concat([meta_df, pd.DataFrame({
                    "name": [meta_name],
                    "nb_voxels": [total_voxels],
                    "SUV": [combined_SUV],
                    "mean_SUV": np.mean(combined_SUV)
                })], ignore_index=True)
        
        meta_data[patient] = meta_df.transpose()
        gc.collect()
    
    return meta_data

