import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 


# generated_files/descriptors/phase_letter/{files}
# generated_files/datasets/phase_letter/dataset_name/{files}
# generated_files/experiments/phase_letter/scaling_factor/dataset_name/{files}


def get_ae_descriptor(ae_dict, dataset_descriptor):
    path = f"generated_files/datasets/{ae_path_dict['phase']}_{ae_path_dict['letter']}/{ae_path_dict['dataset']}/data_descriptor.pickle"
    with open(path, "rb") as f:
        ae_dataset_descriptor = pickle.load(f)
    return ae_dataset_descriptor

def get_ae_latent_model(ae_dict, dataset_descriptor):
    scaling_factor = dataset_descriptor["scaling_factor"]
    path = f"generated_files/experiments/{ae_path_dict['phase']}_{ae_path_dict['letter']}/{scaling_factor}/{ae_path_dict['dataset']}/latent_model"
    ae_model = models.load_model(path)
    return ae_model

#Everything it returns is a list because of the time slice issue 
def get_ae_latent_space(ae_dict, x_columns, x_dataset_list):
    ae_model = get_ae_latent_model(ae_dict, dataset_descriptor)
    ae_dataset_descriptor = get_ae_descriptor(ae_dict, dataset_descriptor)

    latent_space = [] 
    #Recursive case - this AE depends on other AEs to preprocess input 
    if ae_dataset_descriptor["ae_letter"] != None:
        ae_dicts = dataset_object["ae_dicts"]
        for sub_dict in ae_dicts:
            ae_output = get_ae_latent_space(sub_dict, x_columns, x_dataset_list)
            if latent_space == []:
                 latent_space = ae_output
            else:
                 #This should actually work, but should definitely check. 
                 latent_space = np.hstack((latent_space, ae_output))
    #Base case - model does not depend on other AEs to preprocess inputs 
    else:
        model_inputs = ae_dataset_descriptor["input_fields"]
        relevant_indexes = []
        #This is the part we'll have to change if conv. 
        #We need to restrict the input to what the columns reflect 
        for input_col in model_inputs:
            relevant_indexes.append(x_columns.index(input_col))
        ae_input = x_dataset_list[:, relevant_indexes]
        latent_space = ae_model.predict(ae_input)
    return latent_space

 #This function could use a LOT better documentation    
def process_aes(dataset_descriptor, x_dataset_list):
    ae_dicts = dataset_object["ae_dicts"]
    #execute_list, ae_dict = build_ae_tree(ae_paths)
    x_columns = dataset_object["input_fields"]
    #For each identified autoencoder, in each stage. 
    latent_space = []
    #latent_space = []
    for ae_dict in ae_dicts:
        #This might need to be returned as a list 
        ae_output = get_ae_latent_space(ae_dict, x_columns, x_dataset_list)
        if latent_space == []:
            latent_space = ae_output
        else:
            latent_space = np.hstack((latent_space, ae_output))
    #I think returning x_key_list is probably an artifact
    return latent_space


def load_in_datasets(dataset_descriptor):
    all_fields = dataset_descriptor["input_fields"] + dataset_descriptor["output_fields"]
    all_fields = list(set(all_fields))
    datasets = dataset_descriptor["datasets"]
    using_datasets = []
    dataset_list = []
    for dataset in datasets:
        #Load it in - the normalized version 
        case = pd.read_csv(f"vital_csvs/{dataset}_normalized.csv")
        #Check that it has the right fields
        case_cols = list(case.columns)
        contains_needed_fields = all(feature in case_cols for feature in all_fields)
        if contains_needed_fields:
            using_datasets.append(dataset)
            #Add caseid as a column first
            case["caseid"] = dataset
            dataset_list.append(case)
    return dataset_list


#Slice into our time intervals 
def time_slice(dataset_descriptor, x_dataset, y_dataset, x_key_dataset, y_key_dataset):
    #Get sampling data 
    input_samples = dataset_descriptor["input_samples"]
    output_samples = dataset_descriptor["output_samples"]
    output_offset = dataset_descriptor["output_offset"]
    num_rows = len(x_dataset)
    #Start with no sequences 
    x_seq, y_seq, x_key_seq, y_key_seq = [], [], [], []
    #Starting indexes 
    x_start = 0
    x_end = input_samples-1
    y_start = x_end+output_offset
    y_end = y_start+output_samples-1
    #Get the sequences
    while y_end < num_rows - 1:
        x_array = x_dataset[x_start:x_end+1]
        x_key_array = x_key_dataset[x_start:x_end+1]
        if y_start == y_end:
            y_array = y_dataset[y_start]
            y_key_array = y_key_dataset[y_start]
        else:
            y_array = y_dataset[y_start:y_end+1]
            y_key_array = y_key_dataset[y_start:y_end+1]
        #Append new sequence slices to the overall samples array 
        x_seq.append(x_array)
        y_seq.append(y_array)
        x_key_seq.append(x_key_array)
        y_key_seq.append(y_key_array)
        #Increment
        x_start = x_start+1
        x_end = x_end+1
        y_start = y_start+1
        y_end = y_end+1
    #Finally, convert to np array anew
    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)
    x_key_seq = np.array(x_key_seq)
    y_key_seq = np.array(y_key_seq)
    return x_seq, y_seq, x_key_seq, y_key_seq


def print_input_data(x, y, x_key, y_key):
    print("X shape")
    print(x.shape)
    print("Y shape")
    print(y.shape)
    print("X key shape")
    print(x_key.shape)
    print("Y key shape")
    print(y_key.shape)

def create_dataset_from_dataset_descriptor(dataset_descriptor):
    #First, load in all the datasets from 
    list_of_datasets = load_in_datasets(dataset_descriptor)

    x_dataset_list = []
    y_dataset_list = []
    x_key_list = []
    y_key_list = []

    #Preprocessing by dataset 
    for dataset in list_of_datasets:
        #Get the relevant fields only, and convert to numpy 
        x_dataset = dataset[dataset_descriptor["input_fields"]].to_numpy()
        y_dataset = dataset[dataset_descriptor["output_fields"]].to_numpy()
        x_key_dataset = dataset[dataset_descriptor["keys"]].to_numpy()
        y_key_dataset = dataset[dataset_descriptor["keys"]].to_numpy()
        
        #Preprocess with AEs. 
        if dataset_descriptor["ae_letter"] != None:
            for ae_dict in dataset_descriptor["ae_dicts"]:
                x_dataset = process_aes(dataset_descriptor, x_dataset)

        #Time slice it if the target is LSTM
        if dataset_descriptor["target_model"] == "lstm":
            x_dataset, y_dataset, x_key_dataset, y_key_dataset = time_slice(dataset_descriptor, x_dataset, y_dataset, x_key_dataset, y_key_dataset)

        #Append to the full list - this is a but ugly, but otherwise functional
        #X
        if x_dataset_list == []:
            x_dataset_list = x_dataset
        else:        
            x_dataset_list = np.vstack((x_dataset_list, x_dataset))
        #Y
        if y_dataset_list == []:
            y_dataset_list = y_dataset
        else:
            y_dataset_list = np.vstack((y_dataset_list, y_dataset))
        #X KEY
        if x_key_list == []:
            x_key_list = x_key_dataset
        else:
            x_key_list = np.vstack((x_key_list, x_key_dataset))
        #Y KEY 
        if y_key_list == []:
            y_key_list = y_key_dataset
        else:
            y_key_list = np.vstack((y_key_list, y_key_dataset))

    #print_input_data(x_dataset_list, y_dataset_list, x_key_list, y_key_list)

    dataset_result = {
        "x": x_dataset_list,
        "y": y_dataset_list,
        "x_key": x_key_list,
        "y_key": y_key_list
        }

    return dataset_result

# "x_columns": list of names, in order, of the x columns.

# "y_columns": list of names, in order, of the y columns.

# "test": Bool, whether or not a test run 

# "target_model": "ae" or "time_regression" or "predict" depending on the target model for the dataset 

# "delete_stream" - name or list of names of datastreams to delete 

# "phase_metrics" - name of metrics file

# "base_dataset_name": phase and letter identifying base dataset

# "base_name": phase + letter + exp

# "deep_lstm" T/F

# "deep_ae": T/F 

# #Don't think the below needed? 

# "ae_synthesis" - optional, based on what we fuse one

# "conv": T/F

# "conv_and_prev_ae": T/F 

# ## Dataset Result

# x_vect - 

# y_vect - 

# x_key - 

# y_key - 

# x_raw - 

# y_raw - 