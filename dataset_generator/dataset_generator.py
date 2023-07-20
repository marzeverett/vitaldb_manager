import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas as pd 
import importlib
import numpy as np
from datetime import datetime, timedelta
import os
import json 
import pickle 
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import datasets, layers, models

#Reverse_mapping help here: 
#https://www.techiedelight.com/build-dictionary-from-list-of-keys-values-python/

#Pickle help here: https://ianlondon.github.io/blog/pickling-basics/ 
#Help from: https://www.statology.org/pandas-keep-columns/
#From here: https://www.geeksforgeeks.org/how-to-create-an-empty-dataframe-and-append-rows-columns-to-it-in-pandas/
#From here: https://pandas.pydata.org/docs/user_guide/merging.html 
#From here: https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
#https://stackoverflow.com/questions/25254016/get-first-row-value-of-a-given-column 
#https://www.kdnuggets.com/2021/05/deal-with-categorical-data-machine-learning.html 
#https://www.geeksforgeeks.org/how-to-add-and-subtract-days-using-datetime-in-python/
#https://www.geeksforgeeks.org/python-os-makedirs-method/ 
#https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/ 
#https://pbpython.com/categorical-encoding.html 
#https://stackoverflow.com/questions/30510562/get-mapping-of-categorical-variables-in-pandas 
#https://www.geeksforgeeks.org/normalize-a-column-in-pandas/ 
#https://www.w3schools.com/python/python_howto_remove_duplicates.asp 
#https://numpy.org/doc/stable/reference/generated/numpy.load.html
#https://sparkbyexamples.com/pandas/pandas-drop-columns-with-nan-none-values/  
#https://www.techbeamers.com/program-python-list-contains-elements/ 
#https://stackoverflow.com/questions/42916029/indexing-over-the-last-axis-when-you-dont-know-the-rank-in-advance 



#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]
orch = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
#43 of these 
valid_cases = [4481, 3719, 1292, 397, 2327, 6297, 5018, 6009, 1820, 2332, 4255, 1191, 1959, 553, 3631, 2738, 818, 1590, 55, 5175, 4283, 5693, 1730, 5442, 3524, 4684, 5837, 1231, 6227, 985, 3930, 2267, 4573, 5983, 2272, 6246, 5607, 1900, 3694, 2168, 1785, 1018, 251]
total_tracks = snu+bis+orch+solar

categorical = []

track_dict = {
    "snu": snu,
    "bis": bis,
    "orch": orch,
    "solar": solar
}


dataset_descriptor_1 = {
    "datasets": [4481],
    "input_fields": orch,
    "output_fields": orch,
    "input_slices": 30,
    "output_slices": 10,
    "output_slices_offset": 1,
    "task_type": "regression",
    "dataset_name": "test_1",
}



#Returns folder_path,  dataset descriptor filepath and dataset result filepath. 
def get_data_filepaths(dataset_object):
    dataset_name = dataset_object["dataset_name"]
    sub_path = dataset_object["dataset_folder_path"]
    full_path = sub_path+ "/" + str(dataset_name)+"/"
    dataset_result_path = full_path+ "dataset_result.pickle"
    dataset_descriptor_path = full_path+ "dataset_descriptor.pickle"
    return full_path, dataset_result_path, dataset_descriptor_path

def load_in_data(dataset_object):
    full_path, dataset_result_path, dataset_descriptor_path = get_data_filepaths(dataset_object)
    with open(dataset_result_path, "rb") as f:
        dataset_result = pickle.load(f)
    with open(dataset_descriptor_path, "rb") as f:
        dataset_object = pickle.load(f)
    return dataset_result, dataset_object

def save_dataset(x, y, x_key, y_key, x_raw, y_raw, dataset_object):
    full_path, dataset_result_path, dataset_descriptor_path = get_data_filepaths(dataset_object)
    os.makedirs(full_path, exist_ok=True)
    #Make the dataset result object. 
    dataset_result = {
        "x": x,
        "y": y,
        "x_key": x_key,
        "y_key": y_key,
        "x_raw": x_raw,
        "y_raw": y_raw
    }
    #Save to pickle files.
    with open(dataset_result_path, "wb") as f:
        pickle.dump(dataset_result, f, protocol=4)
    with open(dataset_descriptor_path, "wb") as f:
        pickle.dump(dataset_object, f, protocol=4)


def get_input_output_fields(dataset_object, field_object_index):
    i_fields = dataset_object[field_object_index]
    send_i_fields = []
    if isinstance(i_fields, dict):
        for dataset_name in list(i_fields.keys()):
            prefix = dataset_name + "_"
            fields = list(i_fields[dataset_name].values())
            for item in fields:
                item = prefix + item
                send_i_fields.append(item)
    else:
        dataset_list = dataset_object["datasets"]
        for dataset_name in dataset_list:
            prefix = dataset_name + "_"
            for item in i_fields:
                item = prefix + item
                send_i_fields.append(item)
    return send_i_fields    

#Deal with categorical data 
def handle_categorical(df, dataset_object):
    try:
        categorical_list = dataset_object['categorical']
        if "cat_codes" not in list(dataset_object.keys()):
            dataset_object["cat_codes"] = {}
        for field in categorical_list:
            df[field]= df[field].astype('category')
            field_categories = dict(enumerate(df[field].cat.categories)) 
            #print(field_categories)
            #print(list(df[field].cat.codes))
            df[field] = df[field].cat.codes
            dataset_object["cat_codes"][field] = field_categories
    except Exception as e:
        print("Could not code categorical variables: ", e)
    return df  

#Normalize data 
def normalize_data(dataset_name, df, dataset_object, fields):
    prefix_string = dataset_name+"_"
    concat_key = dataset_object["concat_key"]
    if "normalization_data" not in list(dataset_object.keys()):
        dataset_object["normalization_data"] = {}
    #From the geeks for geeks tutorial on normalization 
    for field in fields: 
        if field != "concat_key":
            try:
                max_val = df[field].max()
                min_val = df[field].min()
                n_dict = {
                    "max": max_val,
                    "min": min_val
                }
                diff_between = max_val - min_val
                #Handle potential divide by zero issue 
                if diff_between != 0:
                    df[field] = (df[field] - min_val) / (max_val - min_val)
                    field_name = prefix_string+field
                    dataset_object["normalization_data"][field_name] = n_dict
            except Exception as e:
                pass
    return df 

#Create a dataframe with preprocessed data, and reduce to only necessary data columns 
def create_reduced_dataframe(dataset_name, df, dataset_object):
    normalize = dataset_object["normalize"]
    #Handle categorical variables 
    df = handle_categorical(df, dataset_object)
    #Need input and output fields, concat_key, and renamed here. 
    concat_key = dataset_object["concat_key"]
    i_fields = dataset_object["input_fields"]
    o_fields = dataset_object["output_fields"]
    i_dict = {}
    o_dict = {}
    if isinstance(i_fields, dict):
        if dataset_name in list(i_fields.keys()):
            i_dict = i_fields[dataset_name]
            i_fields = list(i_fields[dataset_name].keys())
        else:
            i_fields = []
    if isinstance(o_fields, dict):
        if dataset_name in list(o_fields.keys()):
            o_dict = o_fields[dataset_name]
            o_fields = list(o_fields[dataset_name].keys())
        else:
            o_fields = []
    #Normalize if necessary
    if normalize:
        normalize_fields = i_fields + o_fields
        normalize_fields = list(dict.fromkeys(normalize_fields))
        df = normalize_data(dataset_name, df, dataset_object, normalize_fields)
    keep_cols = i_fields + o_fields + [concat_key]

    #Handle any missing cols 
    actual_cols = list(df.columns)
    final_cols = list(set(keep_cols) & set(actual_cols))
    df = df[final_cols]
    #df = df[keep_cols]

    if i_dict != {}:
        df = df.rename(columns=i_dict)
    if o_dict != {}:
        df = df.rename(columns=o_dict)
    return df

#Merge the dataframes together, prefixing columns with the dataset they came from 
def create_merged_df(dataset_object):
    whole_df = pd.DataFrame()
    dataset_list = dataset_object["datasets"]
    concat_key = dataset_object["concat_key"]
    input_fields = dataset_object["input_fields"]
    #Will map the renamed field back to its source data, useful for unnormalizing it.  
    reverse_mapping = {}
    #dictionary = dict(zip(keys, values))
    i = 0
    for dataset in dataset_list:
        prefix_string = dataset+"_"
        prefix_concat = prefix_string+concat_key
        #Import the appropriate module 
        save_name = "pickled_datasets/"+dataset+".pkl"
        df = pd.read_pickle(save_name)
        #Drop any columns that are all NaN
        #Reduce where we can 
        df = create_reduced_dataframe(dataset, df, dataset_object)
        #Rename 
        old_columns = list(df.columns)
        df = df.add_prefix(prefix_string)
        df = df.rename(columns={prefix_concat: concat_key})
        new_columns = list(df.columns)
        mapping = dict(zip(new_columns, old_columns))
        reverse_mapping.update(mapping)
        #Merge 
        if i > 0:
            #whole_df = pd.merge(whole_df, df, on=concat_key)
            whole_df = pd.merge(df, whole_df, on=concat_key)
        else:
            whole_df = df
        i = i+1
    dataset_object["reverse_mapping"] = reverse_mapping
    return whole_df 

#Drop or fill missing data (recommend fill)
def deal_with_missing_data(df, dataset_object):
    clean_method = dataset_object["clean_method"]
    if clean_method == "drop": 
        df = df.dropna()
        #Line gets rid of dropped data 
        df = df.reset_index(drop=True)
    elif clean_method == "fill":
        df = df.fillna(method="pad")
        #Drop columns that you can't pad. 
        #This shifts data that was missing to the time period it was available. 
        #Since we filled ahead of time. 
        df= df.dropna()
        df = df.reset_index(drop=True)
    return df

#Don't have great error checking on this... 
#But it gets the latent space. Admittedly, this could fail with latent space being []
#But in that case it probably should fail... 
def get_ae_latent_space(path, x_columns, x_vect, x_key_vect):
    full_model_path = path + "latent_model"
    ae_model = models.load_model(full_model_path)
    full_dd_path = path+"dataset_descriptor.pickle"
    latent_space = [] 
    with open(full_dd_path, "rb") as f:
        dataset_object = pickle.load(f)
    #Recursive case - this depends on other AEs
    if "ae_paths" in list(dataset_object.keys()):
        ae_paths = dataset_object["ae_paths"]
        for path in ae_paths:
            ae_output = get_ae_latent_space(path, x_columns, x_vect,x_key_vect)
            if latent_space == []:
                 latent_space = ae_output
            else:
                 #This should actually work, but should definitely check. 
                 latent_space = np.hstack((latent_space, ae_output))
    #Base case - model does not depend on AE outputs 
    else:
        model_inputs = dataset_object["x_columns"]
        relevant_indexes = []
        #This is the part we'll have to change if conv. 
        for input_col in model_inputs:
            relevant_indexes.append(x_columns.index(input_col))
        ae_input = x_vect[:, relevant_indexes]
        # print("Here")
        # print(ae_input.shape)
        # print(ae_model.summary())
        latent_space = ae_model.predict(ae_input)
    return latent_space

 #This function could use a LOT better documentation    
def process_aes(dataset_object, x_vect, x_key_vect):
    ae_paths = dataset_object["ae_paths"]
    #execute_list, ae_dict = build_ae_tree(ae_paths)
    x_columns = dataset_object["x_columns"]
    #For each identified autoencoder, in each stage. 
    #CHECK
    latent_space = []
    #latent_space = []
    for path in ae_paths:
        ae_output = get_ae_latent_space(path, x_columns, x_vect, x_key_vect)
        if latent_space == []:
            latent_space = ae_output
        else:
            latent_space = np.hstack((latent_space, ae_output))

    return latent_space, x_key_vect


#This makes sure only columns present both in the spec and the dataset make it in. 
def get_actual_input_output_columns(dataset_object, df):
    input_fields = get_input_output_fields(dataset_object, "input_fields")
    output_fields = get_input_output_fields(dataset_object, "output_fields")
    actual_cols = list(df.columns)
    actual_input = list(set(input_fields)&set(actual_cols))
    actual_output = list(set(output_fields)&set(actual_cols))
    return actual_input, actual_output

#Time Slice! 
#This also assumes you already have the data columns you want. 
#This time slice function assumes there are no day gaps - I think this works for 
#this dataset, but is probably not broadly applicable 
def time_slice(df, dataset_object, x, y, x_key, y_key):
    input_slices_days = dataset_object["input_slices_days"]
    output_slices_days = dataset_object["output_slices_days"]
    output_offset_days = dataset_object["output_offset_days"]
    num_rows = len(df)
    x_vect, y_vect, x_key_vect, y_key_vect = [], [], [], []
    x_start = 0
    x_end = input_slices_days-1
    y_start = x_end+output_offset_days
    y_end = y_start+output_slices_days-1
    #Get x and y values indexed properly. 
    while y_end < num_rows-1:
        x_array = x[x_start:x_end+1]
        x_key_array = x_key[x_start:x_end+1]
        if y_start == y_end:
            y_array = y[y_start]
            y_key_array = y_key[y_start]
        else:
            y_array = y[y_start:y_end+1]
            y_key_array = y_key[y_start:y_end+1]
        x_vect.append(x_array)
        y_vect.append(y_array)
        x_key_vect.append(x_key_array)
        y_key_vect.append(y_key_array)
        #If it is nested, this is where we go for it. 
        #Increment    
        x_start = x_start+1
        x_end = x_end+1
        y_start = y_start+1
        y_end = y_end+1
    #Finally, convert to a numpy array 
    x_vect = np.array(x_vect)
    y_vect = np.array(y_vect)
    x_key_vect = np.array(x_key_vect)
    y_key_vect = np.array(y_key_vect)
    return x_vect, y_vect, x_key_vect, y_key_vect

#Prints stuff. Kinda useful for debugging. 
def print_output_data_info(actual_input, x_vect, y_vect, x_key, y_key):
    print("Cols ", actual_input)
    print("Number of x samples", len(x_vect))
    print("Number of y samples", len(y_vect))
    print("First x sample", x_vect[0])
    print("First y sample", y_vect[0])
    print("x_key_shape", x_key.shape)
    print("y_key shape", y_key.shape)
    #print("First x key", x_key[0])
    #print("First y key", y_key[0])
    print("X shape:", x_vect.shape)
    print("Y shape:", y_vect.shape)

#Formats the data so the model can accept it
#Takes care of autoencoders preprocessing and time-slicing, too. 
def format_data_model_ready(dataset_object, df):
    target_model = dataset_object["target_model"]
    concat_key_fields = [dataset_object["concat_key"]]
    #Handles any missing columns 
    actual_input, actual_output = get_actual_input_output_columns(dataset_object, df)
    #Slice x and y, and x and y keys. Add this info to the dataset descriptor
    x_cols = df[actual_input]
    y_cols = df[actual_output]
    key_cols = df[concat_key_fields]
    x_cols_names = list(x_cols.columns)
    y_cols_names = list(y_cols.columns)
    dataset_object["x_columns"] = x_cols_names
    dataset_object["y_columns"] = y_cols_names
    #Convert to numpy array
    x_vect = x_cols.to_numpy()
    y_vect = y_cols.to_numpy()
    x_key_vect = key_cols.to_numpy()
    y_key_vect = key_cols.to_numpy()

    x_raw = x_vect
    y_raw = y_vect

    if dataset_object["conv"] == True and dataset_object["conv_and_prev_ae"] == False:
        x_matrix, y_matrix = matricize_dataset(x_vect, y_vect, dataset_object)
        x_vect = x_matrix
        y_vect = y_matrix
    #If this model needs to be preprocessed through an ae, do that first
    if "ae_paths" in list(dataset_object.keys()):
        x_vect, x_key_vect = process_aes(dataset_object, x_vect, x_key_vect)
    #If this is a time regression model, we need to slice it up. 
    if target_model == "time_regression":
        #For conv? - Maybe CHECK, CHANGE HERE 
        y_vect = y_raw
        x_vect, y_vect, x_key_vect, y_key_vect = time_slice(df, dataset_object, x_vect, y_vect, x_key_vect, y_key_vect)
    #Change also here 
    #print_output_data_info(actual_input, x_vect, y_vect, x_key_vect, y_key_vect)
    return x_vect, y_vect, x_key_vect, y_key_vect, x_raw, y_raw


def print_dataset_info(df):
    print(df.describe())
    print(list(df.columns))
    print("--------------")



#Take in a dataset object, create it, and save it. 
#Takes in a dataset object, returns 
def create_dataset_from_dataset_object(dataset_object):
    #Normalize the rest
    #For each dataset, split into x/y
    #For each dataset, preprocess with an ae if necessary
    #For each dataset, split into time chunks, if necessary
    #Create the final merged x and y values and keys 
    #Might be good to have the raw data so you don't have
    #To normalize and un-normalize? - Lot of data processing there 

    #Just need to keep raw -- which might be a tad tricky, depending
    #on how much data will be generated (a lot)

    #Can we really re-use datasets here? Not really
    #So might make more sense to describe them together?? 


    #Next 10 second prediction of vitals
    #30 second input

    #Regressing: vitals
    #Prediction: dis_risk, emop, gluc_risk (where possible)
    #Will transfer learn based on datastream only (as before)

    #No Conv




    # #1. Creates the merged dataset with the necessary fields 
    # #I don't think we'll merge the datasets here - bad idea. 
    # df = create_merged_df(dataset_object)
    # #2. Drop or fill N/A data - Already done, so won't use. 
    # df = deal_with_missing_data(df, dataset_object)
    # #Change here
    # #print_dataset_info(df)
    # #3. Format for Keras model - this handles LSTM, AE, and (soon) Nested
    # x_vect, y_vect, x_key, y_key, x_raw, y_raw = format_data_model_ready(dataset_object, df)                  
    # #4. Save. 
    # save_dataset(x_vect, y_vect, x_key, y_key, x_raw, y_raw, dataset_object)
    # return x_vect, y_vect, x_key, y_key, dataset_object


def return_test_dataset():
    dataset_result, dataset_descriptor = load_in_data(dataset_1)
    return dataset_result, dataset_descriptor

