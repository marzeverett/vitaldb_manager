#Help from:
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas 
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/ 

import pandas as pd
import matplotlib.pyplot as plt 
import json 
import os 
import scipy.stats as stats 
import seaborn as sn

#Group by Documentation
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html 

# colnames=['TIME', 'X', 'Y', 'Z'] 
# user1 = pd.read_csv('dataset/1.csv', names=colnames, header=None)
#https://www.geeksforgeeks.org/how-to-plot-multiple-data-columns-in-a-dataframe/ 



groups = {
    "ae": ["E", "H", "L", "S", "U", "X", "Z", "AC"],
    "lstm": ["A", "B", "C", "D", "F", "G", "I", "J",
                "M", "N", "Q", "T", "V", "W", "Y", "AA", "AB", "AD"
                ]

}

col_names = {
    "lstm": [
            "phase",
            "letter",
            "model_group",
            "datastream_scheme",
            "location_scheme",
            "ds_combo",
            "l_combo",
            "input_days",
            "output_days",
            "loss",
            "mse",
            "mape",
            "mae",
            "dataset_size",
            "training_time",
            "experiment_name",
            "dataset_name",
            "epochs",
            "inputs",
            "outputs"
    ],

    "ae": [
            "phase",
            "letter",
            "model_group",
            "datastream_scheme",
            "location_scheme",
            "ds_combo",
            "l_combo",
            "input_days",
            "output_days",
            "loss",
            "mse",
            "dataset_size",
            "training_time",
            "experiment_name",
            "dataset_name",
            "epochs",
            "inputs",
            "outputs"
        ],
    "prediction": [
            "phase",
            "letter",
            "model_group",
            "datastream_scheme",
            "location_scheme",
            "ds_combo",
            "l_combo",
            "input_days",
            "output_days",
            "loss",
            "mse",
            "binary_accuracy",
            "precision",
            "recall",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "dataset_size",
            "training_time",
            "experiment_name",
            "dataset_name",
            "epochs",
            "inputs",
            "outputs",
            "f1"
    ],
}


def return_aggregate_metrics_dict():
    aggregate_metrics = {
        "lstm": {
            "letter": [],
            "phase_letter": [],
            "mean_mse": [],
            "min_mse": [],
            "max_mse": [],
            "stdev_mse": [],
            "mean_mape": [],
            "min_mape": [],
            "max_mape": [],
            "stdev_mape": [],
            "mean_mae": [],
            "min_mae": [],
            "max_mae": [],
            "stdev_mae": [],
            "mean_training_time": [],
            "mean_num_epochs": [],
            "location_scheme": [],
            "datastream_scheme": [],
            "num_experiments": [],
        },

        "ae":  {
            "letter": [],
            "phase_letter": [],
            "mean_mse": [],
            "min_mse": [],
            "max_mse": [],
            "stdev_mse": [],
            "mean_training_time": [],
            "mean_num_epochs": [],
            "location_scheme": [],
            "datastream_scheme": [],
            "num_experiments": [],
        },
        "prediction": {
            "letter": [],
            "phase_letter": [],
            "mean_mse": [],
            "min_mse": [],
            "max_mse": [],
            "stdev_mse": [],
            "mean_f1": [],
            "min_f1": [],
            "max_f1": [],
            "stdev_f1": [],
            "mean_training_time": [],
            "mean_num_epochs": [],
            "location_scheme": [],
            "datastream_scheme": [],
            "num_experiments": [],
        }
    }
    return aggregate_metrics


df_dict = {}


def read_in_dfs(file_path, phase, ingroup="lstm", prediction=False):
    df_dict = {}
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename)
        if os.path.isfile(f):
            phase_letter = filename.replace("main_metrics.csv", "") 
            letter_group = phase + "_"
            letter = phase_letter.replace(letter_group, "")
            #If of interest 
            if letter in groups[ingroup]:
                sub_dict = {}
                sub_dict["letter"] = letter
                sub_dict["phase_letter"] = phase_letter
                if prediction:
                    cols = col_names["prediction"]
                    sub_dict["df"] = pd.read_csv(f)
                else:
                    cols = col_names[ingroup]
                    sub_dict["df"] = pd.read_csv(f, names=cols)
                df_dict[letter] = sub_dict

    return df_dict 


def read_in_df_single(letter, phase, ingroup="lstm", prediction=False):
    file_path = f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv"
    if prediction:
        cols = col_names["prediction"]
        df = pd.read_csv(file_path)
    else:
        cols = col_names[ingroup]
        df = pd.read_csv(file_path, names=cols)
    return df 

#This is disgusting but I'm too lazy to change it
def get_metric(df_dict, metric):
    df = df_dict["df"]
    return_metric=None
    if metric == "letter": 
        return_metric = df_dict["letter"]
    elif metric == "phase_letter":
        return_metric = df_dict["phase_letter"]
    elif metric == "mean_mse":
        return_metric = round(df["mse"].mean(), 10)
    elif metric == "min_mse":
        return_metric = round(df["mse"].min(), 10)
    elif metric == "max_mse":
        return_metric = round(df["mse"].max(), 10)
    elif metric == "stdev_mse":
        return_metric = round(df["mse"].std(), 10)
    elif metric == "mean_f1":
        return_metric = round(df["f1"].mean(), 10)
    elif metric == "min_f1":
        return_metric = round(df["f1"].min(), 10)
    elif metric == "max_f1":
        return_metric = round(df["f1"].max(), 10)
    elif metric == "stdev_f1":
        return_metric = round(df["f1"].std(), 10)    
    elif metric == "mean_mape":
        return_metric = round(df["mape"].mean(), 10)
    elif metric == "min_mape":
        return_metric = round(df["mape"].min(), 10)
    elif metric == "max_mape":
        return_metric = round(df["mape"].max(), 10)
    elif metric == "stdev_mape":
        return_metric = round(df["mse"].std(), 10)
    elif metric == "mean_mae":
        return_metric = round(df["mae"].mean(), 10)
    elif metric == "min_mae":
        return_metric = round(df["mae"].min(), 10)
    elif metric == "max_mae":
        return_metric = round(df["mae"].max(), 10)
    elif metric == "stdev_mae":
        return_metric = round(df["mae"].std(), 10)
    elif metric == "mean_training_time":
        return_metric = round(df["training_time"].mean(), 10)
    elif metric == "mean_num_epochs":
        return_metric = round(df["epochs"].mean(), 10)
    elif metric == "location_scheme":
        return_metric = df["location_scheme"].min()
    elif metric == "datastream_scheme":
        return_metric = df["datastream_scheme"].min()
    elif metric ==  "num_experiments":
        return_metric = len(df.index)
    return return_metric



def calc_aggregate_metrics(df_dict, scheme, prediction=False):
    #In AG Metrics
    metrics_dict = {}
    aggregate_metrics = return_aggregate_metrics_dict()
    if prediction:
        metrics_dict = aggregate_metrics["prediction"].copy()
    else:
        metrics_dict = aggregate_metrics[scheme].copy()
    #For each present letter 
    for letter in list(df_dict.keys()):
        #Get the dataframe 
        letter_dict = df_dict[letter]
        #Get the metrics 
        for metric in list(metrics_dict.keys()):
            metrics_dict[metric].append(get_metric(letter_dict, metric))
    return metrics_dict


def save_results(filename, save_dict):
    df = pd.DataFrame(save_dict)
    df.to_csv(filename, index=False)


def save_graphs(phase, scheme, file_path):
    df_dict = read_in_dfs(file_path, phase, ingroup=scheme)
    #test_df = df_dict["AF"]["df"]
    #print(test_df.head())
    metrics_dict = calc_aggregate_metrics(df_dict, scheme)
    save_name = f"{phase}_{scheme}_aggregate_metrics"
    #print(metrics_dict)
    save_results(save_name, metrics_dict)






def table_letters(kind, letters, file_path_1, phase_1, scheme_1, prediction=False):
    letters_dict = {}
    df_1 = read_in_dfs(file_path_1, phase_1, ingroup=scheme_1, prediction=prediction)
    metrics_dict_1 = calc_aggregate_metrics(df_1, scheme_1, prediction=prediction)
    #Row are metrics, columns are letters 
    letters_dict["Metric"] = ["mean_mse", "min_mse", "max_mse", "stdev_mse", "mean_training_time", "mean_num_epochs"]
    if prediction:
        letters_dict["Metric"] = ["mean_f1", "min_f1", "max_f1", "stdev_f1", "mean_training_time", "mean_num_epochs"]
    for letter in letters:
        if letter in metrics_dict_1["letter"]:
            letter_index = metrics_dict_1["letter"].index(letter)
            #Get og metrics
            new_list = []
            for item in letters_dict["Metric"]:
                new_list.append(metrics_dict_1[item][letter_index])
            letters_dict[letter] = new_list

    save_name = f"{phase_1}_analysis/table_{kind}_{scheme_1}_metrics"
    save_results(save_name, letters_dict)
    





def test_letters(kind, letters, file_path_1, phase_1, scheme_1, prediction=False):
    letters_dict = {}
    df_1 = read_in_dfs(file_path_1, phase_1, ingroup=scheme_1, prediction=prediction)
    #Row are metrics, columns are letters 

    #Maybe first, get all columns and labels
    labels_list = []
    columns_list = []

    metric = "mse"
    if prediction:
        metric = "f1"

    for letter in letters:
        if letter in list(df_1.keys()):
            labels_list.append(letter)
            columns_list.append(df_1[letter]["df"][metric])

    letters_dict["letter"] = []
    for i in range(0, len(labels_list)):
        for j in range(0, len(labels_list)):
            if labels_list[j] not in list(letters_dict.keys()):
                letters_dict[labels_list[j]] = []
            if labels_list[i] != labels_list[j]:                
                col1 = columns_list[i]
                col2 = columns_list[j]
                result = stats.wilcoxon(col1, col2)
                letters_dict[labels_list[j]].append(result.pvalue)
            else:
                letters_dict[labels_list[j]].append(-1)
        letters_dict["letter"].append(labels_list[i])
                
    save_name = f"{phase_1}_analysis/test_{kind}_{scheme_1}_metrics"
    save_results(save_name, letters_dict)


def read_in_dfs_concat(file_path_start, letters, phases, kind, prediction=False):
    df = pd.DataFrame()
    #for filename in os.listdir(file_path):
    if prediction:
        kind = "prediction"
    for phase in phases:
        for letter in letters:
            file_path = file_path_start + "phase_"+str(phase)+"/"+str(phase)+"_"+str(letter)+"main_metrics.csv"
            try:
                if kind == "prediction":
                    sub_df = pd.read_csv(file_path)
                else:
                    sub_df = pd.read_csv(file_path, names=col_names[kind])
                df = df.append(sub_df)
            except Exception as e:
                pass
    return df 

def get_min_per_organization(file_path_start, phases, kind, prediction=False):
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]

    datastream_combo = [*range(1, 4)]

    if prediction:
        use_metric = "f1"
        metric_label = "max_f1"
    else:
        use_metric = "mse"
        metric_label = "min_mse"


    separate_datastreams_all_locations_dict = {"ds_index": [], "model_name": [], "dataset_name": [], metric_label: [], "input_size": [], "output_size": []}
    all_datastreams_all_locations_dict = {"model_name": [], "dataset_name": [], metric_label: [], "input_size": [], "output_size": []}

   
    #Separate datastream
    df_3 = read_in_dfs_concat(file_path_start, separate_datastreams_all_locations, phases, kind, prediction=prediction)
    for ds_index in datastream_combo:
        dict_index = f"{ds_index}"
        df_3_correct = df_3[df_3.ds_combo == ds_index]
        if not df_3_correct.empty:
                if prediction:
                    result_3 = df_3_correct[df_3_correct.f1 == df_3_correct.f1.max()]
                    result_3 = result_3.iloc[0]
                else:
                    result_3 = df_3_correct[df_3_correct.mse == df_3_correct.mse.min()]
                    result_3 = result_3.iloc[0]
                separate_datastreams_all_locations_dict["ds_index"].append(ds_index)
                separate_datastreams_all_locations_dict["model_name"].append(result_3["experiment_name"])
                separate_datastreams_all_locations_dict["dataset_name"].append(result_3["dataset_name"])
                separate_datastreams_all_locations_dict[metric_label].append(result_3[use_metric])
                new_dataset_name = result_3["dataset_name"]
                try:
                    separate_datastreams_all_locations_dict["input_size"].append(result_3["inputs"].item())
                    separate_datastreams_all_locations_dict["output_size"].append(result_3["outputs"].item())
                except Exception as e:
                    print(f"Error getting min per org (3) due to {e}")
                    separate_datastreams_all_locations_dict["input_size"].append(-1)
                    separate_datastreams_all_locations_dict["output_size"].append(-1)

                
    #All together
    df_4 = read_in_dfs_concat(file_path_start, all_datastreams_all_locations, phases, kind, prediction=prediction)

    if prediction:
        result_4 = df_4[df_4.f1 == df_4.f1.max()]
        result_4 = result_4.iloc[0]
    else:
        result_4 = df_4[df_4.mse == df_4.mse.min()]
        result_4 = result_4.iloc[0]
    try:
        #new_dataset_name = result_4["dataset_name"].item()
        new_dataset_name = result_4["dataset_name"]
        input_size = result_4["inputs"].item()
        output_size = result_4["outputs"].item()
    except Exception as e:
        print(f"Error getting min per org (4) due to {e}")
        input_size = -1
        output_size = -1 


    all_datastreams_all_locations_dict = {
                    "model_name": [result_4["experiment_name"]],
                    "dataset_name": [result_4["dataset_name"]],
                    metric_label: [result_4[use_metric]],
                    "input_size": input_size,
                    "output_size": output_size
                }

    ind_phase = phases[0]
    save_names = [f"{ind_phase}_analysis/min_datastream", f"{ind_phase}_analysis/min_all"]
    save_dicts = [separate_datastreams_all_locations_dict, all_datastreams_all_locations_dict]
    
    for i in range(0, len(save_names)):
        #print((save_dicts[i]))
        save_results(save_names[i], save_dicts[i])

    

def get_mean_min_per_organization(letters_dict, phase, file_path_start, kind, prediction=False):
    separate_datastreams_all_locations = ["C", "F", ]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]

    datastream_combo = [*range(1, 4)]

    if prediction:
        use_metric = "f1"
        metric_label = "max_f1"
    else:
        use_metric = "mse"
        metric_label = "min_mse"


    separate_datastreams_all_locations_dict = {"ds_index": [], "model_name": [], "dataset_name": [], metric_label: [], "input_size": [], "output_size": []}
    all_datastreams_all_locations_dict = {"model_name": [], "dataset_name": [], metric_label: [], "input_size": [], "output_size": []}

   
    #Separate datastream
    letter_3 = letters_dict["one_all"]
    #Separate case:
    df_3 = read_in_df_single(letter_3, phase, prediction=prediction)
    for ds_index in datastream_combo:
        dict_index = f"{ds_index}"
        df_3_correct = df_3[df_3.ds_combo == ds_index]
        if not df_3_correct.empty:
                if prediction:
                    result_3 = df_3_correct[df_3_correct.f1 == df_3_correct.f1.max()]
                    result_3 = result_3.iloc[0]
                else:
                    result_3 = df_3_correct[df_3_correct.mse == df_3_correct.mse.min()]
                    result_3 = result_3.iloc[0]
                separate_datastreams_all_locations_dict["ds_index"].append(ds_index)
                
                separate_datastreams_all_locations_dict["model_name"].append(result_3["experiment_name"])
                separate_datastreams_all_locations_dict["dataset_name"].append(result_3["dataset_name"])
                separate_datastreams_all_locations_dict[metric_label].append(result_3[use_metric])
                new_dataset_name = result_3["dataset_name"]
                try:
                    separate_datastreams_all_locations_dict["input_size"].append(result_3["inputs"].item())
                    separate_datastreams_all_locations_dict["output_size"].append(result_3["outputs"].item())
                except Exception as e:
                    print(f"Error getting min per org due to {e}")
                    separate_datastreams_all_locations_dict["input_size"].append(-1)
                    separate_datastreams_all_locations_dict["output_size"].append(-1)

                
    #All together
    letter_4 = letters_dict["all_all"]
    #Separate case:
    df_4 = read_in_df_single(letter_4, phase, prediction=prediction)
    if prediction:
        result_4 = df_4[df_4.f1 == df_4.f1.max()]
        result_4 = result_4.iloc[0]
    else:
        result_4 = df_4[df_4.mse == df_4.mse.min()]
        result_4 = result_4.iloc[0]
    try:
        #new_dataset_name = result_4["dataset_name"].item()
        new_dataset_name = result_4["dataset_name"]
        input_size = result_4["inputs"].item()
        output_size = result_4["outputs"].item()
    except Exception as e:
        print(f"Error getting min per org (4) due to {e}")
        input_size = -1
        output_size = -1 


    all_datastreams_all_locations_dict = {
                    "model_name": [result_4["experiment_name"]],
                    "dataset_name": [result_4["dataset_name"]],
                    metric_label: [result_4[use_metric]],
                    "input_size": input_size,
                    "output_size": output_size
                }

    ind_phase = phase
    save_names = [f"{ind_phase}_analysis/mean_min_datastream", f"{ind_phase}_analysis/mean_min_all"]
    save_dicts = [separate_datastreams_all_locations_dict, all_datastreams_all_locations_dict]
    
    for i in range(0, len(save_names)):
        save_results(save_names[i], save_dicts[i])

    

