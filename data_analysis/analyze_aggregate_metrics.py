#Help from:
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas 
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/ 

import pandas as pd
import matplotlib.pyplot as plt 
import json 
import os 

#Group by Documentation
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html 

# colnames=['TIME', 'X', 'Y', 'Z'] 
# user1 = pd.read_csv('dataset/1.csv', names=colnames, header=None)
#https://www.geeksforgeeks.org/how-to-plot-multiple-data-columns-in-a-dataframe/ 



groups = {
    "ae": ["E", "H", "L", "S", "U", "X", "Z", "AC"],
    "lstm": ["A", "B", "C", "D", "F", "G", "I", "J",
                "M", "N", "Q", "T", "V", "W", "Y", "AA", "AB", "AD",
                ]

}

col_names = {
    "lstm": [
            "version",
            "location_scheme",
            "datastream_scheme",
            "l_combo",
            "ds_combo",
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
            "epochs"
    ],
    "prediction": [
        "version",
            "location_scheme",
            "datastream_scheme",
            "l_combo",
            "ds_combo",
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
            "f1"
    ],
    "ae": [
            "version",
            "location_scheme",
            "datastream_scheme",
            "l_combo",
            "ds_combo",
            "input_days",
            "output_days",
            "loss",
            "mse",
            "dataset_size",
            "training_time",
            "experiment_name",
            "dataset_name",
            "epochs"
        ]
}



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

    "lstm_predict": {
        "letter": [],
        "phase_letter": [],
        "mean_f1": [],
        "min_f1": [],
        "max_f1": [],
        "stdev_f1": [],
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
    }
}




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
                else:
                    cols = col_names[ingroup]

                sub_dict["df"] = pd.read_csv(f, names=cols)
                df_dict[letter] = sub_dict

    return df_dict 


def get_metric(df_dict, metric):
    df = df_dict["df"]
    return_metric=None
    if metric == "letter": 
        return_metric = df_dict["letter"]
    elif metric == "phase_letter":
        return_metric = df_dict["phase_letter"]
    elif metric == "mean_mse":
        return_metric = round(df["mse"].mean(), 5)
    elif metric == "min_mse":
        return_metric = round(df["mse"].min(), 5)
    elif metric == "max_mse":
        return_metric = round(df["mse"].max(), 5)
    elif metric == "stdev_mse":
        return_metric = round(df["mse"].std(), 5)
    elif metric == "mean_f1":
        return_metric = round(df["f1"].mean(), 5)
    elif metric == "min_f1":
        return_metric = round(df["f1"].min(), 5)
    elif metric == "max_f1":
        return_metric = round(df["f1"].max(), 5)
    elif metric == "stdev_f1":
        return_metric = round(df["f1"].std(), 5)
    elif metric == "mean_mape":
        return_metric = round(df["mape"].mean(), 5)
    elif metric == "min_mape":
        return_metric = round(df["mape"].min(), 5)
    elif metric == "max_mape":
        return_metric = round(df["mape"].max(), 5)
    elif metric == "stdev_mape":
        return_metric = round(df["mse"].std(), 5)
    elif metric == "mean_mae":
        return_metric = round(df["mae"].mean(), 5)
    elif metric == "min_mae":
        return_metric = round(df["mae"].min(), 5)
    elif metric == "max_mae":
        return_metric = round(df["mae"].max(), 5)
    elif metric == "stdev_mae":
        return_metric = round(df["mae"].std(), 5)
    elif metric == "mean_training_time":
        return_metric = round(df["training_time"].mean(), 5)
    elif metric == "mean_num_epochs":
        return_metric = round(df["epochs"].mean(), 5)
    elif metric == "location_scheme":
        return_metric = df["location_scheme"].min()
    elif metric == "datastream_scheme":
        return_metric = df["datastream_scheme"].min()
    elif metric ==  "num_experiments":
        return_metric = len(df.index)
    return return_metric



def calc_aggregate_metrics(df_dict, scheme, prediction=False):
    if prediction:
        scheme = "lstm_predict"
    metrics_dict = aggregate_metrics[scheme].copy()
    for letter in list(df_dict.keys()):
        #print(letter)
        letter_dict = df_dict[letter]
        for metric in list(metrics_dict.keys()):
            metrics_dict[metric].append(get_metric(letter_dict, metric))
    return metrics_dict


def save_results(filename, save_dict):
    df = pd.DataFrame(save_dict)
    df.to_csv(filename, index=False)


def save_metrics(phase, scheme, file_path, prediction=False):
    df_dict = read_in_dfs(file_path, phase, ingroup=scheme, prediction=prediction)
    #test_df = df_dict["AF"]["df"]
    #print(test_df.head())
    metrics_dict = calc_aggregate_metrics(df_dict, scheme, prediction=prediction)
    dir_name = f"{phase}_analysis/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_name = f"{dir_name}{phase}_{scheme}_aggregate_metrics"
    
    #print(metrics_dict)
    save_results(save_name, metrics_dict)



# # def get_agg_for_both(file_path, phase, scheme):
# #     df_dict = read_in_dfs(file_path, phase, df_dict, ingroup="lstm")
# #     metrics_dict = calc_aggregate_metrics(df_dict, scheme)
# #     save_results(filename, save_dict)


# phase = "4" 
# file_path = f'main_metrics/phase_{phase}/'
# scheme = "lstm"
# #scheme = "ae"
# #Read 'em in 

# #save_metrics(phase, scheme, file_path)

