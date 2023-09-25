#Help from:
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas 
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/ 
#https://stackoverflow.com/questions/12572362/how-to-get-a-string-after-a-specific-substring 

import pandas as pd
import matplotlib.pyplot as plt 
import json 
import os 
import scipy.stats as stats 
import seaborn as sn
import math 

#Group by Documentation
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html 

# colnames=['TIME', 'X', 'Y', 'Z'] 
# user1 = pd.read_csv('dataset/1.csv', names=colnames, header=None)
#https://www.geeksforgeeks.org/how-to-plot-multiple-data-columns-in-a-dataframe/ 



groups = {
    "ae": ["E", "H", "L", "S", "U", "X", "Z", "AC"],
    "lstm": ["A", "B", "C", "D", "F", "G", "I", "J",
                "M", "N", "Q", "T", "V", "W", "Y", "AA", "AB", "AD"]

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
    ]
}



all_letters = ['A', 'B', 'C', 'D', 'F', 'G', 'I', 'J', 'M', 'N', 'Q', 'T', 'V', 'W', 'Y', 'AA', 'AB', 'AD']

network_1_letters = ["A", "B", "C", "D"]
network_2_letters = ["E", "F", "G", "H", "I", "J", "L", "M", "N", "Q", "AF", "AG", "AJ"]
network_3_letters = ["S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB"]
network_4_letters = ["E", "F", "L", "M", "AC", "AD", "AF"]

separate_letters = ['D', 'I']
separate_datastreams_all_locations = ["C", "F", "Q", "AA", "AI"]
all_datastreams_separate_locations = ["B", "J", "M", "V", "AF"]
all_datastreams_all_locations = ["A", "G", "N", "T", "W", "Y", "AB", "AD", "AG", "AJ"]




def get_network(letter):
    if letter in network_1_letters:
        return 1
    elif letter in network_2_letters:
        return 2
    elif letter in network_3_letters:
        return 3
    elif letter in network_4_letters:
        return 4
    return 0

def get_separation_scheme(letter):
    if letter in separate_letters:
        return 1
    elif letter in separate_datastreams_all_locations:
        return 2
    elif letter in all_datastreams_separate_locations:
        return 3
    elif letter in all_datastreams_all_locations:
        return 4
    return 0

def get_model_params(phase, letter, df_row):
    if len(df_row.index) == 1:
        input_days = df_row['input_days'].item()
        output_days = df_row['output_days'].item()
        exp_name = df_row["experiment_name"].item()
    else:
        input_days = df_row['input_days'].iloc[0]
        output_days = df_row['output_days'].iloc[0]
        exp_name = df_row["experiment_name"].values.tolist()
    if isinstance(exp_name, list):
        base_network = []
        for item in exp_name:
            base_network.append(item.split("_")[2])
    else:
        base_network = exp_name.split("_")[2]
    return input_days, output_days, base_network
    #print(my_string.split("world",1)[1])

def per_slate_chart(phase):
    #Letters
    #separation_schemes = [separate_letters, separate_datastreams_all_locations, all_datastreams_separate_locations, all_datastreams_all_locations]
    separation_schemes = [separate_datastreams_all_locations, all_datastreams_all_locations]
    folder = f"{phase}_analysis/"
    slate_metrics = f"{folder}slate_metrics.csv"
    chart_dict = {
        "Slate": [],
        "Model Letter": [],
        "Network": [],
        "Separation_Scheme": [],
        "Mean Metric": [],
        "Best Metric": [],
        "Best Input Days": [],
        "Best Output Days": [],
        "Best Base": []
    }
    slate_metrics_df = pd.read_csv(slate_metrics)
    for letter in all_letters:
        #try:
            #Sub df:
            #df.loc[df['col1'] == value]
            if phase in prediction_slates:
                sub_df = slate_metrics_df[(slate_metrics_df["letter"]==letter) & (slate_metrics_df["min_or_max"]=="Max")]
            else:
                sub_df = slate_metrics_df[(slate_metrics_df["letter"]==letter) & (slate_metrics_df["min_or_max"]=="Min")]
            #whole_df[(whole_df['input_days']==input_var) & (whole_df['output_days']==output_var) & (whole_df['experiment_name']==exp_name)]
            if not sub_df.empty:
                chart_dict["Slate"].append(phase)
                chart_dict["Model Letter"].append(letter)
                chart_dict["Network"].append(get_network(letter))
                chart_dict["Separation_Scheme"].append(get_separation_scheme(letter))
                #Doesn't matter that using max - it will be same 
                chart_dict["Mean Metric"].append(sub_df["mean_metric"].max())
                chart_dict["Best Metric"].append(sub_df["weighted_metric"].max())
                input_days, output_days, base = get_model_params(phase, letter, sub_df)
                chart_dict["Best Input Days"].append(input_days)
                chart_dict["Best Output Days"].append(output_days)
                chart_dict["Best Base"].append(base)
        #except Exception as e:
        #    print(f"Failed for letter {letter} for reason {e}")

    save_path = f"{phase}_analysis/per_slate_chart.csv"
    final_df = pd.DataFrame(chart_dict)
    final_df.to_csv(save_path)


#prediction_slates = ["2", "3", "4", "5", "7", "9", "11", "12", "13", "14", "16", "18"]
#phase = '18'
#per_slate_chart(phase)

