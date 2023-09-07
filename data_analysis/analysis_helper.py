#Help from:
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas 
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/ 
#https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-parent-directories
#

import pandas as pd
import matplotlib.pyplot as plt 
import json 
import os 
import seaborn as sn


import analyze_aggregate_metrics
import analyze_individual_metrics
import analyze_separation_schemes 


groups = {
    "ae": ["E", "S", "AC"],
    "lstm": ["A","C", "F", "G", "T", "AD"]

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
        "version",
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
    }
}

prediction_slates = ["2", "4", "6", "8"]

network_1_letters = ["A", "C"]
network_2_letters = ["E", "F", "G",]
network_3_letters = ["S", "T"]
network_4_letters = ["AC", "AD"]

networks_list = [network_1_letters, network_2_letters, network_3_letters, network_4_letters]


one_all_letters = ["C", "E", "F"]
all_all_letters = ["A", "G", "S", "T", "AC", "AD"]

separation_scheme_list = [one_all_letters, all_all_letters]
separation_scheme_kinds = ["one_all", "all_all" ]



def minimum_comparison_models(phase, prediction=False):
    file_path_start = f"main_metrics/"
    #Have kinda a funky way of doing this. But I think it'll work. 
    analyze_separation_schemes.get_min_per_organization(file_path_start, [phase], "lstm", prediction=prediction)

def aggregate_metrics(phase, prediction=False):
    file_path = f'main_metrics/phase_{phase}/'
    scheme = "lstm"
    analyze_aggregate_metrics.save_metrics(phase, scheme, file_path, prediction=prediction)

def get_correct_letters(sep_scheme, group):
    correct_letters = []
    for letter in sep_scheme:
        if letter in groups[group]:
            correct_letters.append(letter)
    return correct_letters

def test_and_table(sep_kind, correct_letters, phase, prediction=False):
        #Get the table
        file_path_1 = f"main_metrics/phase_{phase}/"
        analyze_separation_schemes.table_letters(sep_kind, correct_letters, file_path_1, phase, "lstm", prediction=prediction)
        #Get the test 
        analyze_separation_schemes.test_letters(sep_kind, correct_letters, file_path_1, phase, "lstm", prediction=prediction)


def get_min_models_per_network_type(sep_kind, correct_letters, phase, prediction=False):
    net_1 = []
    net_2 = []
    net_3 = []
    net_4 = []
    #Get the letters sorted into networks
    for letter in correct_letters:
        if letter in network_1_letters:
            net_1.append(letter)
        if letter in network_2_letters:
            net_2.append(letter)
        if letter in network_3_letters:
            net_3.append(letter)
        if letter in network_4_letters:
            net_4.append(letter)
    all_networks_list = [net_1, net_2, net_3, net_4]
    all_networks_labels = ['Network 1', "Network 2", "Network 3", "Network 4"]
    min_networks_df = pd.DataFrame()
    #For each network
    for i in range(0, len(all_networks_list)):
        sub_network = all_networks_list[i]
        sub_label = all_networks_labels[i]
        min_mse = pd.DataFrame()
        min_letter = None
        max_mse = pd.DataFrame()
        max_letter = None
        #For each letter in that network 
        for sub_letter in sub_network:
            try:
                #load in the letter df.
                if prediction:
                    df = pd.read_csv(f"main_metrics/phase_{phase}/{phase}_{sub_letter}main_metrics.csv", names=col_names["prediction"])
                    #print(sub_letter)
                    #print(df.columns)
                    min_df = df[df.f1 == df.f1.min()]
                    max_df = df[df.f1 == df.f1.max()]
                    main_metric = "f1"
                else:
                    df = pd.read_csv(f"main_metrics/phase_{phase}/{phase}_{sub_letter}main_metrics.csv", names=col_names["lstm"])
                    min_df = df[df.mse == df.mse.min()]
                    max_df = df[df.mse == df.mse.max()]
                    main_metric = "mse"
                if min_mse.empty:
                    min_mse = min_df
                    min_letter = sub_letter
                else:
                    if min_df[main_metric].item() < min_mse[main_metric].item():
                        min_mse = min_df
                        min_letter = sub_letter
                if max_mse.empty:
                    max_mse = max_df
                    max_letter = sub_letter
                else:
                    if max_df[main_metric].item() > max_mse[main_metric].item():
                        max_mse = max_df
                        max_letter = sub_letter
            except Exception as e:
                print(f"No letter present {sub_letter} {e}")
        #At the end, will have a min and max row value for a network
        min_mse = min_mse.assign(network=sub_label)
        max_mse = max_mse.assign(network=sub_label)
        min_mse= min_mse.assign(letter=min_letter)
        max_mse = max_mse.assign(letter=max_letter)
        min_mse = min_mse.assign(Max_or_Min="Min")
        max_mse = max_mse.assign(Max_or_Min="Max")
        if min_networks_df.empty:
            min_networks_df = min_mse
            min_networks_df = pd.concat([min_networks_df, max_mse], axis=0)
        else:
            min_networks_df = pd.concat([min_networks_df, min_mse], axis=0)
            min_networks_df = pd.concat([min_networks_df, max_mse], axis=0)
    #Finally, save the csv
    save_name = f"{phase}_analysis/network_outliers_{sep_kind}.csv"
    min_networks_df.to_csv(save_name)
        



#This is a B thing - may want to do it manually? 
#I think we very much do want to do this manually
#It creates the graphs for that letter
def get_letter_graphs(phase, letter, prediction=False):
    graphing_vars = ["location_scheme", "datastream_scheme", "l_combo", "ds_combo", "input_days", "output_days", "dataset_size", "dataset_name", "experiment_name"]
    #only works for continuous vars
    if prediction:
        correlation_vars =["f1", "input_days", "output_days", "dataset_size", "training_time", "epochs"]
        df = pd.read_csv(f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv", names=col_names["prediction"])
        graph_against = "f1"
    else:
        correlation_vars = ["mse", "input_days", "output_days", "dataset_size", "training_time", "epochs"]
        #Load in the df
        df = pd.read_csv(f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv", names=col_names["lstm"])
        graph_against = "mse"
    graph_path = f"{phase}_analysis/graphs/{letter}/"
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    #Make the graphs 
    for graph_var in graphing_vars:
        #graph_df = df.groupby([graph_var]).mean()
        graph_df = df.groupby([graph_var]).mean()
        graph_df.plot(kind="bar", y=graph_against)
        plt.title(f"{graph_against} by {graph_var} for phase {phase} {letter}")
        save_name = graph_path+f"{phase} {letter} {graph_var}_{graph_against}.jpg"
        plt.savefig(save_name)
        plt.clf()

    #Make the correlation matrix
    corr_df = df[correlation_vars]
    corr_matrix = corr_df.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.title(f"Correlation Matrix for phase {phase} {letter}")
    plt.xticks(rotation=50)
    plt.yticks(rotation=50)
    save_name = graph_path+f"correlation_matrix.jpg"
    plt.savefig(save_name)
    plt.clf()



def run_basic_analysis(phases):
    #For each slate of experiments 
    for phase in phases:
        if phase in prediction_slates:
            prediction = True
        else:
            prediction = False
        #First, get the aggregate metrics on the whole thing 
        try:
            aggregate_metrics(phase, prediction=prediction)
        except Exception as e:
             print(f"Couldn't get aggregate metrics for reason {e}")
        #We also want C 
        try:
            minimum_comparison_models(phase, prediction=prediction)
        except Exception as e:
            print(f"Couldn't get minimum comparison models for reason {e}")
        #Then get the metrics per separation scheme:
        #Will need to make sure this saves correctly 
        for i in range(0, len(separation_scheme_list)):
            #Get the separation scheme letters 
            sep_scheme = separation_scheme_list[i]
            #Get the separation kind name
            sep_kind = separation_scheme_kinds[i]
            #Limit to only LSTM letters 
            correct_letters = get_correct_letters(sep_scheme, "lstm")
            #Test and Table (A I and II)
            try:
                test_and_table(sep_kind, correct_letters, phase, prediction=prediction)
            except Exception as e:
                print(f"Couldn't test and table for reason {e}")
            #Now for B -- We will need to break it up by Network Type
            try:
                get_min_models_per_network_type(sep_kind, correct_letters, phase, prediction=prediction)
            except Exception as e:
                print(f"Couldn't get min models per network type for reason {e}")


def adjust_prediction_models(phases):
    read_in = [
            "version",
            "location_scheme",
            "datastream_scheme",
            "l_combo",
            "ds_combo",
            "input_days",
            "output_days",
            "loss",
            "mse",
            "f1",
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
            "outputs"
    ]
    for phase in phases:
        for letter in groups["lstm"]:
            try:
                df = pd.read_csv(f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv", names=read_in)
                df = df.assign(f1=(2*df["precision"]*df["recall"])/(df["precision"]+df["recall"]))
                #Save it back 
                df.to_csv(f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv")
            except Exception as e:
                print(f"No metrics for {phase} {letter}")


#ADJUSTMENT - DO FIRST 
#DONT ACCIDENTALLY DO ON A NON-PREDICTION MODEL!
# phase = ["16"]
# adjust_prediction_models(phase)



# # # # # #But we need to find a per-separation scheme, per-network ad-hoc analysis 
phases = ["5"]
run_basic_analysis(phases)

# # # ##### For inspecting individual graphs! 
# phase = "14"
# letter = "T"
# prediction = True
# get_letter_graphs(phase, letter, prediction=prediction)

