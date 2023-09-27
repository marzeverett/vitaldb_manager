#Help from:
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas 
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/ 

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


#Calculated the weighted metric 
def calc_weighted_metric(df, total_outputs, prediction=False):
    #print(df["dataset_name"])
    #And the mse.
    if prediction:
        metric = "f1"
    else:
        metric = "mse"

    new_dataset_name = df["dataset_name"].item()
    #print(new_dataset_name)
    #print(i_o_csv["output_size"])
    output_size = df["outputs"].item()
    value_metric = df[metric].item()
    weighting = output_size/total_outputs
    # print("Value Metric", value_metric)
    #print("Weighting", weighting)
    if not prediction:
        weighted_metric = value_metric*weighting
    else:
        weighted_metric = value_metric
    return weighted_metric

#This is ONE variation. 
def load_and_weight(phase, letter, curr_sep_dict, curr_sep_kind, input_var, output_var, exp_var, total_outputs, prediction=False):
    try:
        datastream_combo = [*range(1, 4)]
        exp_name = f"{phase}_{letter}_{exp_var}"
        #Try to load in the whole df for this phase and letter 
        metrics_path = f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv"
        if prediction:
            cols = col_names["prediction"]
            whole_df = pd.read_csv(metrics_path)
        else:
            cols = col_names["lstm"]
            whole_df = pd.read_csv(metrics_path, names=cols)

        #Restrict the df to our particular variation
        weighted_mse = 0
        df_restrict = whole_df[(whole_df['input_days']==input_var) & (whole_df['output_days']==output_var) & (whole_df['experiment_name']==exp_name)]
        if curr_sep_kind == "all_all":
            if not df_restrict.empty:
                weighted_mse = calc_weighted_metric(df_restrict,  total_outputs, prediction=prediction)
        elif curr_sep_kind == "one_all":
            weighted_sum = 0 
            num_datastreams = 0
            for ds_index in datastream_combo:
                new_df = df_restrict[df_restrict['ds_combo']==ds_index]
                if not new_df.empty:
                    weighted_sum = weighted_sum + calc_weighted_metric(new_df, total_outputs, prediction=prediction)
                    num_datastreams += 1
            weighted_mse = weighted_sum
            if prediction:
                weighted_mse = weighted_mse/num_datastreams

        #Add info to dict
        curr_sep_dict["letter"].append(letter)
        curr_sep_dict["input_days"].append(input_var)
        curr_sep_dict["output_days"].append(output_var)
        curr_sep_dict["experiment_name"].append(exp_name)
        curr_sep_dict["weighted_metric"].append(weighted_mse)
    except Exception as e:
      print(f"Issue with letter {letter} {input_var} {output_var} {exp_var}: {e}")

    

def get_all_weighted_variations(phase, letter, curr_sep_kind, total_outputs, prediction=False):
    variation_dict = {
        "letter": [],
        "input_days": [],
        "output_days": [],
        "experiment_name": [], 
        "weighted_metric": []
    }
    #CHANGE IS HERE!! - NOTE: Will change based on hyperparameters
    input_days = [300]
    output_days = [10]
    scaling_factors = [8, 32, 64]
    
    for input_var in input_days:
        for output_var in output_days:
            for exp_var in scaling_factors:
                load_and_weight(phase, letter, variation_dict, curr_sep_kind, input_var, output_var, exp_var, total_outputs, prediction=prediction)
    
    #Then save 
    save_folder = f"{phase}_analysis/combo_models/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_name = f"{letter}_combos_weighted.csv"

    save_path = save_folder+save_name

    df = pd.DataFrame(variation_dict)
    df.to_csv(save_path, index=False)


def get_best_weighted_model_per_organization(phase, total_outputs, prediction=False):
    #Letters
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]


    #Alltogether now 
    #separation_schemes = [separate_letters, separate_datastreams_all_locations, all_datastreams_separate_locations, all_datastreams_all_locations]
    separation_schemes = [all_datastreams_all_locations, separate_datastreams_all_locations]
    #separation_kinds = ["one_one", "one_all", "all_one", "all_all"]
    separation_kinds = ["all_all", "one_all"]

    #For each separation scheme: 
    for i in range(0, len(separation_schemes)):
        curr_sep_scheme = separation_schemes[i]
        curr_sep_kind = separation_kinds[i]
        for letter in curr_sep_scheme:
            get_all_weighted_variations(phase, letter, curr_sep_kind, total_outputs, prediction=prediction)

    

def get_best_weighted_model_per_slate_per_scheme(phase, prediction=False):
    #Letters
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]
    separation_schemes = [all_datastreams_all_locations, separate_datastreams_all_locations]
    final_df = pd.DataFrame()
    if prediction:
        metric = "weighted_metric"
    else:
        metric = "weighted_metric"
    for scheme_letters in separation_schemes:
        scheme_df = pd.DataFrame()
        scheme_min = None
        min_val = None
        min_row = pd.DataFrame()
        for letter in scheme_letters:
            #Try to load it in 
            try:
                #print(letter)
                df_path = f"{phase}_analysis/combo_models/{letter}_combos_weighted.csv"
                df = pd.read_csv(df_path)
                if prediction:
                    df_row = df[df[metric] == df[metric].max()]
                else:
                    df_row = df[df[metric] == df[metric].min()]
                #print(df_row.empty)
                if min_row.empty:
                    min_row = df_row
                    if prediction:
                        min_val = df_row[metric].max()
                    else:
                        min_val = df_row[metric].min()
                    #print(min_val)
                else:
                    if prediction:
                        curr_min_val = df_row[metric].max()
                    else:
                        curr_min_val = df_row[metric].min()
                    #print(curr_min_val)
                    if prediction:
                        if curr_min_val > min_val:
                            min_val = curr_min_val
                            min_row = df_row
                    else:
                        if curr_min_val < min_val:
                            min_val = curr_min_val
                            min_row = df_row
            except Exception as e:
                print(f"Could not load {phase} {letter} because {e} weighted slate scheme")
        if scheme_df.empty:
            scheme_df = min_row
            scheme_min = min_val
        else:
            if prediction:
                if min_val > scheme_min:
                    scheme_df = min_row
                    scheme_min = min_val
            else:
                if min_val < scheme_min:
                    scheme_df = min_row
                    scheme_min = min_val
        if final_df.empty:
            final_df = scheme_df
        else:
            final_df = pd.concat([final_df, scheme_df])
        #print(final_df.head())

    
    save_path = f"{phase}_analysis/overall_weighted_models.csv"
    final_df.to_csv(save_path)

    #For each separation scheme
    


#Get lowest or highest mean per scheme
def get_best_weighted_mean_per_scheme(phase, prediction=False):
    #Letters
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]
    separation_schemes = [separate_datastreams_all_locations, all_datastreams_all_locations]

    final_dict = {"letters": [], "metric": []}
    if prediction:
        metric = "weighted_metric"
    else:
        metric = "weighted_metric"

    for scheme_letters in separation_schemes:
        scheme_letter = None
        scheme_min = None
        min_val = None
        min_letter = None
        for letter in scheme_letters:
            #Try to load it in 
            try:
                #print(letter)
                df_path = f"{phase}_analysis/combo_models/{letter}_combos_weighted.csv"
                df = pd.read_csv(df_path)
                df_mean = df[metric].mean()
                #print(letter)
                if not math.isnan(df_mean):
                    #print("Df mean ", df_mean)
                    #print(df_row.empty)
                    if min_letter == None:
                        min_letter = letter
                        min_val = df_mean
                        #print(min_val)
                    else:
                        curr_min_val = df_mean
                        #print(curr_min_val)
                        if prediction:
                            if curr_min_val > min_val:
                                min_val = curr_min_val
                                min_letter = letter
                        else:
                            if curr_min_val < min_val:
                                min_val = curr_min_val
                                min_letter = letter
            except Exception as e:
                print(f"Could not load {phase} {letter} because {e} weighted mean")
        if scheme_letter == None:
            scheme_letter = min_letter
            scheme_min = min_val
        else:
            if prediction:
                if min_val > scheme_min:
                    scheme_letter = min_letter
                    scheme_min = min_val
            else:
                if min_val < scheme_min:
                    scheme_letter = min_letter
                    scheme_min = min_val

        final_dict["letters"].append(scheme_letter)
        final_dict["metric"].append(scheme_min)
        #print(scheme_letter)
        #print(scheme_min)
       
        #print(final_df.head())
    final_df = pd.DataFrame(final_dict)
    save_path = f"{phase}_analysis/mean_overall_weighted_models.csv"
    final_df.to_csv(save_path)

    #For each separation scheme




def get_more_useful_slate_info(phase, prediction=False):
    #Letters
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]
    separation_schemes = [separate_datastreams_all_locations, all_datastreams_all_locations]
    final_df = pd.DataFrame()
    if prediction:
        metric = "weighted_metric"
    else:
        metric = "weighted_metric"

    for scheme_letters in separation_schemes:
        min_val = None
        min_row = pd.DataFrame()
        scheme_letter_dict = {"letter": [], "mean metric base 8": [], "mean metric base 32": [], "mean metric base 64": []}
        for letter in scheme_letters:
            #Try to load it in 
            try:
                #print(letter)
                df_path = f"{phase}_analysis/combo_models/{letter}_combos_weighted.csv"
                df = pd.read_csv(df_path)
                df_row_min = df[df[metric] == df[metric].min()]
                df_row_max = df[df[metric] == df[metric].max()]
                df_mean = df[metric].mean()
                if not math.isnan(df_mean):
                    df_row_min = df_row_min.assign(mean_metric=df_mean)
                    df_row_max = df_row_max.assign(mean_metric=df_mean)
                    df_row_min = df_row_min.assign(min_or_max="Min")
                    df_row_max = df_row_max.assign(min_or_max="Max")
                    add_rows = pd.concat([df_row_min, df_row_max])
                    if final_df.empty:
                        final_df = add_rows
                    else:
                        final_df = pd.concat([final_df, add_rows])  
            except Exception as e:
                print(f"Could not load {phase} {letter} because {e} slate info")
       
        #print(final_df.head())

    save_path = f"{phase}_analysis/slate_metrics.csv"
    final_df.to_csv(save_path)

    #For each separation scheme
    

def get_model_arch_comparison(phase, prediction=False):
    #Letters
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]
    separation_schemes = [separate_datastreams_all_locations, all_datastreams_all_locations]
    if prediction:
        metric = "f1"
    else:
        metric = "mse"
    scheme_letter_dict = {"letter": [], "mean base 8": [], "mean base 32": [], "mean base 64": []}
    for scheme_letters in separation_schemes:
        min_val = None
        min_row = pd.DataFrame()
        for letter in scheme_letters:
            #Try to load it in 
            try:
                #print(letter)
                df_path = f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv"
                if prediction:
                    cols = col_names["prediction"]
                    df = pd.read_csv(df_path)
                else:
                    cols = col_names["lstm"]
                    df = pd.read_csv(df_path, names=cols)
                exp_8_name = f"{phase}_{letter}_8"
                exp_32_name = f"{phase}_{letter}_32"
                exp_64_name = f"{phase}_{letter}_64"

                #Group by experiment name and get the mean metric 
                df_8 = df[df["experiment_name"] == exp_8_name]
                df_8_mean = df_8[metric].mean()

                df_32 = df[df["experiment_name"] == exp_32_name]
                df_32_mean = df_32[metric].mean()

                df_64 = df[df["experiment_name"] == exp_64_name]
                df_64_mean = df_64[metric].mean()

                if not math.isnan(df_8_mean):
                    scheme_letter_dict["letter"].append(letter)
                    scheme_letter_dict["mean base 8"].append(df_8_mean)
                    scheme_letter_dict["mean base 32"].append(df_32_mean)
                    scheme_letter_dict["mean base 64"].append(df_64_mean)

            except Exception as e:
                print(f"Could not load {phase} {letter} because {e} model arch")
       
    final_df = pd.DataFrame(scheme_letter_dict)
    save_path = f"{phase}_analysis/compare_by_nodes.csv"
    final_df.to_csv(save_path)



def compare_stdev(phase, prediction=False):
    #Letters
    separate_datastreams_all_locations = ["C", "F"]
    all_datastreams_all_locations = ["A", "G", "T", "AD"]
    separation_schemes = [separate_datastreams_all_locations, all_datastreams_all_locations]
    if prediction:
        metric = "f1"
    else:
        metric = "mse"
    scheme_letter_dict = {"letter": [], "stdev base 8": [], "stdev base 32": [], "stdev base 64": []}
    for scheme_letters in separation_schemes:
        min_val = None
        min_row = pd.DataFrame()
        for letter in scheme_letters:
            #Try to load it in 
            try:
                #print(letter)
                df_path = f"main_metrics/phase_{phase}/{phase}_{letter}main_metrics.csv"
                if prediction:
                    cols = col_names["prediction"]
                    df = pd.read_csv(df_path)
                else:
                    cols = col_names["lstm"]
                    df = pd.read_csv(df_path, names=cols)
                exp_8_name = f"{phase}_{letter}_8"
                exp_32_name = f"{phase}_{letter}_32"
                exp_64_name = f"{phase}_{letter}_64"

                #Group by experiment name and get the mean metric 
                df_8 = df[df["experiment_name"] == exp_8_name]
                df_8_mean = df_8[metric].std()

                df_32 = df[df["experiment_name"] == exp_32_name]
                df_32_mean = df_32[metric].std()

                df_64 = df[df["experiment_name"] == exp_64_name]
                df_64_mean = df_64[metric].std()

                if not math.isnan(df_8_mean):
                    scheme_letter_dict["letter"].append(letter)
                    scheme_letter_dict["stdev base 8"].append(df_8_mean)
                    scheme_letter_dict["stdev base 32"].append(df_32_mean)
                    scheme_letter_dict["stdev base 64"].append(df_64_mean)

            except Exception as e:
                print(f"Could not load {phase} {letter} because {e} stdev")

       
    final_df = pd.DataFrame(scheme_letter_dict)
    save_path = f"{phase}_analysis/compare_by_stdev.csv"
    final_df.to_csv(save_path)


#NOTE: Need to change some hyperparams here 

# phase = "28"
# total_outputs = 3
# #total_outputs = 18 
# prediction = True
# get_best_weighted_model_per_organization(phase, total_outputs, prediction=prediction)
# get_best_weighted_mean_per_scheme(phase, prediction=prediction)
# get_best_weighted_model_per_slate_per_scheme(phase, prediction=prediction)
# get_more_useful_slate_info(phase, prediction=prediction)
# get_model_arch_comparison(phase, prediction=prediction)
# #compare_stdev(phase, prediction=prediction)

# # #prediction_slates = ["2", "3", "4", "5", "7", "9", "11", "12", "13", "14", "16", "18"]
