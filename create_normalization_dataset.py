import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import json 


#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas 

#For predictions, since this is "new" data, we are going to have to unnormalize anyway 

cases = {
    4481: {"index": 0},
    3719: {"index": 1},
    1292: {"index": 2},
    397: {"index": 3},
    2327:{"index": 4},
    5018: {"index": 5},
    6009: {"index": 6},
    1820: {"index": 7},
    2332: {"index": 8},
    4255: {"index": 9},
    1191: {"index": 10},
    1959: {"index": 11},
    553: {"index": 12},
    3631: {"index": 13},
    2738: {"index": 14},
    818: {"index": 15},
    1590: {"index": 16},
    55: {"index": 17},
    5175: {"index": 18},
    4283: {"index": 19},
    5693: {"index": 20},
    1730: {"index": 21},
    5442: {"index": 22},
    3524: {"index": 23},
    4684: {"index": 24},
    5837: {"index": 25},
    1231: {"index": 26},
    6227: {"index": 27},
    985: {"index": 28},
    3930: {"index": 29},
    2267: {"index": 30},
    4573: {"index": 31},
    5983: {"index": 32},
    2272: {"index": 33},
    6246: {"index": 34},
    5607: {"index": 35},
    1900: {"index": 36},
    3694: {"index": 37},
    2168: {"index": 38},
    1785: {"index": 39},
    1018: {"index": 40},
    251: {"index": 41}
}

valid_cases = list(cases.keys())
list_1 = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
list_2 = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
list_3 = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
list_4 = ["anestart", "aneend", "age", "sex", "height", "weight",
                    "bmi", "dx", "dis", "preop_pft", "preop_plt", "preop_pt", 
                    "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_cr", 
                    "intraop_uo", "intraop_ffp"]
predictions = ["emop", "dis_mortality_risk", "gluc_risk"]
all_features_list = list_1 + list_2 + list_3 + list_4 + predictions


#Cols - feature_name, max, min 
norm_df = {
    "feature_name": [],
    "max": [],
    "min": []
}


def create_normalization_csv():
    all_features_max_and_mins = {}

    for item in all_features_list:
        all_features_max_and_mins[item] = {"max": [], "min": []}

    #Find max and min of each 
    for dataset in valid_cases:
        #Load it in
        case = pd.read_csv(f"vital_csvs/{dataset}.csv")
        #Check that it has the right fields
        case_cols = list(case.columns)

        for col in case_cols:
            if col in all_features_list:
                all_features_max_and_mins[col]["max"].append(case[col].max())
                all_features_max_and_mins[col]["min"].append(case[col].min())

    #
    for feature in list(all_features_max_and_mins.keys()):
        print(feature)

        norm_df["feature_name"].append(feature)
        norm_df["max"].append(max(all_features_max_and_mins[feature]["max"]))
        norm_df["min"].append(min(all_features_max_and_mins[feature]["min"]))
        
    #print(json.dumps(norm_df, indent=4))
    df = pd.DataFrame(norm_df)
    df.to_csv("vital_csvs/normalization_info.csv")

def normalize_csvs():
    normalization_info = pd.read_csv("vital_csvs/normalization_info.csv")
    for dataset in valid_cases:
        case = pd.read_csv(f"vital_csvs/{dataset}.csv")
        case_cols = list(case.columns)
        for col in case_cols:
            if col in all_features_list:
                relevant_ds = normalization_info.loc[normalization_info["feature_name"] == col]
                max_val = relevant_ds["max"].item()
                min_val = relevant_ds["min"].item()
                diff_between = max_val - min_val
                if diff_between != 0:
                    case[col] = (case[col] - min_val)/(diff_between)
        case.to_csv(f"vital_csvs/{dataset}_normalized.csv")
#Think we should load in all normalized already to save some time! 
#Just save as case_name_normalzied 


#create_normalization_csv()
#normalize_csvs()