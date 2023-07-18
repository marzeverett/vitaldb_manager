import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 

#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 






snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
bis = ["BIS/BIS", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW"]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
#43 of these 
valid_cases = [4481, 3719, 1292, 397, 2327, 6297, 5018, 6009, 1820, 2332, 4255, 1191, 1959, 553, 3631, 2738, 818, 1590, 55, 5175, 4283, 5693, 1730, 5442, 3524, 4684, 5837, 1231, 6227, 985, 3930, 2267, 4573, 5983, 2272, 6246, 5607, 1900, 3694, 2168, 1785, 1018, 251]
total_tracks = snu+bis+orch+solar

track_dict = {
    "snu": snu,
    "orch": orch,
    "bis": bis,
    "solar": solar
}


df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # patient information


def trim_df(df, columns):
    start_indexes = []
    end_indexes = []
    for column in columns:
        #find first valid index and append it 
        first = df[column].first_valid_index()
        start_indexes.append(first)
        #find last valid index and append it 
        last = df[column].last_valid_index()
        end_indexes.append(last)
    #Take max start index
    max_val = max(start_indexes)
    min_val = min(end_indexes)
    #Take min end index
    print(max_val, min_val)
    df = df.iloc[max_val:min_val]
    return df 

def fill_gaps(df):
    df = df.fillna(method="ffill")
    return df 


def add_static_info(case_id, df):
    df_cases = pd.read_csv("vital_csvs/clinical_info.csv")
    df_case = df_cases.loc[df_cases["caseid"] == case_id]
    case_columns = df_case.columns.values.tolist()
    for col in case_columns:
        df[col] = df_case[col].item()
    return df 

folder = "vital_csvs/"
#caseids = [caseids[0]]
# caseids = [5304]

def make_cases()
    for caseid in caseids:
        print("Make VitalFile")
        vf = vitaldb.VitalFile(caseid, total_tracks)
        print("Make Dataframe")
        df = vf.to_pandas(total_tracks, 1)
        print("Save Dataframe")
        #process it 
        df= trim_df(df, total_tracks)
        df = fill_gaps(df)
        df = add_static_info(caseid, df)

        save_name = f"{folder}{caseid}.csv"
        df.to_csv(save_name)

#make_cases()
