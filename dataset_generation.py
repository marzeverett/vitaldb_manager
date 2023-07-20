import vitaldb 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

#Help from here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#And here: https://sparkbyexamples.com/pandas/pandas-add-constant-column-to-dataframe/ 
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas 




orch = ["Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_CT", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL"]
snu = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/FEM", "SNUADC/CVP" ]
solar = ["Solar8000/VENT_MAWP", "Solar8000/VENT_RR", "Solar8000/VENT_TV", "Solar8000/VENT_PPLAT", "Solar8000/VENT_PIP", "Solar8000/VENT_MV", "Solar8000/VENT_INSP_TM", "Solar8000/BT"]
#43 of these 
valid_cases = [4481, 3719, 1292, 397, 2327, 5018, 6009, 1820, 2332, 4255, 1191, 1959, 553, 3631, 2738, 818, 1590, 55, 5175, 4283, 5693, 1730, 5442, 3524, 4684, 5837, 1231, 6227, 985, 3930, 2267, 4573, 5983, 2272, 6246, 5607, 1900, 3694, 2168, 1785, 1018, 251]

total_tracks = snu+orch+solar
#total_tracks = snu

clinical = ["anestart", "aneend", "age", "sex", "height", "weight",
"bmi", "emop", "dx", "dis", "preop_pft", "preop_plt", "preop_pt", 
"preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_cr", 
"intraop_uo", "intraop_ffp"]

cat_list = ["sex", "emop", "dx", "preop_pft"]
lab = ["wbc", "hb", "hct", "gluc", "cr", "na", "k", "ammo",
 "ptinr", "pt%", "ptsec", "aptt", "ph"]


track_dict = {
    "snu": snu,
    "orch": orch,
    "solar": solar
}


df_cases = pd.read_csv("vital_csvs/clinical_info.csv")  # patient information



#Deal with categorical data 
def handle_categorical(df, categorical_list):
    cat_info = {}
    try:
        for field in categorical_list:
            df[field]= df[field].astype('category')
            field_categories = dict(enumerate(df[field].cat.categories)) 
            #print(field_categories)
            #print(list(df[field].cat.codes))
            df[field] = df[field].cat.codes
            cat_info[field] = field_categories
    except Exception as e:
        print("Could not code categorical variables: ", field, " ", e)
    return df, cat_info 

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
    #Forward fill first
    df = df.fillna(method="ffill")
    #Then backward fill, if needed 
    df = df.fillna(method="bfill")
    return df 


def add_static_info(case_id, df):
    df_cases = pd.read_csv("vital_csvs/clinical_info_cat.csv")
    df_case = df_cases.loc[df_cases["caseid"] == case_id]
    df_case = df_case[clinical]
    case_columns = df_case.columns.values.tolist()
    for col in case_columns:
        df[col] = df_case[col].item()
    return df 

def add_lab_info(caseid, df):
    df_labs = pd.read_csv("vital_csvs/lab_info.csv")
    empty_df = pd.DataFrame()
    first_time = df["Time"].iloc[0]
    last_time = df["Time"].iloc[-1]
    for lab_item in lab:
        empty_df = pd.DataFrame()
        sub_lab = df_labs[df_labs["caseid"] == caseid]
        sub_lab = sub_lab[sub_lab["name"] == lab_item]
        empty_df[lab_item] = sub_lab["result"]
        empty_df["Time"] = sub_lab["dt"]
        empty_df = empty_df[empty_df.Time >= first_time]
        empty_df = empty_df[empty_df.Time <= last_time]
        if not empty_df.empty:
            for index, row in empty_df.iterrows():
                df.loc[df["Time"] == row["Time"], lab_item] = row[lab_item]
    return df 

#https://www.geeksforgeeks.org/create-a-new-column-in-pandas-dataframe-based-on-the-existing-columns/ 
# df['Discounted_Price'] = df.apply(lambda row: row.Cost -
#                                   (row.Cost * 0.1), axis = 1)
#df.drop(df[df['Fee'] >= 24000].index, inplace = True)

def categorize_clinical_info():
    df_cases = pd.read_csv("vital_csvs/clinical_info.csv")
    df, cat_info = handle_categorical(df_cases, cat_list)
    print(cat_info)
    cat_df = pd.DataFrame.from_dict(cat_info)
    print(cat_df.head())
    df.to_csv("vital_csvs/clinical_info_cat.csv")
    cat_df.to_csv("vital_csvs/categorical_coding.csv")

#categorize_clinical_info()


folder = "vital_csvs/"
#caseids = [caseids[0]]
#caseids = [4481]
caseids = valid_cases

def make_cases():
    for caseid in caseids:
        print("Make VitalFile")
        vf = vitaldb.VitalFile(caseid, total_tracks)
        print("Make Dataframe")
        df = vf.to_pandas(total_tracks, 1, return_timestamp=True)
        print("Save Dataframe")
        #process it 
        df= trim_df(df, total_tracks)
        
        df = add_static_info(caseid, df)
        #df = pd.read_csv("vital_csvs/4481.csv")
        df = add_lab_info(caseid, df)
        df = fill_gaps(df)
        save_name = f"{folder}{caseid}.csv"
        df.to_csv(save_name)

def add_other_pred_columns():
    for caseid in caseids:
        print(caseid)
        read_name = f"{folder}{caseid}.csv"
        df = pd.read_csv(read_name)
        df["dis_mortality_risk"] = np.where(df['dis'] > 1209600, 1, 0)
        #df['gluc_risk'] = np.where(df['gluc'] > 150, 1, 0)
        df.to_csv(read_name)


#make_cases()

#add_other_pred_columns()

