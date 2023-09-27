#Help from:
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas 
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/ 

import pandas as pd
import matplotlib.pyplot as plt 
import json 
import os 
import scipy.stats as stats 
import seaborn as sn
import numpy as np 


def test_letters():
    #letters = ['A', 'G', 'T', 'AD']
    #also_letters = ['A', 'G', 'T', 'AD']
    letters = ['A', 'G', 'AD']
    also_letters = ['A', 'G', 'AD']
    #letters = ['N', 'W', 'Y', 'AB']
    #df = pd.read_csv("Full_Wilcoxon.csv")
    df = pd.read_csv("Wilcoxon_Redone.csv")
    letters_dict = {}
    #sub_letters_dict = {"A": [], "G": [], "T": [], "AD": []}
    sub_letters_dict = {"A": [], "G": [], "AD": []}
    col1 = df["A"].values.tolist()
    results_dict = {}
    for letter in letters:
        col1 = df[letter].values.tolist()
        new_dict = sub_letters_dict.copy()
        for also_letter in also_letters:
            if letter == also_letter:
                new_dict[also_letter].append(1)
            else:
                col2 = df[also_letter].values.tolist()
                result = stats.wilcoxon(col1, col2)
                new_dict[also_letter] = result.pvalue
        results_dict[letter] = new_dict
    print(results_dict)
    save_df = pd.DataFrame(results_dict)
    #save_df.to_csv("Wilcoxon_Full_Results.csv")
    save_df.to_csv("Wilcoxon_Full_Results_Redone.csv")
test_letters()
