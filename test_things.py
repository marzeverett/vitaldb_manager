
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import create_metrics 

path = "generated_files/experiments/1_A/8/1_A_0_0_0_0_0_30_10/"

dataset_descriptor, dataset_result, experiment_descriptor, experiment_result = create_metrics.load_everything(path)
create_metrics.just_visualize(dataset_descriptor, dataset_result, experiment_descriptor, experiment_result, filter_results=4481)

#print(experiment_result.keys())
#print(experiment_result["per_feature"].keys())