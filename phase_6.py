import json 
import run_experiments


#Actual letters we will use!!! 
letters = ['A', 'C', 'E', 'F', 'G', 'S', 'T', 'AC', 'AD']
#letters = ['F', 'G', 'S', 'T', 'AC', 'AD']

phase_dict = {
    "phase_name": "6",
    "phase_path_start": "generated_files/",
    "delete_stream": ["orch"],
    "task_type": "regression",
    "model_group": "0",
    "input_samples": [30],
    "output_samples": [10],
    "test": False,
    "retrain": False,
    "predict_type": None,
    "building_on_phase": False,
    "retrain_dict": {
        "prev_delete_stream": "clinical",
        "retrain_from_phase": "4",
        "separation_scheme": "ds",
        "retrained_letters": ["E"],
    },
    "lstm_scaling_factors": [8, 32, 64],
    "ae_scaling_factors": [0.7]
}

#REMEMBER TO CHANGE PREDICT TYPE AND TASK TYPE BOTH
#predictions = ["emop", "dis_mortality_risk", "gluc_risk"]


#run_experiments.test_phase_experiments(phase_dict, letters)
run_experiments.run_phase_experiments(phase_dict, letters)

# #Regular
# phase_dict = {
#     "phase_name": "1",
#     "phase_path_start": "generated_files/",
#     "delete_stream": False,
#     "task_type": "regression",
#     "model_group": "0",
#     "input_samples": [30],
#     "output_samples": [10],
#     "test": False,
#     "retrain": False,
#     #"predict_type": None,
#     "predict_type": None,

#     "building_on_phase": False,
#     "retrain_dict": {
#         "prev_delete_stream": "clinical",
#         "retrain_from_phase": "4",
#         "separation_scheme": "ds",
#         "retrained_letters": ["E"],
#     },
#     "lstm_scaling_factors": [8, 32, 64],
#     "ae_scaling_factors": [0.7]
# }


# phase_dict = {
#     "phase_name": "2",
#     "phase_path_start": "generated_files/",
#     "delete_stream": False,
#     "task_type": "regression",
#     "model_group": "0",
#     "input_samples": [30],
#     "output_samples": [10],
#     "test": False,
#     "retrain": False,
#     #"predict_type": None,
#     "predict_type": ["emop", "gluc_risk"],

#     "building_on_phase": False,
#     "retrain_dict": {
#         "prev_delete_stream": "clinical",
#         "retrain_from_phase": "4",
#         "separation_scheme": "ds",
#         "retrained_letters": ["E"],
#     },
#     "lstm_scaling_factors": [8, 32, 64],
#     "ae_scaling_factors": [0.7]
# }
