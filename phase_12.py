import json 
import run_experiments

#Actual letters we will use!!! 
#letters = ['A', 'C', 'E', 'F', 'G', 'S', 'T', 'AC', 'AD']
#Minus AEs - we can use from previous I believe. 
letters = ['A', 'C', 'F', 'G', 'T', 'AD']

phase_dict = {
    "phase_name": "12",
    "phase_path_start": "generated_files/",
    "delete_stream": False,
    "task_type": "prediction",
    "model_group": "0",
    "input_samples": [300],
    "output_samples": [10],
    "test": False,
    "retrain": False,
    "predict_type": ["dis_mortality_risk"],
    "building_on_phase": "10",
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
print("Finished!")
