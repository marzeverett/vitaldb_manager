import json 
import run_experiments

#Actual letters we will use!!! 
#letters = ['A', 'C', 'E', 'F', 'G', 'S', 'T', 'AC', 'AD']
#Minus AEs - we can use from previous I believe. 
letters = ['G', 'T']

phase_dict = {
    "phase_name": "8",
    "phase_path_start": "generated_files/",
    "delete_stream": False,
    "task_type": "prediction",
    "model_group": "0",
    "input_samples": [30],
    "output_samples": [10],
    "test": False,
    "retrain": True,
    #MAY CHANGE
    "predict_type": ["emop"],
    #"predict_type": ["dis_mortality_risk"],
    "building_on_phase": "5",
    "retrain_dict": {
        "sub_ae_phase": "7",
        "prev_delete_stream": "orch",
        "retrain_from_phase": "5",
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
