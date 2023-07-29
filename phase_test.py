import json 
import run_experiments

letters = ['A', 'B', 'C', 'E', 'F', 'G', 'L', 
    'M', 'S', 'T', 'AC', 'AD']

#letters = ['A', 'B', 'C', 'E']


phase_dict = {
    "phase_name": "1",
    "phase_path_start": "generated_files/",
    "delete_stream": False,
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
        "retrained_letters": ["E", "H"],
    },
    "lstm_scaling_factors": [8, 32, 64],
    "ae_scaling_factors": [0.7]
}


run_experiments.run_phase_experiments(phase_dict, letters)
 
