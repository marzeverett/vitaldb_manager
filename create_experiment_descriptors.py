import json 
import itertools 
import math 

#Broad group (lstm)
    #Model index (which kind of group, like shallow or deep, or predict)
        #Scaling factor 

loss_function = "mse"
num_epochs = 200
verbose_bool = True 
#NEED TO FIX 
def return_base_lstm_model(num_nodes):
    model = {
            "kind": "LSTM",
            "model_type": "Sequential",
            #Don't include input, code will figure it out. 
            #Don't include output, code will figure it out. 
            "layers": 
                [
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes*6),
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes*4),
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes, 2)
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    }
                ],
            "final_activation": "relu",
            "loss": loss_function,
            "optimizer": "adam",
            "batch_size": 32,
            "epochs": num_epochs,
            "test_split": 0.1,
            "validation_split": 0.2,
            "use_multiprocessing": True,
            "metrics": ["mse", "mape", "mae"],
            "verbose": verbose_bool,
        }
    return model

def create_deep_lstm_model_object(num_nodes):
    if num_nodes == 8:
        layers = [
                    {
                        "type": "LSTM",
                        "num_nodes": num_nodes*3,
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": num_nodes
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                ]
    elif num_nodes == 32:
        layers = [
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes*2),
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes*1.5),
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": num_nodes
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    }
                ]

    elif num_nodes == 64:
        layers = [
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes*2),
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": round(num_nodes*1.5),
                        "return_sequences": True,
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                        "type": "LSTM",
                        "num_nodes": num_nodes
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                    {
                    "type": "Dense",
                    "num_nodes": num_nodes,
                    "activation": "relu",
                    }
                ]

    model = {
            "kind": "LSTM",
            "model_type": "Sequential",
            #Don't include input, code will figure it out. 
            #Don't include output, code will figure it out. 
            "layers": layers,
            "final_activation": "relu",
            "loss": loss_function,
            "optimizer": "adam",
            "batch_size": 32,
            "epochs": num_epochs,
            "test_split": 0.1,
            "validation_split": 0.2,
            "use_multiprocessing": True,
            "metrics": ["mse", "mape", "mae"],
            "verbose": verbose_bool,
        }
    return model 


def create_predict_lstm_model_object(num_nodes):
    model = {
            "kind": "LSTM",
            "model_type": "Sequential",
            #Don't include input, code will figure it out. 
            #Don't include output, code will figure it out. 
            "layers": 
                [
                    {
                        "type": "LSTM",
                        "num_nodes": num_nodes
                    },
                    {
                        "type": "Dropout",
                        "percent": 0.2,
                    },
                ],
            "final_activation": "sigmoid",
            "loss": loss_function,
            "optimizer": "adam",
            "batch_size": 32,
            "epochs": num_epochs,
            "test_split": 0.1,
            "validation_split": 0.2,
            "use_multiprocessing": True,
            "metrics": ['mse', 'BinaryAccuracy', 'Precision', 'Recall', 
        'TruePositives', 'TrueNegatives','FalsePositives', 'FalseNegatives'],
            "verbose": verbose_bool,
        }
    return model 

def create_basic_ae_model_object(num_nodes):
    model = {
        "kind": "AE",
        "model_type": "Sequential",
        "layers": 
            [
                {
                    "type": "Dense",
                    "num_nodes": num_nodes * 3,
                    "activation": "relu",
                },
                {
                    "type": "Dense",
                    "num_nodes": num_nodes,
                    "activation": "relu",
                    "name": "latent_space"
                },
                {
                    "type": "Dense",
                    "num_nodes": num_nodes * 3,
                    "activation": "relu",
                },
            ],
        "final_activation": "relu",
        "loss": loss_function,
        #"loss_function": "mean_square_error",
        "optimizer": "adam",
        "batch_size": 32,
        "epochs": num_epochs,
        "test_split": 0.1,
        "validation_split": 0.2,
        "use_multiprocessing": True,
        #"metrics": ["mse"]
        "metrics": ["mse"],
        "verbose": verbose_bool,
    }
    return model 




def create_deep_ae_model_object(num_nodes):
    model = {
        "kind": "AE",
        "model_type": "Sequential",
        "layers": 
            [
                {
                    "type": "Dense",
                    "num_nodes": num_nodes * 2,
                    "activation": "relu",
                },
                {
                    "type": "Dense",
                    "num_nodes": num_nodes,
                    "activation": "relu",
                    "name": "latent_space"
                },
                {
                    "type": "Dense",
                    "num_nodes": num_nodes *2,
                    "activation": "relu",
                },
            ],
        "final_activation": "relu",
        "loss": loss_function,
        #"loss_function": "mean_square_error",
        "optimizer": "adam",
        "batch_size": 32,
        "epochs": num_epochs,
        "test_split": 0.1,
        "validation_split": 0.2,
        "use_multiprocessing": True,
        #"metrics": ["mse"]
        "metrics": ["mse"],
        "verbose": verbose_bool,
    }
    return model 


def return_model(dataset_descriptor, kind, index, num_nodes):
    #LSTM Regression
    if dataset_descriptor["predict_type"]:
        predict = True
    else:
        predict = False 
    if kind == "lstm":
        #Shallow regression model 
        if index == "0":
            if not predict:
                model = return_base_lstm_model(num_nodes)
            else:
                model = create_predict_lstm_model_object(num_nodes)
        #Deep regression model
        elif index == "1": 
            model = create_deep_lstm_model_object(num_nodes)
            
    #Autoencoding 
    elif kind == "ae":
        if index == "0":
            model = create_basic_ae_model_object(num_nodes)
        elif index == "1": 
            model = create_deep_ae_model_object(num_nodes)
    return model 

def return_experiment_path(scaling_factor, dataset_descriptor):
    dataset_name = dataset_descriptor["dataset_name"]
    phase_name = dataset_descriptor["phase_name"]
    letter = dataset_descriptor["letter"]
    experiment_path = f"generated_files/experiments/{phase_name}_{letter}/{scaling_factor}/{dataset_name}/"
    return experiment_path

def return_experiment_name(scaling_factor, dataset_descriptor):
    experiment_name =  f"{dataset_descriptor['phase_name']}_{dataset_descriptor['letter']}_{scaling_factor}"
    return experiment_name 

def create_experiment_descriptor(scaling_factor, dataset_descriptor, dataset_result):
    kind = dataset_descriptor["target_model"]
    group = dataset_descriptor["model_group"]
    x_vect = dataset_result["x"]
    if dataset_descriptor["target_model"] == "ae":
        #Number of features - might want to check this so model is not HUGE 
        nodes = math.ceil(x_vect.shape[1]*scaling_factor)
    else:
        nodes = scaling_factor
    model = return_model(dataset_descriptor, kind, group, nodes)
    experiment_1 = {
        "model": model,
        "dataset_name": dataset_descriptor["dataset_name"],
        "experiment_folder_path": return_experiment_path(scaling_factor, dataset_descriptor),
        "experiment_base_name": f"{dataset_descriptor['phase_name']}_{dataset_descriptor['letter']}",
        "experiment_name": return_experiment_name(scaling_factor, dataset_descriptor),
        "name_append": scaling_factor,
        "scaling_factor": scaling_factor,
    }
    return experiment_1




# #NEED TO FIX 
# def return_base_lstm_model(num_nodes):
#     model = {
#             "kind": "LSTM",
#             "model_type": "Sequential",
#             #Don't include input, code will figure it out. 
#             #Don't include output, code will figure it out. 
#             "layers": 
#                 [
#                     {
#                         "type": "LSTM",
#                         #Placeholder
#                         "num_nodes": num_nodes,
#                     },
#                     {
#                         "type": "Dropout",
#                         "percent": 0.2,
#                     },
#                 ],
#             "final_activation": "relu",
#             "loss": loss_function,
#             "optimizer": "adam",
#             "batch_size": 32,
#             "epochs": num_epochs,
#             "test_split": 0.1,
#             "validation_split": 0.2,
#             "use_multiprocessing": True,
#             "metrics": ["mse", "mape", "mae"],
#             "verbose": verbose_bool,
#         }
#     return model


# def create_basic_ae_model_object(num_nodes):
#     model = {
#         "kind": "AE",
#         "model_type": "Sequential",
#         "layers": 
#             [
#                 {
#                     "type": "Dense",
#                     "num_nodes": num_nodes,
#                     "activation": "relu",
#                     "name": "latent_space"
#                 },
#             ],
#         "final_activation": "relu",
#         "loss": loss_function,
#         #"loss_function": "mean_square_error",
#         "optimizer": "adam",
#         "batch_size": 32,
#         "epochs": num_epochs,
#         "test_split": 0.1,
#         "validation_split": 0.2,
#         "use_multiprocessing": True,
#         #"metrics": ["mse"]
#         "metrics": ["mse"],
#         "verbose": verbose_bool,
#     }
#     return model 
