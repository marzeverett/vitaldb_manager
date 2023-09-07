#Help from here for reshaping. VERY HELPFUL! https://stackoverflow.com/questions/56255643/in-keras-how-to-get-3d-input-and-3d-output-for-lstm-layers 
#Code from here:
#https://levelup.gitconnected.com/install-tensorflow-2-0-0-on-ubuntu-18-04-with-nvidia-gtx1650-gtx1660ti-9fd7a837d5fd
#Keras has a built in way to separate time steps" https://keras.io/examples/timeseries/timeseries_weather_forecasting/ 
#Extensive use of the keras documentation 
#For reshaping y: 
#https://stackoverflow.com/questions/54948059/keras-2d-dense-layer-for-output
#Help from here: https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
#On suppressing annoying tf output!!
#These two lines suppress annoying tf output 
#Plotting help from here: https://stackabuse.com/matplotlib-plot-multiple-line-plots-same-and-different-scales/ 
#Getting numpy column help from here: https://stackoverflow.com/questions/4455076/how-do-i-access-the-ith-column-of-a-numpy-multidimensional-array 
#Pickle help here: https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/WorkingWithFiles.html 
#Use this info for per-feature error: https://neptune.ai/blog/keras-loss-functions
#Helpful for intermediate model layer: https://androidkt.com/get-output-of-intermediate-layers-keras/  
#Encoder example from Keras: https://keras.io/examples/vision/autoencoder/ 
#https://stackoverflow.com/questions/63053427/keras-add-layers-to-another-model 
#Regularization help here: https://johnthas.medium.com/regularization-in-tensorflow-using-keras-api-48aba746ae21 

import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
#
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import dataset_generator as dg 
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#NOTES - You need to better define your saving/loading naming schema for a particular dataset. 

#true first, then pred. 
#add binary cross entropy here 
loss_dict = {
    "mape": tf.keras.losses.MeanAbsolutePercentageError(),
    "mse": tf.keras.losses.MeanSquaredError(),
    "mae": tf.keras.losses.MeanAbsoluteError(),
    "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy(),
    "CategoricalCrossentropy": tf.keras.losses.CategoricalCrossentropy(),
    "BinaryAccuracy": tf.keras.metrics.BinaryAccuracy(),
    "Precision": tf.keras.metrics.Precision(name="precision"),
    "Recall": tf.keras.metrics.Recall(name="recall"),
    "TruePositives": tf.keras.metrics.TruePositives(),
    "TrueNegatives": tf.keras.metrics.TrueNegatives(),
    "FalsePositives": tf.keras.metrics.FalsePositives(),
    "FalseNegatives":  tf.keras.metrics.FalseNegatives()
}

#CHANGE is here! 
def split_training_test(prepared_dataset, experiment_object):
    model_def = experiment_object["model"]
    if "test_split" in list(model_def.keys()):
        test_split = model_def["test_split"]
    else:
        test_split = 0.2
    random_state=None
    if "random_state" in list(model_def.keys()):
        random_state = random_state
    # x_train, x_test, y_train, y_test, x_train_key, x_test_key, y_train_key, y_test_key = train_test_split(
    #     prepared_dataset["x"], prepared_dataset["y"], prepared_dataset["x_key"], prepared_dataset["y_key"],
    #     test_size=test_split,
    #     random_state=random_state
    # )

    x_train, x_test, y_train, y_test, x_train_key, x_test_key, y_train_key, y_test_key = train_test_split(
        prepared_dataset["x"], prepared_dataset["y"], prepared_dataset["x_key"], prepared_dataset["y_key"],
        test_size=test_split,
        shuffle=False
    )
    prepared_dataset["x_train"] = x_train
    prepared_dataset["x_test"] = x_test
    prepared_dataset["y_train"] = y_train
    prepared_dataset["y_test"] = y_test
    prepared_dataset["x_train_key"] = x_train_key
    prepared_dataset["x_test_key"] = x_test_key
    prepared_dataset["y_train_key"] = y_train_key
    prepared_dataset["y_test_key"] = y_test_key
    return prepared_dataset


def build_layer(model, layer_object):
    layer_type = layer_object["type"]
    #LSTM layer 
    if layer_type == "LSTM":
        if "num_nodes" in list(layer_object.keys()):
            num_nodes=layer_object["num_nodes"]
        else:
            num_nodes = 20 
        if "return_sequences" in list(layer_object.keys()):
            return_sequences=layer_object["return_sequences"]
        else:
            return_sequences = False
        model.add(layers.LSTM(num_nodes, return_sequences=return_sequences))
        #model.add(layers.LSTM(num_nodes, return_sequences=return_sequences, kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.01)))
    
    #Dropout Layer 
    if layer_type == "Dropout":
        if "percent" in list(layer_object.keys()):
            percent = layer_object["percent"]
        else:
            percent = 0.5
        model.add(layers.Dropout(percent))
    #Dense Layer 
    if layer_type == "Dense":
        activation = None
        use_bias = True
        name = None
        if "num_nodes" in list(layer_object.keys()):
            num_nodes=layer_object["num_nodes"]
        else:
            num_nodes = 20 
        if "activation" in list(layer_object.keys()):
            activation = layer_object["activation"]
        if "use_bias" in list(layer_object.keys()):
            use_bias = layer_object["use_bias"]
        if "name" in list(layer_object.keys()):
            name = layer_object["name"]
        
        model.add(layers.Dense(num_nodes,
        activation=activation,
        use_bias=use_bias,
        name=name
        ))
    if layer_type == "ConvLSTM2D":
        activation = "tanh"
        recurrent_activation = "hard_sigmoid"
        if "num_nodes" in list(layer_object.keys()):
            num_nodes=layer_object["num_nodes"]
        else:
            num_nodes = 20 
        if "kernel_size" in list(layer_object.keys()):
            kernel_size=layer_object["kernel_size"]
        else:
            kernel_size = 3 
        if "activation" in list(layer_object.keys()):
            activation = layer_object["activation"]
        if "recurrent_activation" in list(layer_object.keys()):
            recurrent_activation = layer_object["recurrent_activation"]

        model.add(layers.ConvLSTM2D(num_nodes, kernel_size,
        activation=activation,
        recurrent_activation=recurrent_activation))

    if layer_type == "Conv2D":
        activation = None
        name = None
        kernel_size = 3
        strides = (1,1)
        if "num_nodes" in list(layer_object.keys()):
            num_nodes=layer_object["num_nodes"]
        else:
            num_nodes = 20 
        if "activation" in list(layer_object.keys()):
            activation = layer_object["activation"]
        if "kernel_size" in list(layer_object.keys()):
            kernel_size = layer_object["kernel_size"]
        if "strides" in list(layer_object.keys()):
            strides = layer_object["strides"]
        if "name" in list(layer_object.keys()):
            name = layer_object["name"]

        model.add(layers.Conv2D(num_nodes, kernel_size, name=name,
        activation=activation, strides=strides))

    if layer_type == "Conv2DTranspose":
        activation = None
        kernel_size = 3
        strides = (1,1)
        name=None
        if "num_nodes" in list(layer_object.keys()):
            num_nodes=layer_object["num_nodes"]
        else:
            num_nodes = 20 
        if "activation" in list(layer_object.keys()):
            activation = layer_object["activation"]
        if "kernel_size" in list(layer_object.keys()):
            kernel_size = layer_object["kernel_size"]
        if "strides" in list(layer_object.keys()):
            strides = layer_object["strides"]
        if "name" in list(layer_object.keys()):
            name = layer_object["name"]

        model.add(layers.Conv2DTranspose(num_nodes, kernel_size, name=name,
        activation=activation, strides=strides))

    if layer_type == "MaxPooling2D":
        pool_size= (2,2)
        strides = (1,1)
        name=None
        if "pool_size" in list(layer_object.keys()):
            pool_size=layer_object["pool_size"]
        if "strides" in list(layer_object.keys()):
            strides = layer_object["strides"]
        if "name" in list(layer_object.keys()):
            name = layer_object["name"]

        model.add(layers.MaxPooling2D(pool_size=pool_size, name=name,
        strides=strides))

    if layer_type == "UpSampling2D":
        size= (2,2)
        name=None
        if "size" in list(layer_object.keys()):
            size=layer_object["size"]
        if "name" in list(layer_object.keys()):
            name = layer_object["name"]

        model.add(layers.UpSampling2D(size=size, name=name))

    if layer_type == "BatchNormalization":
        model.add(layers.BatchNormalization())

    if layer_type == "Flatten":
        model.add(layers.Flatten())
        
    return model

def return_base_model(model_type):
    if model_type == "Sequential":
        model = models.Sequential()
        return model 

def add_input_layer(model, prepared_dataset, experiment_object):
    model_def = experiment_object["model"]
    activation = model_def["final_activation"]
    x = prepared_dataset["x"]
    dimensions = x.ndim
    #Num samples, num samples in sequnece, num features 
    if dimensions == 3:
        dims = (x.shape[1], x.shape[2])
        model.add(layers.Input(shape=dims))
    if model_def["kind"] == "AE":
        dims = x.shape[1]
        model.add(layers.Input(shape=dims))
        model.add(layers.Dense(dims, activation=activation))
    if model_def["kind"] == "CONV_AE":
        shape = x.shape
        #Everything but the first shape for the Conv AE 
        input_shape = shape[1:]
        print("INPUT SHAPE", input_shape)
        model.add(layers.Input(shape=input_shape))
    if model_def["kind"] == "CONV_LSTM":
        shape = x.shape
        #Everything but the first shape for the Conv AE 
        input_shape = shape[1:]
        print("INPUT SHAPE", input_shape)
        model.add(layers.Input(shape=input_shape))

    return model 

def add_output_layer(model, prepared_dataset, experiment_object):
    model_def = experiment_object["model"]
    activation = model_def["final_activation"]
    y = prepared_dataset["y"]
    x = prepared_dataset["x"]
    #print("OUTPUT SHAPE")
    #print(y.shape)
    num_dimensions = y.ndim
    if model_def["kind"] == "CONV_AE":
        model.add(layers.Conv2D(y.shape[-1], (1,1)))
    else:
        #I think this is fine, just needs some rework. -- Lots of rework for a 
        #conv ae
        if num_dimensions <= 2:
            features = y.shape[1]
            model.add(layers.Dense(features,
                activation=activation))
        else:
            timesteps = y.shape[1]
            features = y.shape[2]
            model.add(layers.Dense(timesteps*features,
                activation=activation))
            model.add(layers.Reshape((timesteps, features)))
    return model 


#10 = timesteps, 2 = features 
# model.add(Dense(10 * 2))
# model.add(Reshape((10, 2)))
def build_model(prepared_dataset, experiment_object, dataset_descriptor):
    model_def = experiment_object["model"]
    model_type = model_def["model_type"]
    model = return_base_model(model_type)
    #Create input layer 
    model = add_input_layer(model, prepared_dataset, experiment_object)
    
    #Maybe put in later? 
    # #Create defined layers - Maybe add the transfer model here instead? 
    # if dataset_descriptor["transfer_learn"]:
    #     model = transfer_learn(model, experiment_object, dataset_descriptor)
    # else:
    layer_list = model_def["layers"]
    for layer in layer_list:
        model = build_layer(model, layer)
    
    #Create output layer
    model = add_output_layer(model, prepared_dataset, experiment_object)
    #Compile the model 
    model = compile_model(model, experiment_object)
    return model 

def compile_model(model, experiment_object):
    model_def = experiment_object["model"]
    optimizer = None
    loss = None
    metrics = None
    loss_weights = None
    weighted_metrics = None
    run_eagerly = True
    steps_per_execution=None
    jit_compile=None
    if "optimizer" in list(model_def.keys()):
        #Change is here- make nicer!
        #optimizer = model_def["optimizer"]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if "loss" in list(model_def.keys()):
        loss = model_def["loss"]
    if "metrics" in list(model_def.keys()):
        metrics = model_def["metrics"]
    if "loss_weights" in list(model_def.keys()):
        loss_weights = model_def["loss_weights"]
    if "weighted_metrics" in list(model_def.keys()):
        weighted_metrics = model_def["weighted_metrics"]
    if "run_eagerly" in list(model_def.keys()):
        run_eagerly = model_def["run_eagerly"]
    if "steps_per_execution" in list(model_def.keys()):
        steps_per_execution = model_def["steps_per_execution"]
    if "jit_compile" in list(model_def.keys()):
        jit_compile = model_def["jit_compile"]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
        run_eagerly=run_eagerly,
        steps_per_execution=steps_per_execution,
        jit_compile=jit_compile
    )
    return model 


#Experiment folder path 
def get_full_experiment_folder(experiment_object):
    full_path = experiment_object["experiment_folder_path"]
    return full_path

#Return checkpoint filepath
def return_checkpoint_filepath(experiment_object):
    save_path = get_full_experiment_folder(experiment_object)+"tmp/checkpoint"
    return save_path

#The callbacks we will use 
def build_callbacks(experiment_object):
    save_path = return_checkpoint_filepath(experiment_object)
    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=True,
        monitor='val_loss',
        mode="min",
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        #Change is here!
        patience=10,
        mode="min",
        restore_best_weights=True,
    )
    return [save_best, early_stop]

def fit_model(model, prepared_dataset, experiment_object):
    model_def = experiment_object["model"]
    if "x_train" in list(prepared_dataset.keys()):
        x_vect = prepared_dataset["x_train"]
        y_vect = prepared_dataset["y_train"]
    else:
        x_vect = prepared_dataset["x"]
        y_vect = prepared_dataset["y"]
    batch_size=None
    epochs=1
    verbose=False
    #verbose=True
    callbacks = build_callbacks(experiment_object)
    validation_split=0.0
    validation_data=None
    shuffle=True
    use_multiprocessing=False
    if "batch_size" in list(model_def.keys()):
        batch_size = model_def["batch_size"]
    if "epochs" in list(model_def.keys()):
        epochs = model_def["epochs"]
    if "verbose" in list(model_def.keys()):
        verbose = model_def["verbose"]
    if "validation_split" in list(model_def.keys()):
        validation_split = model_def["validation_split"]
    if "validation_data" in list(model_def.keys()):
        validation_data = model_def["validation_data"]
    if "shuffle" in list(model_def.keys()):
        shuffle = model_def["shuffle"]
    if "use_multiprocessing" in list(model_def.keys()):
        use_multiprocessing = model_def["use_multiprocessing"]

    start_time = time.time()
    print(f"Training model for {epochs} epochs")
    #Actually train it 
    history = model.fit(
        x_vect, 
        y_vect, 
        batch_size=batch_size, 
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        use_multiprocessing=use_multiprocessing)
    end_time = time.time()
    #Restore the best model
    save_path = return_checkpoint_filepath(experiment_object)
    try:
        model.load_weights(save_path)
    except Exception as e:
        print("Issue loading model, likely ", e)
    total_time = end_time - start_time
    return history, total_time




def evaluate_model(model, prepared_dataset, experiment_object):
    if "x_test" in list(prepared_dataset.keys()):
        x_vect = prepared_dataset["x_test"]
        y_vect = prepared_dataset["y_test"]
    else:
        x_vect = prepared_dataset["x"]
        y_vect = prepared_dataset["y"]

    final_metrics = model.evaluate(x=x_vect, y=y_vect, verbose=0)
    print("evaluated model")
    metrics_dict = {}
    #Dictionary workaround from here: https://github.com/keras-team/keras/issues/14045
    final_metrics = {name: final_metrics[val] for val, name in enumerate(model.metrics_names)}
    #print("Final metrics", final_metrics)
    return final_metrics


def predict_values(model, prepared_dataset):
    predictions = {} 
    values_to_predict = ["x", "x_train", "x_test"]
    for item in values_to_predict:
        predictions[item] = model.predict(prepared_dataset[item], verbose=0)
    return predictions 

def graph_predictions(prepared_dataset, pred_key=None, y_key=None):
    if pred_key == None:
        pred_index = "predictions"
    else:
        pred_index = pred_key
    if y_key == None:
        y_index = "y_key"
    else:
        y_index = y_key
    fig, ax = plt.subplots()
    ax.plot(prepared_dataset[y_index], prepared_dataset[pred_index][:, 0])
    plt.show()
    
def eval_feature(true_column, pred_column, loss):
    loss_function = loss_dict[loss]
    # if true_column.ndim > 1: 
    #     true_column = true_column.reshape(true_column[0]*true_column[1])
    # if pred_column.ndim > 1: 
    #     pred_column = pred_column.reshape(pred_column[0]*pred_column[1])

    value = loss_function(true_column, pred_column).numpy()
    return value

    #y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1], y_true.shape[2])

def eval_all_features(true_column, pred_column, metric, experiment_object):
    if true_column.ndim > 2: 
        true_column = true_column.reshape(true_column.shape[0]*true_column.shape[1], true_column.shape[2])
    if pred_column.ndim > 2: 
        pred_column = pred_column.reshape(pred_column.shape[0]*pred_column.shape[1], pred_column.shape[2])
    feature_list = []
    #For each feature
    for i in range(0, len(true_column[1])):
        val = eval_feature(true_column[:, i], pred_column[:, i], metric)
        feature_list.append(val)
    return feature_list

    #y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1], y_true.shape[2])

def get_all_per_feature_evals(predictions, prepared_dataset, experiment_object):
    per_feature = {}
    x_test = predictions["x_test"]
    y_test = prepared_dataset["y_test"]
    metrics = experiment_object["model"]["metrics"]
    for metric in metrics:
        feature_list = eval_all_features(y_test.copy(), x_test.copy(), metric, experiment_object)
        per_feature[metric] = feature_list
    return per_feature


#Create the experiment result object. 
def create_experiment_result_object(history, total_time, model, prepared_dataset, experiment_object):
    experiment_result = {} 
    #Get the entire model history
    experiment_result["model_history"] = history.history
    #experiment training time
    experiment_result["training_time"] = total_time
    #Get the test evaluation metrics 
    test_metrics = evaluate_model(model, prepared_dataset, experiment_object)
    experiment_result["test_metrics"] = test_metrics
    #Predict on all x's, also test and train x's. 
    experiment_result["predictions"] = predict_values(model, prepared_dataset)
    #Get the per-feature metrics on test set.
    #CHANGE IS HERE - Can't quite get a per feature idea yet -- REVISIT 
    if experiment_object["model"]["kind"] != "CONV_AE":
        experiment_result["per_feature"] = get_all_per_feature_evals(experiment_result["predictions"], prepared_dataset, experiment_object)
    #ax.plot(prepared_dataset[y_index], prepared_dataset[pred_index][:, 0])
    return experiment_result


def save_model(model, experiment_object):
    model_def = experiment_object["model"]
    path = get_full_experiment_folder(experiment_object)
    save_path = path+"model"
    model.save(save_path)
    #If this an autoencoder, we also want to save the latent space model 
    if model_def["kind"] == "AE" or model_def["kind"] == "CONV_AE":
        latent_space = model.get_layer(name="latent_space")
        latent_output = latent_space.output
        latent_model = models.Model(inputs = model.input, outputs=latent_output)
        latent_save_path = path+"latent_model"
        latent_model.save(latent_save_path)

def save_experiment(dataset_descriptor, dataset_result, experiment_descriptor, experiment_result):
    dd_path = get_full_experiment_folder(experiment_descriptor)+"dataset_descriptor.pickle"
    dr_path = get_full_experiment_folder(experiment_descriptor)+"dataset_result.pickle"
    ed_path = get_full_experiment_folder(experiment_descriptor)+"experiment_descriptor.pickle"
    er_path = get_full_experiment_folder(experiment_descriptor)+"experiment_result.pickle"
    with open(dd_path, "wb") as f:
        pickle.dump(dataset_descriptor, f)
    with open(dr_path, "wb") as f:
        pickle.dump(dataset_result, f)
    with open(ed_path, "wb") as f:
        pickle.dump(experiment_descriptor, f)
    with open(er_path, "wb") as f:
        pickle.dump(experiment_result, f)

def load_in_experiment_files(experiment_descriptor):
    dd_path = get_full_experiment_folder(experiment_descriptor)+"dataset_descriptor.pickle"
    dr_path = get_full_experiment_folder(experiment_descriptor)+"dataset_result.pickle"
    ed_path = get_full_experiment_folder(experiment_descriptor)+"experiment_descriptor.pickle"
    er_path = get_full_experiment_folder(experiment_descriptor)+"experiment_result.pickle"
    with open(dd_path, "rb") as f:
        dataset_descriptor = pickle.load(f)
    with open(dr_path, "rb") as f:
        dataset_result = pickle.load(f)
    with open(ed_path, "rb") as f:
        experiment_descriptor = pickle.load(f)
    with open(er_path, "rb") as f:
        experiment_result = pickle.load(f)
    return dataset_descriptor, dataset_result, experiment_descriptor, experiment_result


def experiment_from_experiment_object(dataset_descriptor, prepared_dataset, experiment_object):
    prepared_dataset = split_training_test(prepared_dataset, experiment_object)
    #Build the model
    model = build_model(prepared_dataset, experiment_object, dataset_descriptor)
    #Change here 
    print(model.summary())
    history, total_time = fit_model(model, prepared_dataset, experiment_object)
    save_model(model, experiment_object)
    experiment_result = create_experiment_result_object(history, total_time, model, prepared_dataset, experiment_object)
    #Save the experiment 
    save_experiment(dataset_descriptor, prepared_dataset, experiment_object, experiment_result)
    return dataset_descriptor, prepared_dataset, experiment_object, experiment_result
  

