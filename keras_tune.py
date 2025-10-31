'''
function name: keras_tune.py

keras_tune.py used for KerasTuner etc.
ref:https://keras.io/guides/keras_tuner/getting_started/
'''

#kera tuner
from tensorflow import keras
from keras import layers
import keras_tuner
import tensorflow as tf
from keras import metrics
import subprocess

'''
#embed #1. rmdir /s /q mlsubtuner in python code to clean kera tuner cache
#directory path to remove (replace with your desired path)

#for windows only

#embed #1. rmdir /s /q mlsubtuner in python code to clean kera tuner cache
#directory path to remove (replace with your desired path)
directory_to_remove = 'mlsubtuner'
try:
    #run the rmdir command to remove the directory
    subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', directory_to_remove], check=True)
    print(f"Directory '{directory_to_remove}' removed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")



#access tensor - note log directory is differnet from the one in tensorboard_tune
#1. rmdir /s /q mlsubtuner     must clean catch
#2. run python file
#3. type tensorboard --logdir /tmp/tb_logs
#4. open http://localhost:6006/ in browser
'''
#keras tuner build model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())

    #tune the number of layers
    for i in range(hp.Int("num_layers", 1, 10)):
        model.add(
            layers.Dense(
                #tune number of units separately
                units=hp.Int(f"units_{i}", min_value=32, max_value=1024, step=32),
                activation=hp.Choice("activation", ["relu", "tanh", "softmax"]),
            )
        )

    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.20))

    model.add(layers.Dense(n_outputs))
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-1, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[metrics.MeanSquaredError()],
    )
    return model

def keras_tune(X_train, X_val, y_train, y_val):
    global n_outputs #n_outputs is the number of output as well as the number of units used in the output layer

    n_outputs = y_train.shape[1] #define output shape
    #n_inputs = X_train.shape[1] #input shape only for information purpose

    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)#define early stop

    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.RandomSearch(
        hypermodel = build_model,
        objective = "val_loss",
        max_trials = 40,
        #max_trials = 5,
        project_name = 'mlsubtuner'
    )
   
    num_epochs = 50
    num_batch = 64
  
    tuner.search(X_train, y_train, epochs=num_epochs, batch_size=num_batch, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stop,keras.callbacks.TensorBoard("/tmp/tb_logs")],)

    #print out the tunning summary
    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    #print out all hyperparameters and their values
    print('\n\n best parameters are:\n')
    for hyperparameter, value in best_hps.values.items():
        print(f"{hyperparameter}: {value}")

    best_model = build_model(best_hps)

    # Open a text file for writing
    with open("besthyperparameters.txt", "w") as file:
        # Print out all hyperparameters and their values to the file
        file.write('\n\nBest parameters are:\n')
        for hyperparameter, value in best_hps.values.items():
            file.write(f"{hyperparameter}: {value}\n")

    return best_model
