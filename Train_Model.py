import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time
from kerastuner.tuners import RandomSearch
import gc
from kerastuner.engine.hyperparameters import HyperParameters

LOG_DIR = f"{int(time.time())}"

tensorboard = TensorBoard(log_dir='Logs/{}'.format(str(time.time())))

data = pd.read_excel('Full-Data-Set.xlsx')
data = data.iloc[:15068]
test = data.iloc[15068:]

scores = data['Score']
margin = data['Home-Team-Win']
data.drop(['Score'], axis=1, inplace=True)
data.drop(['Home-Team-Win'], axis=1, inplace=True)

data = data.drop(columns=['Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1'])
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(margin)


scoresy = test['Score']
marginy = test['Home-Team-Win']
test.drop(['Score'], axis=1, inplace=True)
test.drop(['Home-Team-Win'], axis=1, inplace=True)

test = test.drop(columns=['Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1'])
test = test.values
test = test.astype(float)

x_test = tf.keras.utils.normalize(test, axis=1)
y_test = np.asarray(marginy)

del data
del test
del scores
del margin
del scoresy
del marginy
gc.collect()


def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    for i in range(hp.Int('n_hidden_layers', min_value=1, max_value=5, step=1)):
        model.add(tf.keras.layers.Dense(hp.Int(f"hidden_{i}_units", min_value=128, max_value=1028, step=32), activation=tf.nn.relu))

    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory=LOG_DIR
)
gc.collect()
tuner.search(x=x_train, y=y_train, epochs=20, batch_size=32,  validation_split=0.1)
print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())
