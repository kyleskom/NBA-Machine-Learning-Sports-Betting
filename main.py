import tensorflow as tf
import pandas as pd
import os
import numpy as np

data = pd.read_excel('2007-2008-Games.xlsx')
scores = data['Score']
data.drop(['Score'], axis=1, inplace=True)

data = data.drop(columns=['Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1'])
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(scores)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(299, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
print('DONE')
