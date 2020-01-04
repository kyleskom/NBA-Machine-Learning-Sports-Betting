import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time

tensorboard = TensorBoard(log_dir='Logs/{}'.format(str(time.time())))

data = pd.read_excel('Full-Data-Set.xlsx')
scores = data['Score']
margin = data['Home-Team-Win']
data.drop(['Score'], axis=1, inplace=True)
data.drop(['Home-Team-Win'], axis=1, inplace=True)

data = data.drop(columns=['Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1'])
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(margin)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_split=0.1)
# model.fit(x_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard])



#
# val_loss, val_acc = model.evaluate(x=x_test,y=y_test,  callbacks=[tensorboard])
# print(val_loss)
# print(val_acc)

#model.save('Trained-Model')

print('DONE')
