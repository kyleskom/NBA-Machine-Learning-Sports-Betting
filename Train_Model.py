import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import time


t = str(time.time())
tensorboard = TensorBoard(log_dir='Logs/{}'.format(t))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('Trained-Model-' + t, save_best_only=True, monitor='val_loss', mode='min')

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
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#history = model.fit(x_train, y_train, epochs=20, validation_split=0.1)
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32, callbacks=[tensorboard, earlyStopping, mcp_save])


#
# val_loss, val_acc = model.evaluate(x=x_test,y=y_test,  callbacks=[tensorboard])
# print(val_loss)
# print(val_acc)

#print(history.history)

#model.save('Trained-Model-' + str(history.history['val_loss']))

print('DONE')
