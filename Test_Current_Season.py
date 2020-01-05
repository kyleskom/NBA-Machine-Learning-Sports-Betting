from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import copy

model = load_model('Trained-Model')
data = pd.read_excel('Full-Data-Set.xlsx')
data = data.iloc[15068:]
copy = copy.deepcopy(data)
scores = data['Score']
margin = data['Home-Team-Win']
data.drop(['Score'], axis=1, inplace=True)
data.drop(['Home-Team-Win'], axis=1, inplace=True)

data = data.drop(columns=['Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1'])
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
arr = []
for row in x_train:
      arr.append(model.predict(np.array([row])))

for x in arr:
      print(np.argmax(x))
