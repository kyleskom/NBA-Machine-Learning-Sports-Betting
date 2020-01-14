import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('../Models/Trained-Model-OU')
data = pd.read_excel('Full-Data-Set-UnderOver.xlsx')

data = data.iloc[15068:]
data.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover'], axis=1, inplace=True)

data = data.values
data = data.astype(float)

x_test = tf.keras.utils.normalize(data, axis=1)
predictions_array = []
for row in x_test:
    predictions_array.append(model.predict(np.array([row])))

for index in predictions_array:
    print(np.argmax(index))
