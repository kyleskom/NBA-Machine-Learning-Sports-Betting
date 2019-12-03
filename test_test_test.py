import tensorflow as tf
import pandas as pd
import os
import numpy as np

directory = os.fsdecode('Odds-Data-Clean')
df = pd.read_excel(directory + '/2007-08-Clean.xlsx')

directory = os.fsdecode('Team-Data')
data = pd.read_excel(directory + '/11-2-2007-08.xlsx')



scores = []
count = 0
for row in df.itertuples():
    if count == 6:
        break
    else:
        scores.append(row[9])
        count += 1

team1 = data.iloc[29]
team2 = data.iloc[20]
series = team1.append(team2)

team1 = data.iloc[13]
team2 = data.iloc[10]
series2 = team1.append(team2)

team1 = data.iloc[15]
team2 = data.iloc[3]
series3 = team1.append(team2)

team1 = data.iloc[9]
team2 = data.iloc[23]
series4 = team1.append(team2)

team1 = data.iloc[5]
team2 = data.iloc[24]
series5 = team1.append(team2)

team1 = data.iloc[28]
team2 = data.iloc[8]
series6 = team1.append(team2)

#x = pd.DataFrame(data=series)
#stack = x.T

x = pd.concat([series, series2, series3, series4, series5, series6], ignore_index=True, axis=1)
x = x.T

frame = x.drop(columns=['TEAM_ID', 'TEAM_NAME', 'CFID', 'CFPARAMS', 'Date', 'Unnamed: 0'])

y = frame.values
y = y.astype(float)

x_train = tf.keras.utils.normalize(y, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(244, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#x_train = np.asarray(y)
y_train = np.asarray(scores)
model.fit(x_train, y_train, epochs=100)
print('HERE')