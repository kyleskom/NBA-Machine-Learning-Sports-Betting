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
x = pd.DataFrame(data=series)
stack = x.T

print('HERE')
