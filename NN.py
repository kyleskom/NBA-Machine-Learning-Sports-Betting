import tensorflow as tf
import pandas as pd
import os
import numpy as np

directory = os.fsdecode('Odds-Data-Clean')
df = pd.read_excel(directory + '/2007-08-Clean.xlsx')

directory = os.fsdecode('Team-Data')
data = pd.read_excel(directory + '/11-2-2007-08.xlsx')

x = pd.DataFrame()

# y = x.as_matrix()
#
# x_train = tf.keras.utils.normalize(y, axis=1)
# for row in df.itertuples():
#     date = row[2]
#     home = row[3]
#     away = row[4]
#     score = row[9]
#     ou = row[5]
#
print('HERE')

#x = pd.DataFrame(co)