import os
from tqdm import tqdm
import pandas as pd

directory = os.fsdecode('Odds-Data')

df = pd.DataFrame

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith('.xlsx'):
        temp = pd.read_excel(directory + '/' + filename)
        df.append(temp)
print('TEST')