import os
import sqlite3
import pandas as pd
from tqdm import tqdm

directory = os.fsdecode('../../Odds-Data/Odds-Data-Clean')
con = sqlite3.connect("../../Data/odds.sqlite")

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    df = pd.read_excel(f"../../Odds-Data/Odds-Data-Clean/{file}") # create DataFrame by reading Excel
    df.to_sql(f"odds_{file[:-5]}", con, if_exists="replace")

con.close()