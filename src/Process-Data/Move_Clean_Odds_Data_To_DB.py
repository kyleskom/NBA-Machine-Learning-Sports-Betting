import os
import sqlite3
import pandas as pd
import sys
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.Dictionaries import team_codes

directory = os.fsdecode('../../Odds-Data/Odds-Data-Clean')
con = sqlite3.connect("../../Data/odds.sqlite")

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)

    try:
        df = pd.read_excel(f"../../Odds-Data/Odds-Data-Clean/{file}") # create DataFrame by reading Excel
        df.to_sql(f"odds_{file[:-5]}", con, if_exists="replace")

    except Exception as e:
        print(e)

con.close()