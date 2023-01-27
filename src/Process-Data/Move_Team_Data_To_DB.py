import os
import sqlite3
import pandas as pd
import sys
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.Dictionaries import team_codes

top_directory = os.fsdecode('../../Team-Data')
con = sqlite3.connect("../../Data/db.sqlite")

for dir in tqdm(os.listdir(top_directory)):
    for file in tqdm(os.listdir(f"../../Team-Data/{dir}")):
        filename = os.fsdecode(file)

        try:
            df = pd.read_excel(f"../../Team-Data/{dir}/{file}") # create DataFrame by reading Excel
            print(df.head()) # Print top 5 rows as sample
            df.to_sql(f"teams_{file[:-5]}", con, if_exists="replace")

        except Exception as e:
            print(e)

con.close()