import os
import sqlite3
import pandas as pd
import re
from tqdm import tqdm

top_directory = os.fsdecode('../../Team-Data')
con = sqlite3.connect("../../Data/teams.sqlite")
date_re = re.compile(r'(?P<month>\d+)-(?P<day>\d+)-(?P<season>\d+-\d+).xlsx')

for dir in tqdm(os.listdir(top_directory)):
    for file in tqdm(os.listdir(f"../../Team-Data/{dir}")):
        filename = os.fsdecode(file)

        df = pd.read_excel(f"../../Team-Data/{dir}/{file}") # create DataFrame by reading Excel
        match = date_re.search(file)
        if not match:
            continue
        month, day, season = match.groups()
        df.to_sql(f"teams_{season}-{month}-{day}", con, if_exists="replace")

con.close()