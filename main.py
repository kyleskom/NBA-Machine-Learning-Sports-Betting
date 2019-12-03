import tensorflow as tf
import pandas as pd
import os
import numpy as np

directory = os.fsdecode('Odds-Data-Clean')
df = pd.read_excel(directory + '/2007-08-Clean.xlsx')

