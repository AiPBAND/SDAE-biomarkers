import pandas as pd
import os

FILE_NAME = 'GBM_class.csv'

pd.read_csv('data/csv/'+FILE_NAME, header=0, index_col=0).to_pickle('data/pd/'+os.path.splitext(FILE_NAME)[0])
