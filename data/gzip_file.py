import gzip
import shutil, os

PATH = "C:/Users/jerem/Documents/"
FILE="uni_exp_data_MA_7.csv"
FILE_PATH = os.path.join(PATH, FILE)
FILE_ZIP = os.path.join("data/compressed", FILE+".gz")

with open(FILE_PATH, 'rb') as f_in:
    with gzip.open(FILE_ZIP, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)