import glob, os, shutil

RESULTS_DIR = 'results'
DATA_DIR = 'data'

for f in glob.glob("[0-9]{*}.txt"):
    if os.path.isfile(f):
        os.remove(f)

shutil.rmtree(RESULTS_DIR, ignore_errors=True)
shutil.rmtree(DATA_DIR, ignore_errors=True)