
import wandb, os
import gzip

PROJECT='SDAE-biomarkers'
ENTITY="aipband"
FILE="uni_exp_data_MA_7.csv.gz"
FILEPATH=os.path.join("data\\compressed", FILE)

wandb.login()
wandb.init(project=PROJECT, entity=ENTITY)

def file_name(fn):
    with gzip.open(fn, 'rb') as f:
        file_content = f.read()
        return file_content

artifact = wandb.Artifact(FILE, type='dataset')
artifact.add_file(file_name(FILEPATH))
wandb.run.log_artifact(artifact) 