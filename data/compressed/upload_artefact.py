
import wandb, os
import gzip
import shutil

PROJECT='SDAE-biomarkers'
ENTITY="aipband"
FILE="gene_exp_MA.csv.gz"
FILEPATH=os.path.join("data\\compressed", FILE)

wandb.login()
wandb.init(project=PROJECT, entity=ENTITY)

with gzip.open(FILEPATH, 'rb') as f_in:
    with open(FILE[:-3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

artifact = wandb.Artifact('input_tables', type='dataset')
artifact.add_file(FILE[:-3])
wandb.run.log_artifact(artifact) 