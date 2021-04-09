#!/usr/bin/env python
#log_artifact.py
import wandb
wandb.login()
wandb.init(project='SDAE-biomarkers', entity='aipband')
artifact = wandb.Artifact('geo', type='dataset')
with open("./gene_exp_MA.csv") as sheet:
    artifact.add_file(sheet)
    run.log_artifact(artifact) 