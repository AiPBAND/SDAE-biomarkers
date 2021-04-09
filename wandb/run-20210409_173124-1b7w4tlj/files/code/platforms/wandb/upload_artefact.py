#!/usr/bin/env python
#log_artifact.py
import wandb
wandb.login()
wandb.init(project='SDAE-biomarkers', entity='aipband')
artifact = wandb.Artifact('geo', type='dataset')
artifact.add_file("gene_exp_MA.csv")
run.log_artifact(artifact) 