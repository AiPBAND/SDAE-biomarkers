# Import comet_ml at the top of your file
import tensorflow as tf
import tensorrt as trt
import argparse
import pandas as pd
import numpy as np
from models import Autoencoder
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import normalize
import os
from decouple import config as env_config
import wandb
from wandb.keras import WandbCallback

wandb.login()
wandb.init(project='SDAE-biomarkers', entity='aipband',job_type="model-training")
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-path",
    type=str,
    dest="data_path",
    help="Path to the CSV input data. First row headers. First column IDs",
    default="uni_exp_data_MA_7.csv",
)

parser.add_argument(
    "--num-nodes",
    default=[2000, 1000, 500],
    dest="N_NODES",
    metavar="N",
    type=int,
    nargs="+",
    help="Number of nodes in each layer.",
)
parser.add_argument(
    "--dropout",
    default=[0.1],
    dest="DROPOUT",
    type=int,
    nargs="+",
    help="Number of nodes in each layer.",
)
parser.add_argument(
    "--batch",
    default=5,
    dest="BATCH_SIZE",
    type=int,
    help="Number of samples per batch.",
)
parser.add_argument(
    "--epochs", default=10, dest="EPOCHS", type=int, help="Number of epochs."
)
parser.add_argument(
    "--test",
    default=0.2,
    dest="TEST_RATIO",
    type=float,
    help="Ratio of samples kept out for testing.",
)
parser.add_argument(
    "--verbose",
    default=1,
    dest="VERBOSITY",
    type=int,
    choices=[0, 1, 2],
    help="Verbosity level: 0 None, 1 Info, 2 All",
)
parser.add_argument(
    "--tolerance",
    default=3,
    dest="PATIENCE",
    type=int,
    help="Tolenrance to the rate of improvment between each batch. Low values terminate quicker.",
)

config = wandb.config
args = parser.parse_args()
wandb.config.update(args)

run = wandb.init()
wandb.tensorboard.patch(save=True, tensorboardX=True) 
tensorboard_logs = "./out/ts_logs"

artifact = run.use_artifact('data_splits:latest')
artifact_dir = artifact.download()

data = np.load(split_data.npy)

x_train, x_test,  x_val = data["train"], data["test"], data["validation"]

for X, y, val in zip(x_train, x_test):
 
    X_out, y_out = X, y
    for idx, num_hidden in enumerate(args.N_NODES):

        print("Training layer {} with {} hidden nodes..".format(idx, num_hidden))
        encoder = Autoencoder(x_train_out.shape[1], num_hidden, tensorboard_logs)

        recon_mse = encoder.fit(
            X,
            y,
            batch_size=args.BATCH_SIZE,
            num_epochs=args.EPOCHS,
            verbose=args.VERBOSITY,
            patience=args.PATIENCE,
        )
        
        recon_train = recon_mse[0]
        recon_test = recon_mse[1]
        
        artifact.add_file(recon_train, name="epoch_recon_err_train")
        artifact.add_file(recon_test, name="epoch_recon_err_train")
        wandb.run.log_artifact(artifact)    

        x_train_out = encoder.encoder_model.predict(x_train_out)
        x_test_out = encoder.encoder_model.predict(x_test_out)

        print("Training loss for layer {}: {} ".format(idx, recon_mse[0]))
        print("Testing loss for layer {}: {} ".format(idx, recon_mse[1]))

        experiment.log_metrics({"trained_layer_mse": recon_mse[0], "test_layer_mse": recon_mse[1]})

    model_path = os.path.join("encoders", "model_{}_{}".format(idx, num_hidden))
    encoder.encoder_model.save(model_path)

    wandb.save("mymodel.h5")
    
    
