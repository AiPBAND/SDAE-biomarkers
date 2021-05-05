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
from random import random

os.environ["WANDB_MODE"] = "offline"

wandb.login()

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
parser.add_argument(
    "--folds",
    default=5,
    dest="FOLDS",
    type=int,
    help="Number of cross-validation folds.",
)

config = wandb.config
args = parser.parse_args()
wandb.config.update(args)
wandb.tensorboard.patch(save=True, tensorboardX=True)
tensorboard_logs = "./out/ts_logs"


for idx, num_hidden in enumerate(args.N_NODES):
    for idy in range(args.FOLDS):
        cv_group_id = random.choice("ABCDE123456789",6)
        cv_group_name = "CVgroup_{}".format(cv_group_name)

        with wandb.init(project='SDAE-biomarkers', 
                        entity='aipband', 
                        job_type="training_layer_{}".format(idx), 
                        group=cv_group_name) as run:

            if idx == 0:
                artifact = run.use_artifact('data_splits:latest')
                artifact_dir = artifact.download()  
                data = np.load(split_data.npy)
                X_out, X_text_out, X_val = data["train"][idy], data["test"][idy], data["validation"]
            else:
                X_out, X_test_out = X, X_test

            print("Training layer {} from group {} with {} hidden nodes..".format(idx, cv_group_name ,num_hidden))
            encoder = Autoencoder(x_train_out.shape[1], num_hidden, tensorboard_logs).get()

            encoder.compile(loss=loss_fn,
                            metrics=[tf.keras.metrics.KLDivergence(),tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError()]
                            optimizer=optimizer)

            early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=2)

            history = encoder.fit(
                X_out,
                X_out,
                callbacks=[early_stop, self.tsb, WandbCallback()],
                batch_size=args.BATCH_SIZE,
                num_epochs=args.EPOCHS,
                verbose=args.VERBOSITY,
                patience=args.PATIENCE,
                validation_data=(x_val, x_val)
            )
            wandb.log(history)

            result = encoder.evaluate(X_test)

            wandb.summary({'metrics': dict(zip(encoder.metrics_names, result)),
                            'layer': encoder.encoder_layer.get_weights(),
                            'name': encoder.name,
                            'fold': idy
                            'cv_group_id': cv_group_name,
                            'mse':result 
            })

            model_path = os.path.join("encoders", model.name)
            encoder.encoder_model.save(model_path)
            wandb.save(model_path)
            
            embeded_train = encoder.model.predict(X_out)
            embeded_test = encoder.model.predict(X_test_out)
            embeded_val = encoder.model.predict(x_val)
            

            artifact = wandb.Artifact("embeded", type="dataset")

            np.save("temp_file", data)
            artifact.add_file("temp_file.npy", name="split_data", is_tmp=True)
            wandb.run.log_artifact(artifact)
