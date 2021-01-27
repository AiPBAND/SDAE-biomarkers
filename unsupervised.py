# Import comet_ml at the top of your file
from comet_ml import Experiment

import tensorflow as tf
import argparse
import pandas as pd
import numpy as np
from models import Autoencoder
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import normalize
import os
from config import config



parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-path",
    type=str,
    dest="data_path",
    help="Path to the CSV input data. First row headers. First column IDs",
    default="/data/GEO_features.csv",
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
args = parser.parse_args()

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "num-nodes": args.N_NODES,
    "dropout": args.DROPOUT,
    "batch_size": args.BATCH_SIZE,
    "epochs": args.EPOCHS,
    "test_ratio": args.TEST_RATIO,
    "tolerance": args.PATIENCE,
}

dataframe = pd.read_csv("./data/GEO_features.csv", index_col=0)
data = dataframe.values
data = normalize(data)

rs = ShuffleSplit(n_splits=1, test_size=args.TEST_RATIO, random_state=0)
split_itterator = rs.split(data)
i_train, i_test = next(split_itterator)
train_path = os.path.join("./out", "train_indices.npy")
test_path = os.path.join("./out", "test_indices.npy")
np.save(train_path, i_train)
np.save(test_path, i_test)

x_train, x_test = data[i_train], data[i_test]

tensorboard_logs = "./out/ts_logs"


x_train_out, x_test_out = x_train, x_test
for idx, num_hidden in enumerate(args.N_NODES):
    experiment = Experiment(**config)
    experiment.log_parameters(hyper_params)

    with experiment.train():
        print("Training layer {} with {} hidden nodes..".format(idx, num_hidden))
        encoder = Autoencoder(x_train_out.shape[1], num_hidden, tensorboard_logs)

        recon_mse = encoder.fit(
            x_train_out,
            x_test_out,
            batch_size=args.BATCH_SIZE,
            num_epochs=args.EPOCHS,
            verbose=args.VERBOSITY,
            patience=args.PATIENCE,
        )

    with experiment.test():
        x_train_out = encoder.encoder_model.predict(x_train_out)
        x_test_out = encoder.encoder_model.predict(x_test_out)

        print("Training losss for layer {}: {} ".format(idx, recon_mse[0]))
        print("Testing loss for layer {}: {} ".format(idx, recon_mse[1]))

        experiment.log_metrics({"trained_layer_mse": recon_mse[0], "test_layer_mse": recon_mse[1]})

    model_path = os.path.join("encoders", "model_{}_{}".format(idx, num_hidden))
    encoder.encoder_model.save(model_path)
