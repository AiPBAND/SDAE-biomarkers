import tempfile, os, yaml
os.environ("TF_XLA_FLAGS") = "--tf_xla_enable_xla_devices"
import tensorflow as tf
tf.autograph.set_verbosity(0, True)
import tensorflow as tf
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
   print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled()) 
from ludwig.api import kfold_cross_validate, LudwigModel
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import glob

RUN_CONFIG = "data/211331417.yaml"
OUT = "results"
with open(RUN_CONFIG) as fin:
  config = yaml.full_load(fin)

for l in config["layers"]:
  with open("data/"+l) as fin:
    config_l = yaml.full_load(fin)

  train = "data/"+l.split("-")[0] + "-" + l.split("-")[1].replace("config.yaml", "train.csv")
  test = "data/"+l.split("-")[0] + "-" + l.split("-")[1].replace("config.yaml", "test.csv")
  (stats, splits) = kfold_cross_validate(config=config_l, dataset=train, output_directory=OUT, num_folds=5)

print(stats['overall'])

model = LudwigModel(config=config_l, logging_level=logging.ERROR)
training_stats = model.train(training_set=train, output_directory="results")
test_stats, recon, _ = model.evaluate(dataset=test)

print(test_stats)

a = plt.axes(aspect='equal')
sns.scatterplot(test.values, recon.values, s=300)
plt.xlabel('True Values')
plt.ylabel('Predictions')

#plt.xlim(lims)
#plt.ylim(lims)