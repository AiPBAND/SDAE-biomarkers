import tempfile, os, yaml
from ludwig.api import kfold_cross_validate, LudwigModel
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import glob

RUN_ID = None or sorted([x.split("~")[0] for x in glob.glob("data/*")])[-1]
OUT = "results"

config = yaml.load(RUN_ID)
print(config)

for l in config["splits"]: 
  train ="run_id~"+l+"test.csv"
  test ="run_id~"+l+"train.csv"
  (stats, splits) = kfold_cross_validate(config=config, dataset=train, output_directory=OUT)

print(stats['overall'])

model = LudwigModel(config=config, logging_level=logging.ERROR)
training_stats = model.train(training_set=train, output_directory="results")
test_stats, recon, _ = model.evaluate(dataset=test)

print(test_stats)

a = plt.axes(aspect='equal')
sns.scatterplot(test.values, recon.values, s=300)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(config["splits"])
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
plt.plot()