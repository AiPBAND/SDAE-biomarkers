import numpy as np
import glob
from scipy.sparse import *
import os

file_names = glob.glob('input/GSE*.csv')
probe_ids = set()
sample_ids = set()

start_col = 1
end_col = -2
gene_id = -2

def ss(l):
    return l.strip().replace('"', '').split(',')


for file in file_names:
    with open(file, 'r') as fin:
        samples = ss(fin.readline())[start_col:end_col]
        sample_ids.update(samples)
        for line in fin:
            line = ss(line)
            probe_ids.add(line[gene_id])

probe_ids = sorted(list(probe_ids))
sample_ids = sorted(list(sample_ids))

num_probes = len(probe_ids)
num_samples = len(sample_ids)

probes_i = {p: i for i, p in enumerate(probe_ids)}
samples_i ={s: i for i, s in enumerate(sample_ids)}

with open('output_parse/samples.csv', 'w+') as fout:
    for s in samples_i:
        fout.write(s+','+str(samples_i[s])+'\n')

with open('output_parse/probes.csv', 'w+') as fout:
    for p in probes_i:
        fout.write(p+','+str(probes_i[p])+'\n')

data_matrix = np.zeros((num_probes, num_samples))
categorical = np.empty((num_samples, 2))
studies_i = {}

for file in file_names:
    print('Processing file: '+file)
    control_flag = 1 if 'control' in file else 0
    study = file.split('_')[0]

    if os.path.exists("output_parse/studies.csv"):
        os.remove("output_parse/studies.csv")

    if study not in studies_i:
        studies_i[study] = len(studies_i)
        study_i = studies_i[study]
        with open('output_parse/studies.csv', 'a+') as fout:
            fout.write(','.join([study, str(study_i)]) + '\n')
    else:
        study_i = studies_i[study]

    with open(file, 'r') as fin:
        samples = ss(fin.readline())[start_col:end_col]

        for s in samples:
            categorical[samples_i[s]] = [study_i, control_flag]

        for line in fin:
            line = ss(line)
            probe = line[gene_id]
            values = line[start_col:end_col]
            for s, v in zip(samples, values):
                data_matrix[probes_i[probe], samples_i[s]] = v

data_matrix = data_matrix.transpose()

np.save('output_parse/data', data_matrix)
np.save('output_parse/categorical', categorical)

