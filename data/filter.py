import numpy as np


NZ_MIN = 0.9

data = np.load('output_parse/data.npy')
print(data.shape)
data = data.transpose()
print(data.shape)

data_keep = []
for i, row in enumerate(data):
    nz = np.count_nonzero(row)
    if nz/len(row) >= NZ_MIN:
        data_keep.append(row)

data_keep = np.array(data_keep)
data_keep = data_keep.transpose()
print(data_keep.shape)

np.save('output_filter/data', data_keep)