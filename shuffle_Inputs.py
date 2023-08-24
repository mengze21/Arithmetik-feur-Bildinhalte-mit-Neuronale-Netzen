import numpy as np
import tensorflow as tf
import helpful_functions as hf
import matplotlib.pyplot as plt

# load and shuffle the data_set
data_to_change = hf.load_dataset_npz('data_set/10000samples_3_8_ratio03.npz')
data_set = np.zeros((10000, 3, 32, 32, 1))
for i in range(10000):
    rng = np.random.default_rng()
    random_num = rng.choice([0, 1])
    if random_num == 1:
        data_set[i, 0] = data_to_change[i, 1]
        data_set[i, 1] = data_to_change[i, 0]
    else:
        data_set[i, 0] = data_to_change[i, 0]
        data_set[i, 1] = data_to_change[i, 1]
    data_set[i, 2] = data_to_change[i, 2]

hf.pic_plt_sel(data=data_set, sample_low=1, sample_high=5)
hf.save_dataset_npz(data_set=data_set, file_name='data_set/10000samples_3_8_ratio03_shuffle.npz')

#%%
