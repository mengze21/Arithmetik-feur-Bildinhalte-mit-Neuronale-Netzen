import helpful_functions as hf
import numpy as np

data_set = hf.load_dataset_npz('data_set/2Inputs_64pix_3-8_ratio05_1.npz')
hf.pic_plt_Multi(data=data_set, sample_low=1117, sample_high=1123, file='data_set/2Inputs_64pix_3-8.npy')

# count how many samples are overlapped
counter = 0
index = []
for i in range(10000):
    check = 0
    sum = np.zeros((64,64,1))
    sum = np.add(data_set[i,0], data_set[i,1])
    check = np.count_nonzero(sum>1)
    if check > 1:
        index.append(i)
        counter = counter + 1
print(f"there are {counter} samples overlapped")
print(f"the overlapped samples are {index}")