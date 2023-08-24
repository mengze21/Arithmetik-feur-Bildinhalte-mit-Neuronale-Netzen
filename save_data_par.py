import numpy as np


# save the parameters for date set generation
def set_data_par(sample_num=10000, img_num=4, pixel=32, rgb=1):
    data_par = {'samples': sample_num, 'img_num': img_num, 'pixel': pixel, 'RGB': rgb}
    np.save('data_set/data_set_4_.npy', data_par)


set_data_par()
