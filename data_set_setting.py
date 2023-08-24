import numpy as np


def set_data_par(file_path, sample_num=10000, img_num=3, pixel=32, rgb=1, overlap_ratio=0.3, size_min=3,
                 size_max=8, bg_noise=False, overlapped=False, random_input_img=False):

    data_par = {'samples': sample_num, 'img_num': img_num, 'pixel': pixel, 'RGB': rgb,
                'overlap_ratio': overlap_ratio, 'size_min': size_min, 'size_max': size_max,
                'bg_noise': bg_noise, 'overlapped': overlapped, 'random_input_img': random_input_img}
    fp = file_path
    np.save(fp, data_par)


set_data_par('data_set/loops_test_64pix_3-8', sample_num=10, img_num=9, pixel=64, size_min=3, size_max=8)
print("finished")