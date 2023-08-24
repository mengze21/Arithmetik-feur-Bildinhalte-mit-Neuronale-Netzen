import numpy as np
import matplotlib.pyplot as plt
import save_data_par as pa
from datetime import datetime
import os


def check_array(array):
    print(array)
    print(array.ndim)
    print(array.shape)


def save_fields(data_set, samples, pic_num):
    """save the selected sample data, 3 pictures will be saved in different files"""
    i = samples
    for j in range(pic_num):
        fields = data_set[i, j, :, :, 0]
        np.savetxt('data_set/fields_' + str(i) + '_' + str(j) + '.csv', fields, delimiter=',')


def save_dataset(data_set, file_name='samples'):
    """reshape the data into 2D array and then save it as csv file"""
    today = datetime.today().strftime('%Y_%m_%d')
    data_set_reshaped = data_set.reshape(data_set.shape[0], -1)
    np.savetxt(file_name + '_' + today + '.csv', data_set_reshaped, delimiter=',')


def save_dataset_npz(data_set, file_name='samples_' + datetime.today().strftime('%Y_%m_%d')):
    """save the data_set as compressed format"""
    file = file_name
    n = 1
    while os.path.exists(f'{file}.npz'):
        file = file + f'_{n}'
        n = n + 1
    np.savez_compressed(f'{file}.npz', data_set)


def load_dataset(filename, sample_num):
    """load data from csv file and reshape to 5D"""
    load_data = np.loadtxt(filename, delimiter=',')
    original_data = load_data.reshape((sample_num, 3, 32, 32, 1))
    return original_data


def load_dataset_npz(file):
    """load data from compressed format"""
    npz_data = np.load(file)
    data_set = npz_data['arr_0']
    return data_set


def pic_plt(data, sample_num, save=False):
    """plot the 3 pictures from given sample"""
    dict_val = np.load('data_set.npy', allow_pickle=True)
    data_par = dict_val.item()
    samples = data_par.get('samples')
    if sample_num > samples:
        print('the given number is bigger than the number of samples')
    else:
        fig, ax = plt.subplots(1, 3, figsize=(8, 3),constrained_layout=True)
        for i in range(len(ax)):
            ax[i].imshow(data[sample_num - 1, i, :, :, 0], origin='lower')
            ax[i].set_title('Bild ' + str(i + 1))
            ax[i].axis('off')
        # fig.suptitle('generated pictures in sample ' + str(sample_num), fontsize=14)
    if save:
        #plt.savefig('data_set/sample_' + str(sample_num) + '.pdf')
        plt.savefig('for_ausarbeitung/sample_' + str(sample_num) + '.svg')
        #plt.savefig('data_set/sample_' + str(sample_num) + '.png')
    plt.show()


def pic_plt_sel(data, sample_low, sample_high, save=False):
    """
    show multiple samples
    Args:
        data:
        sample_low: start sample
        sample_high: end sample
        save: default False, when True the sample pictures will save in 3 format
    """
    dict_val = np.load('data_set.npy', allow_pickle=True)
    data_par = dict_val.item()
    samples = data_par.get('samples')
    if sample_low > samples or sample_high > samples:
        print('the given number is out of range')
    else:
        for j in range(sample_high - sample_low + 1):
            fig, ax = plt.subplots(1, 3, figsize=(8, 3))
            for i in range(len(ax)):
                ax[i].imshow(data[j + sample_low - 1, i, :, :, 0], origin='lower', cmap='binary', vmin=0, vmax=1)
                ax[i].set_title('picture ' + str(i + 1))
            fig.suptitle('generated pictures in sample ' + str(j + sample_low), fontsize=14)
        if save:
            plt.savefig('data_set/sample_' + str(j + sample_low) + '.pdf')
            plt.savefig('data_set/sample_' + str(j + sample_low) + '.svg')
            plt.savefig('data_set/sample_' + str(j + sample_low) + '.png')
        plt.show()


def pic_plt_Multi(data, sample_low, sample_high, file, save=False):
    """
    show multiple samples
    Args:
        data:
        sample_low: start sample
        sample_high: end sample
        save: default False, when True the sample pictures will save in 3 format
    """
    dict_val = np.load(file, allow_pickle=True)
    data_par = dict_val.item()
    samples = data_par.get('samples')
    img_num = data_par.get('img_num')
    if sample_low > samples or sample_high > samples:
        print('the given number is out of range')
    else:
        for j in range(sample_high - sample_low + 1):
            fig, ax = plt.subplots(1, img_num, figsize=(8, 3))
            for i in range(len(ax)):
                ax[i].imshow(data[j + sample_low - 1, i, :, :, 0], origin='lower')
                ax[i].set_title('image ' + str(i + 1))
                ax[i].axis('off')
            fig.suptitle('generated images in sample ' + str(j + sample_low), fontsize=14)
        if save:
            plt.savefig('data_set/sample_' + str(j + sample_low) + '.pdf')
            plt.savefig('data_set/sample_' + str(j + sample_low) + '.svg')
            plt.savefig('data_set/sample_' + str(j + sample_low) + '.png')
        plt.show()


def check_overlap(array1, array2):
    array_1 = np.ones((32, 32))
    a1 = array1
    a2 = array2
    overlap = False
    a3 = np.add(a1, a2)
    result = np.greater(a3, array_1)
    if True in result:
        overlap = True
    return overlap


def check_overlap_1argu(array):
    """the checked picture must be 32*32"""
    dict_val = np.load('data_set.npy', allow_pickle=True)
    data_par = dict_val.item()
    pixel = data_par.get('pixel')
    array_1 = np.ones((pixel, pixel))
    checked_array = array
    overlap = False
    result = np.greater(checked_array, array_1)
    if True in result:
        overlap = True
    return overlap


def check_overlap_64x64(array):
    """the checked picture must be 32*32"""
    dict_val = np.load('data_set/data_set_4_(64x64).npy', allow_pickle=True)
    data_par = dict_val.item()
    pixel = data_par.get('pixel')
    array_1 = np.ones((pixel, pixel))
    checked_array = array
    overlap = False
    result = np.greater(checked_array, array_1)
    if True in result:
        overlap = True
    return overlap

def check_overlap_Multi(array, pix):
    """the checked picture must be 32*32"""
    pixel = pix
    array_1 = np.ones((pixel, pixel))
    checked_array = array
    overlap = False
    result = np.greater(checked_array, array_1)
    if True in result:
        overlap = True
    return overlap

def equal_check(data, samples_num):
    counter = 0
    index_list = []
    data_tocheck = data
    samples = samples_num
    for i in range(samples):
        data_compare = data_tocheck[i, 2, :, :, 0]
        for j in range(samples - i - 1):
            equal = np.array_equal(data_compare, data_tocheck[i + j + 1, 2, :, :, 0])
            if equal:
                counter = counter + 1
                index1 = i
                index2 = i + j + 1
                index_tup = (index1, index2)
                index_list.append(index_tup)
    print(f"there are {counter} same samples and the index of same samples are {index_list}")


def show_image(sample_num, img_data1, img_data2):
    """
    without file path, must load the data_set first
    # load data_set
    data_set = hf.load_dataset_npz('data_set/10000samples_3_8_2ell.npz')
    #data_set = hf.load_dataset_npz('data_set/10000samples_3_8_bgnoise(03005).npz')
    x_train = data_set[:8000, :4]
    y_train = data_set[:8000]
    x_test = data_set[8000:, :4]
    y_test = data_set[8000:]
    Args:
        img_data2: data_set from loaded file
        img_data1: predictions from autoencoder
        sample_num: the sample to show

    Returns: images

    """
    data_set = img_data1
    x_test = data_set[8000:, :4]
    predictions = img_data2
    img_num = len(data_set[0])
    fig1, ax1 = plt.subplots(1, img_num, figsize=(10, 8), constrained_layout=True)
    fig2, ax2 = plt.subplots(1, img_num, figsize=(10, 8), constrained_layout=True)
    fig1.suptitle('Original image', y=0.75)
    fig2.suptitle('reconstructed image', y=0.75)
    for i in range(img_num):
        ax1[i].imshow(x_test[sample_num - 1, i], origin='lower')
        ax1[i].set_title(f'image {i+1}')
        ax1[i].axis('off')
        ax2[i].imshow(predictions[i][sample_num-1], origin='lower')
        ax2[i].set_title(f'image {i+1}')
        ax2[i].axis('off')

