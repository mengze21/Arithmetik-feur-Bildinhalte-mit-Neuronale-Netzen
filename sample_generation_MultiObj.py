import math
import numpy as np
import nnlib.mo_renderer_new as mrn
import helpful_functions as hf


# get the data parameters
def get_data_parameters(f_path):
    # dictionary stores all parameters for date_set
    file_path = f_path
    dict_val = np.load(file_path, allow_pickle=True)
    data_par = dict_val.item()
    # samples = 5
    samples = data_par.get('samples')
    pixel = data_par.get('pixel')
    img_num = data_par.get('img_num')
    # create the data structure
    # samples, picture pro samples, pixel, RGB
    empty_data = np.zeros((samples, img_num, pixel, pixel, 1))
    return empty_data, samples, img_num, pixel


def draw_samples(data_set, samples, img_num, pixel, set_overlap_ratio=0.5, size_min=3, size_max=8,
                 bg_noise=False, overlapped=False, random_input_img=False):
    """

    Args:
        set_overlap_ratio: default overlap ratio is 0.5
        size_min: minimum length in pixel
        size_max: maximum length in pixel
        bg_noise: image with or without background noise
        overlapped: set if overlap is allowed
        random_input_img: set if the object in the image is randomly chosen

    """

    # define array_access function
    fields = np.zeros((pixel, pixel))

    def array_access(value, x, y):
        """modified the function"""
        value = np.clip(value, 0.0, 1.0)
        fields[x, y] = value

    # create boundary boxes
    def draw_boundary(pos_x, pos_y, length):
        box_field = np.zeros((pixel, pixel))
        x = round(pos_x)
        y = round(pos_y)
        l = round(length)
        for i in range(l + 1):
            for j in range(l + 1):
                box_field[x + i, y + j] = 1
                box_field[x + i, y - j] = 1
                box_field[x - i, y + j] = 1
                box_field[x - i, y - j] = 1
        return box_field

    # overlap ratio deactivate
    def check_overlap_ratio(boxs_set, boxs_sum):
        area_size = []
        # calculate the area size of boxs in box_set and save it in area_box
        for i in range(img_num):
            area_size.append(np.count_nonzero(boxs_set[i]))
            # sum of all boxs in one variable

        # take the smaller area as base area
        base_size = np.amin(area_size)
        # calculate overlapped area size
        ovlap_size = np.count_nonzero(boxs_sum > 1)
        # calculate the overlapped ratio
        ratio = ovlap_size / base_size
        return ratio

    def image_para_generator(len_min=size_min, len_max=size_max):
        # generate the random parameters
        rng = np.random.default_rng()
        length_1 = rng.integers(low=len_min, high=len_max)
        length_2 = rng.integers(low=len_min, high=len_max)
        max_length = np.maximum(length_1, length_2)
        # can be improved by position = sqrt(2*power(max_length))
        position_x = rng.integers(low=max_length + 5, high=pixel - max_length - 5)
        position_y = rng.integers(low=max_length + 5, high=pixel - max_length - 5)
        rot_angle = rng.uniform(low=-math.pi / 2, high=math.pi / 2)
        return length_1, length_2, position_x, position_y, rot_angle, max_length

    def draw_ell(length1, length2, position_x, position_y, img_angl):
        # register Ellipse and draw

        ell = mrn.Ellipse(pos_x=position_x, pos_y=position_y, len_1=length1, len_2=length2, angle=img_angl,
                          additive=1)
        fs = mrn.FieldScope()
        fs.register_shape(ell)
        ell_aa = array_access
        fs.draw(ell_aa)
        # save the fields in pic1
        # eliminate boundary mask by multiplying
        # img = np.multiply(fields, data_set[i, 0, :, :, 0])
        return fields

    def draw_rec(length1, length2, position_x, position_y, img_angl):
        # register Ellipse 2 and draw

        rec = mrn.Rectangle(pos_x=position_x, pos_y=position_y, len_1=length1, len_2=length2, angle=img_angl,
                            additive=1)
        fs = mrn.FieldScope()
        fs.register_shape(rec)
        rec_aa = array_access
        fs.draw(rec_aa)
        # save the fields in pic1
        # eliminate boundary mask by multiplying
        # img = np.multiply(fields, data_set[i, 2, :, :, 0])
        return fields

    # draw samples
    for i in range(samples):
        counter = 0
        overlap = True
        arr_all = np.zeros((pixel, pixel))
        if not overlapped:
            # validate random parameters to make sure the boundary of two objects are not overlapped
            # the minimal distance of two objects is 4 Pixels
            while overlap:
                len_1 = []
                len_2 = []
                pos_x = []
                pos_y = []
                angle = []
                len_max = []
                arr_box = np.zeros((pixel, pixel))
                # arr_all = np.zeros((pixel, pixel))
                # reset data_set[i,:,:,:,0] to zero
                data_set[i, :, :, :, 0] = np.zeros((img_num, pixel, pixel))
                # random parameters for input image i
                for j in range(img_num - 1):
                    # generate the parameters for image
                    # len_1[j], len_2[j], pos_x[j], pos_y[j], angle[j], len_max[j] = image_para_generator()
                    item = image_para_generator()
                    len_1.append(item[0])
                    len_2.append(item[1])
                    pos_x.append(item[2])
                    pos_y.append(item[3])
                    angle.append(item[4])
                    len_max.append(item[5])
                    # draw boundary box
                    # create a dynamic boundary box in data_set[i, 0, :, :, 0]
                    # the length is maximum of len_1 and len_2
                    # "length=len_max_ell + 2" make sure the ellipse is always in boundary box
                    data_set[i, j, :, :, 0] = draw_boundary(pos_x=pos_x[j], pos_y=pos_y[j], length=len_max[j] + 2)
                    arr_box = np.add(arr_box, data_set[i, j, :, :, 0])
                    # sum all image in last index
                    # data_set[i, img_num - 1, :, :, 0] = np.add(data_set[i, img_num - 1, :, :, 0], data_set[i, j, :, :, 0])
                    # check = data_set[0, 4, :, :, 0]
                data_set[i, img_num - 1, :, :, 0] = arr_box
                check = data_set[i, img_num - 1, :, :, 0]
                # check the overlap between the boundary of ellipses and rectangle
                overlap = hf.check_overlap_Multi(
                    data_set[i, img_num - 1, :, :, 0], pixel)  # data_set.npy is loaded, can cause bug
                counter = counter + 1
            print(f"tried {counter} times for sample {i + 1}")
        else:
            # validate random parameters to make sure the boundary of two objects are not overlapped
            # the minimal distance of two objects is 4 Pixels
            while overlap:
                len_1 = []
                len_2 = []
                pos_x = []
                pos_y = []
                angle = []
                len_max = []
                arr_box = np.zeros((pixel, pixel))
                # arr_all = np.zeros((pixel, pixel))
                # reset data_set[i,:,:,:,0] to zero
                data_set[i, :, :, :, 0] = np.zeros((img_num, pixel, pixel))
                # random parameters for input image i
                for j in range(img_num - 1):
                    # generate the parameters for image
                    # len_1[j], len_2[j], pos_x[j], pos_y[j], angle[j], len_max[j] = image_para_generator()
                    item = image_para_generator()
                    len_1.append(item[0])
                    len_2.append(item[1])
                    pos_x.append(item[2])
                    pos_y.append(item[3])
                    angle.append(item[4])
                    len_max.append(item[5])
                    # draw boundary box
                    # create a dynamic boundary box in data_set[i, 0, :, :, 0]
                    # the length is maximum of len_1 and len_2
                    # "length=len_max_ell + 2" make sure the ellipse is always in boundary box
                    data_set[i, j, :, :, 0] = draw_boundary(pos_x=pos_x[j], pos_y=pos_y[j], length=len_max[j] + 2)
                    arr_box = np.add(arr_box, data_set[i, j, :, :, 0])
                    # sum all image in last index
                    # data_set[i, img_num - 1, :, :, 0] = np.add(data_set[i, img_num - 1, :, :, 0], data_set[i, j, :, :, 0])
                    # check = data_set[0, 4, :, :, 0]
                data_set[i, img_num - 1, :, :, 0] = arr_box
                bound_box = data_set[i, 0:img_num]
                check = data_set[i, img_num - 1, :, :, 0]

                # check the overlap ratio between two boxes
                overlap_ratio = check_overlap_ratio(bound_box, arr_box)
                if overlap_ratio < set_overlap_ratio:
                    overlap = False
                counter = counter + 1
                print(counter)
            print(f"tried {counter} times for sample {i + 1}")
        # draw sample i using validated random parameters
        if not random_input_img:
            # draw ellipse and rectangle one by one
            index = 0
            while index < img_num:
                # initial fields
                fields = np.zeros((pixel, pixel))
                draw_ell(length1=len_1[index], length2=len_2[index], position_x=pos_x[index],
                         position_y=pos_y[index], img_angl=angle[index])
                # eliminate boundary mask by multiplying
                data_set[i, index, :, :, 0] = np.multiply(fields, data_set[i, index, :, :, 0])
                # sum all image in arr_all
                arr_all = np.add(arr_all, data_set[i, index, :, :, 0])
                index = index + 1
                # check index value
                if index == img_num - 1:
                    break
                # initial fields
                fields = np.zeros((pixel, pixel))
                draw_rec(length1=len_1[index], length2=len_2[index], position_x=pos_x[index],
                         position_y=pos_y[index], img_angl=angle[index])
                # eliminate boundary mask by multiplying
                data_set[i, index, :, :, 0] = np.multiply(fields, data_set[i, index, :, :, 0])
                # sum all image in arr_all
                arr_all = np.add(arr_all, data_set[i, index, :, :, 0])
                index = index + 1
                # check index value
                if index == img_num - 1:
                    break


        else:
            index = 0
            # get random choice for drawing
            # 1 for ellipse 0 for rectangle
            rng = np.random.default_rng()
            choice = rng.choice([0, 1], img_num)
            for index in range(img_num - 1):
                if choice[index] == 0:
                    # initial fields
                    fields = np.zeros((pixel, pixel))
                    # draw ellipse
                    draw_ell(length1=len_1[index], length2=len_2[index], position_x=pos_x[index],
                             position_y=pos_y[index], img_angl=angle[index])
                    # eliminate boundary mask by multiplying
                    data_set[i, index, :, :, 0] = np.multiply(fields, data_set[i, index, :, :, 0])
                    # sum all image in arr_all
                    arr_all = np.add(arr_all, data_set[i, index, :, :, 0])
                if choice[index] == 1:
                    # initial fields
                    fields = np.zeros((pixel, pixel))
                    # draw ellipse
                    draw_rec(length1=len_1[index], length2=len_2[index], position_x=pos_x[index],
                             position_y=pos_y[index], img_angl=angle[index])
                    # eliminate boundary mask by multiplying
                    data_set[i, index, :, :, 0] = np.multiply(fields, data_set[i, index, :, :, 0])
                    # sum all image in arr_all
                    arr_all = np.add(arr_all, data_set[i, index, :, :, 0])
            #print("random_input_img")

        if bg_noise:
            print("not implement")

        else:
            # make sure the values of overlapped area smaller than one
            arr_all[arr_all > 1] = 1
            data_set[i, img_num - 1, :, :, 0] = arr_all

    print("generation finished")


data_set, samples, img_num, pixel = get_data_parameters('data_set/2Inputs_64pix_3-8.npy')
print(f"the shape of data_set is {data_set.shape}")
print(f"samples is {samples}")
print(f"img_num is {img_num}")
print(f"pixel is {pixel}")

draw_samples(data_set, samples, img_num, pixel, size_min=3, size_max=8,
             set_overlap_ratio=0.8, bg_noise=False, random_input_img=True, overlapped=True)
print(data_set.shape)
hf.pic_plt_Multi(data=data_set, sample_low=1, sample_high=5, file='data_set/2Inputs_64pix_3-8.npy')

hf.save_dataset_npz(data_set=data_set, file_name='data_set/2Inputs_64pix_3-8_ratio08')
print("data saved")
