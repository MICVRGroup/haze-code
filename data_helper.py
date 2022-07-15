import os
from PIL import Image
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
int

def gen_batch(x, y, batch_size=3):
    images, names = x[0], x[1]
    s_index, e_index, batches = 0, 0 + batch_size, len(y) // batch_size
    if batches * batch_size < len(y):
        batches += 1
    for i in range(batches):
        if e_index > len(y):
            e_index = len(y)
        batch_x = [images[s_index: e_index], names[s_index: e_index]]
        batch_y = y[s_index: e_index]
        s_index, e_index = e_index, e_index + batch_size
        yield batch_x, batch_y



class load_data():
    def __init__(self, dir='/share/home/zhangyujie/zyj/wumai/data2/lidu1/0and255',
                 date_begin='20100711',
                 date_end='20100713',
                 time_begin='0600',
                 time_end='2000',
                 time_step=3
                 ):
        """
        :param dir:
        :param date_begin:  wen jian jia kai shi de ri qi
        :param date_end: wen jian jia jie shu de ri qi
        :param time_begin: yi tian kai shi de shi jian
        :param time_end: yi tian de jie shu shi jian
        :param time_step: shi xu chang du
        """
        abs_dir = os.path.abspath(__file__)
        print("===",abs_dir)
        self.file_dir = os.path.join(abs_dir[:-14], dir)  # ./data/
        print(self.file_dir)
        self.date_begin = date_begin
        self.date_end = date_end
        self.time_begin = time_begin
        self.time_end = time_end
        self.time_step = time_step

    def read_one_image1(self, filename):
        isExit = os.path.isfile(filename)
        if isExit == False:
            return False
        img = Image.open(filename)
        img_arr = np.array(img, dtype=np.uint8)
        return img_arr, np.shape(img_arr)

    def read_one_image(self, filename):
        isExit = os.path.isfile(filename)
        if isExit == False:
            return False
        img = np.load(filename)
        img_arr = np.array(img, dtype=np.float32)
        #img_arr = img_arr1
        if len(img_arr.shape) != 3:
           img_arr = np.expand_dims(img_arr, axis=2)
        return img_arr, np.shape(img_arr)
    '''
    def visualization(self):
        dir = './data/20100711/CAMS_GTCI_CLOUD_20100711060000.jpg'
        image, shape = self.read_one_image(dir)
        print("ke shi hua yi ge ce shi tu pian :")
        print(dir, 'shape:', shape)
        image = image.transpose(2, 0, 1)
        r = Image.fromarray(image[0]).convert('L')
        g = Image.fromarray(image[1]).convert('L')
        b = Image.fromarray(image[2]).convert('L')
        image = Image.merge("RGB", (r, g, b))
        plt.imshow(image, cmap='bone')
        plt.tight_layout()
        plt.show()
        
    '''
    def read_images(self, file_dir, image_name_pre='quanp_'):
        print("-------------------------------------")
        print(f"load data in {file_dir}")
        day_begin = datetime.datetime.strptime(
            "%04i-%02i-%02i-%02i-%02i" % (int(self.date_begin[:4]),
                                          int(self.date_begin[4:6]),
                                          int(self.date_begin[6:8]),
                                          int(self.time_begin[:2]),
                                          int(self.time_begin[-2:])), "%Y-%m-%d-%H-%M")  # 2010-07-11 06:00:00
        day_end = datetime.datetime.strptime(
            "%04i-%02i-%02i-%02i-%02i" % (int(self.date_end[:4]),
                                          int(self.date_end[4:6]),
                                          int(self.date_end[6:8]),
                                          int(self.time_end[:2]),
                                          int(self.time_end[-2:])), "%Y-%m-%d-%H-%M")  # 2010-07-12 20:00:00
        temp = str(day_end - day_begin).split(' ')
        total_days = int(temp[0]) + 1
        total_hours = int(temp[2][:2])
        all_images = []
        num_images = 0
        time_begin = day_begin  # 2010-07-11 06:00:00
        image_names = []
        for i in range(total_days):  # bian li mei ge wen jian jia
            tmp = []
            tmp_name = []
            print("===============")
            print("duqutupian", time_begin.strftime("%Y%m%d%H%M")[:8])  # 201007110600
            for j in range(total_hours * 60):  # bian li mei ge wen jian jia zhong de tu pian
                time_str = time_begin.strftime("%Y%m%d%H%M")
                image_dir = os.path.join(file_dir, time_str[:8],
                                         image_name_pre + time_str + '.npy')
                image = self.read_one_image(image_dir)
                time_begin += datetime.timedelta(minutes=1)  # xia yi fen zhong
                if image == False:
                    continue
                print("cheng gong du qu tu pian", image_dir[-34:])
                tmp_name.append(image_dir[-18:-4])
                tmp.append(image[0])
                num_images += 1
            print("gai wen jian jia xia gong ji tu pian: {} zhang".format(len(tmp)))
            all_images.append(tmp)
            image_names.append(tmp_name)
            time_begin = day_begin + datetime.timedelta(days=1)  # xia yi tian

        return all_images, num_images, image_names

    def data_resample(self, ):
        data_x_dir = os.path.join(self.file_dir, 'train')
        images_data_x, num_data_x, image_names_data_x = self.read_images(data_x_dir)
        data_y_dir = os.path.join(self.file_dir, 'true')
        images_data_y, num_data_y, image_names_data_y = self.read_images(data_y_dir, 'p_')

        # dui yuan shi shu ju jin xing chong cai yang gou jian shu ju ji
        x_images, y, y_name = [], [], []
        for item_day in zip(images_data_x,images_data_y,image_names_data_y):
            image_x,image_y,image_name = item_day[0],item_day[1],item_day[2]
            s_idx, e_idx = 0, self.time_step
            while e_idx < len(image_x):
                x_images.append(image_x[s_idx:e_idx])
                y.append(image_y[e_idx])
                y_name.append(image_name[e_idx])
                s_idx += 1
                e_idx += 1
        print("chong cai yang hou shu ju ji wei ", len(x_images))
        x_images, y, y_name = np.array(x_images), np.array(y), np.array(y_name).reshape(-1, 1)
        x_min, x_max = x_images.min(), x_images.max()
        x_images = (x_images - x_min) / (x_max - x_min)
        y = (y - x_min) / (x_max - x_min)

        print(x_images.shape)#(9, 3, 920, 920, 3)
        print(y.shape)#(9, 800, 800)
        print(y_name.shape)
        print(y_name)
        shuffle_idx = np.arange(len(x_images))
        np.random.seed(2020)
        np.random.shuffle(shuffle_idx)
        x_images, y, y_name = x_images[shuffle_idx, :], y[shuffle_idx, :], y_name[shuffle_idx, :]

        return x_images, y, x_max, x_min, y_name

    def load_dataset(self, test_size=0.3):
        x_images, y, x_max, x_min, y_name = self.data_resample()
        y = np.array(y,dtype=np.float32)
        test_len = int(len(y) * test_size)
        images_train, y_train = x_images[:-test_len, :], y[:-test_len, :]
        images_test, y_test = x_images[-test_len:, :], y[-test_len:, :]
        y_train_name, y_test_name = y_name[:-test_len, :], y_name[-test_len:, :]
        print("========")
        print(y_train_name)
        print(y_test_name)
        print("train:test  -  {}:{}".format(len(y_train), len(y_test)))
        return images_train, y_train, images_test, y_test, x_max, x_min, y_train_name, y_test_name


if __name__ == '__main__':
    data_loader = load_data()

    # data_loader.visualization()
    images_train, y_train, images_test, y_test, x_max, x_min, y_train_name, y_test_name \
        = data_loader.load_dataset()
    #
    print(images_train.shape)
    print(y_train.shape)

    print("-------------")
    # print(y_train_name)
    # print(y_test_name)
