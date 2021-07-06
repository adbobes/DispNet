import os
import re
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2


class ImageConverter:
    def read(self, filename):
        """
        Determines what type of file is and convert it to np.array's data.

        :filename: string, file name
        :return:
        """
        try:
            if (filename[-4:] == ".png"):
                return self.readPNG(filename)
            elif (filename[-4:] == ".pfm"):
                return self.readPFM(filename)[0]
            else:
                raise Exception("Type of the file is unknown, please use only '.png' and '.pmg' files.")
        except:
            print("ERROR: Exception to converting function.")

    def readPNG(self, filename):

        image = cv2.imread(filename)

        image = cv2.resize(image, (768, 384))
        return image

    def readPFM(self, filename):
        """
        Read and convert PFM files to np.array.

        :filename: string, file name
        :return: tuple, (data, scale) - target and scale
        """

        # open the file
        file = open(filename, 'rb')

        # parameters for reading pmf file
        color = None  # color of pmf, True/False
        width = None  # width of pmf map
        height = None  # height of pmf map
        scale = None  #
        endian = None  # little/big - endian

        # color
        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception("Not a '.pfm' file")

        # map dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        # scale
        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # create np array and reshape it
        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # flip each column in up-down direction

        data = cv2.resize(data, (384, 192), interpolation = cv2.INTER_CUBIC)
        return data, scale


    def convert(self, samples, left_root, right_root, disparity_root_left, disparity_root_right):

        self.data = []
        self.target = []
        self.length = samples

        list_files = os.listdir(left_root)  # dir is your directory path
        number_files_left = len(list_files)

        list_files = os.listdir(right_root)  # dir is your directory path
        number_files_right = len(list_files)

        list_files = os.listdir(disparity_root_left)  # dir is your directory path
        number_files_gt = len(list_files)

        list_files = os.listdir(disparity_root_right)  # dir is your directory path
        number_files_gt = len(list_files)

        if number_files_left != number_files_right:
            raise Exception("Number of samples from left/right folders are not equal.")

        if number_files_left != number_files_gt:
            raise Exception("Number of samples from ground truth folder is not equal with left/right samples number.")

        for idx in range(samples):

            if idx < 10:
                file_format = "000000"
            elif 9 < idx < 100:
                file_format = "00000"
            elif 99 < idx < 1000:
                file_format = "0000"
            else:
                file_format = "000"

            current_left_path = left_root + file_format + str(idx) + ".png"
            current_right_path = right_root + file_format + str(idx) + ".png"
            current_disparity_path_left = disparity_root_left + file_format + str(idx) + ".pfm"
            current_disparity_path_right = disparity_root_right + file_format + str(idx) + ".pfm"

            left_image = self.read(current_left_path)
            right_image = self.read(current_right_path)
            disparity_image_left = self.read(current_disparity_path_left)
            disparity_image_right = self.read(current_disparity_path_right)

            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f'photo/new photo/left/' + file_format + str(idx) + ".png", cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'photo/new photo/right/' + file_format + str(idx) + ".png", cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
            np.save('photo/new photo/disparity left/' + file_format + str(idx), disparity_image_left)
            np.save('photo/new photo/disparity right/' + file_format + str(idx), disparity_image_right)

    def previewPhoto(self, data):
        plt.imshow(data, cmap='hot')
        plt.show()

LEFT_ROOT = "photo/left/"             # LEFT PNG'S FOLDER ROOT
RIGHT_ROOT = "photo/right/"           # RIGHT PNG'S FOLDER ROOT
DISPARITY_ROOT_LEFT = "photo/disparity/"   # DISPARITY PFM'S FOLDER ROOT
DISPARITY_ROOT_RIGHT = "photo/disparity right/"   # DISPARITY PFM'S FOLDER ROOT

dataset_train = ImageConverter()
#dataset_train.convert(samples = 1150, left_root = LEFT_ROOT, right_root = RIGHT_ROOT, disparity_root_left = DISPARITY_ROOT_LEFT,
                      #disparity_root_right=DISPARITY_ROOT_RIGHT)

left = np.load("photo/new photo/disparity left/0000000.npy")
right = np.load("photo/new photo/disparity right/0000000.npy")

leftt = cv2.imread("photo/new photo/left/0000000.png")
rightt = cv2.imread("photo/new photo/right/0000000.png")

leftt = cv2.cvtColor(leftt, cv2.COLOR_BGR2RGB)
rightt = cv2.cvtColor(rightt, cv2.COLOR_BGR2RGB)

print(left.shape)

dataset_train.previewPhoto(leftt)
dataset_train.previewPhoto(rightt)

dataset_train.previewPhoto(left)
dataset_train.previewPhoto(right)