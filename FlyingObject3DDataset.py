import os
import cv2
import ImageConverter
import numpy as np
from torch.nn import Module
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


class FlyingObject3DDataset(Dataset):
    def __init__(self, train=True, samples=100, left_root="", right_root="", disparity_root=""):
        """
        Create inputs and outputs using files from FlyingThings3D dataset.

        CONSTRUCTOR PARAMTERS:

        :train: bool, indicates if the inputs or outputs are from training/testing
        :samples: int, number of samples
        :left_root: string, path of the folder where left photos are
        :right_root: string, path of the folder where right photos are
        :dispratiy_root: strin, path of the folder where disparity photos are

        INTERN PARAMETERS:

        :data: list, represents a list of inputs; the inputs are left and right png's concatenated
               by channel axes
        :target: list, list of targets heatmaps; converted pfm's files to float values
        :converter: ImageConvertor, used for converting png's and pfm's files to nparray's
        :lenght: int, number of samples

        """

        self.converter = ImageConverter.ImageConverter()
        self.data = []
        self.target = []
        self.length = samples

        list_files = os.listdir(left_root)  # dir is your directory path
        number_files_left = len(list_files)

        list_files = os.listdir(right_root)  # dir is your directory path
        number_files_right = len(list_files)

        list_files = os.listdir(disparity_root)  # dir is your directory path
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
            else:
                file_format = "0000"

            current_left_path = left_root + file_format + str(idx) + ".png"
            current_right_path = right_root + file_format + str(idx) + ".png"
            current_disparity_path = disparity_root + file_format + str(idx) + ".pfm"

            left_image = self.converter.read(current_left_path)
            right_image = self.converter.read(current_right_path)
            disparity_image = self.converter.read(current_disparity_path)

            left_image, right_image, disparity_image = self.distortion(left_image, right_image, disparity_image)


            self.data.append(np.dstack((left_image, right_image)))
            self.target.append(disparity_image)

    def __getitem__(self, idx):
        inputs = self.data[idx]
        inputs = inputs.transpose((2, 0, 1))

        outputs = self.target[idx].copy()
        outputs = outputs / 500



        return torch.from_numpy(inputs), torch.from_numpy(outputs)

    def __len__(self):
        return self.length

    def previewImage(self, image):
        """
        Preview image using heatmap's from MatPlotLib package.

        PARAMETERS:
        :image: np.array, the image to be plotted
        """
        self.converter.previewPhoto(image)

    def distortion(self, left_image, right_image, disparity_image):
            left = left_image
            right = right_image
            disparity = disparity_image

            if np.random.randint(4) == 1:
                left = cv2.flip(left_image, 1)
                right = cv2.flip(right_image, 1)
                disparity = cv2.flip(disparity_image, 1)

            if np.random.randint(4) == 0:
                height_random = np.random.randint(30,100)
                lenght_random = height_random * 2
                h_shape = left.shape[0]
                w_shape = left.shape[1]

                corner = np.random.randint(20,50)
                corner_h = corner * np.random.randint(2,3)
                corner_w = corner_h * 2

                left = left[corner:h_shape - height_random + corner_h, corner: w_shape - lenght_random + corner_w]
                right = right[corner:h_shape - height_random + corner_h, corner: w_shape - lenght_random + corner_w]

                disparity = disparity[corner:h_shape - height_random + corner_h,corner: w_shape - lenght_random + corner_w]

                left = cv2.resize(left, (768, 384))
                right = cv2.resize(right, (768, 384))
                disparity = cv2.resize(disparity, (384, 192), interpolation = cv2.INTER_CUBIC)

            if np.random.randint(1) == 0:

                rand = np.random.randint(5)
                #brightness
                if rand == 0:
                    brightness = np.random.randint(10, 80)
                    left = cv2.convertScaleAbs(left, beta=brightness)
                    right = cv2.convertScaleAbs(right, beta=brightness)

                #contrast
                if rand == 1:
                    contrast = round(np.random.uniform(1, 1.5), 2)
                    left = cv2.convertScaleAbs(left, alpha = contrast)
                    right = cv2.convertScaleAbs(right, alpha = contrast)

                #saturation
                if rand == 2:
                    left = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
                    rand = np.random.randint(10, 60)  # 0 60
                    h, s, v = cv2.split(left)
                    s = s + rand
                    left = cv2.merge([h, s, v])
                    left = cv2.cvtColor(left, cv2.COLOR_HSV2RGB)

                    right = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)  # 0 60
                    h, s, v = cv2.split(right)
                    s = s + rand
                    right = cv2.merge([h, s, v])
                    right = cv2.cvtColor(right, cv2.COLOR_HSV2BGR)

                #hue
                if rand == 3:
                    left = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
                    rand = np.random.randint(20, 240)
                    h, s, v = cv2.split(left)
                    h = h + rand
                    left = cv2.merge([h, s, v])
                    left = cv2.merge([h, s, v])
                    left = cv2.cvtColor(left, cv2.COLOR_HSV2BGR)

                    right = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(right)
                    h = h + rand
                    right = cv2.merge([h, s, v])
                    right = cv2.cvtColor(right, cv2.COLOR_HSV2BGR)

            return left, right, disparity
