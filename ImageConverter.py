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
        """
        Read and convert png to np.array.

        :filename: string, file name
        :return: 3D np.array ( W * H * 3 Channels )
        """

        if filename[-4:-1] == ('.pfm') or filename[-4:-1] == ('.PFM'):
            data = self.readPFM()[0]
            if len(data.shape) == 3:
                return data[:, :, 0:3]
            else:
                return data

        image = cv2.imread(filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (768, 384))

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

    def previewPhoto(self, data):
        plt.imshow(data, cmap='hot')
        plt.show()


