import cv2
import numpy as np
import torch

def shiftImage(image):
    left_image = image[:, :,0:-1]
    right_image = image[:, :,1:]
    new_image = torch.abs(left_image - right_image).sum()
    # model suma de new image
    new_image = np.concatenate((new_image, image[:,-1].reshape(-1, 1)),axis=1)
    new_image = np.concatenate((image[:,0].reshape(-1, 1), new_image),axis=1)

    return new_image

def epe(output, target):
    return np.linalg.norm(output - target) / (output.shape[0] * output.shape[1])

output = np.random.rand(32, 384, 768)
target = np.random.rand(32, 384, 768)

print(epe(output, target))
print(shiftImage(output).shape)


