import cv2
from pylab import *
from scipy.ndimage import filters
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, alpha, sigma, random_state=None):

    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    print('image : ', id(image))
    if random_state is None:
        random_state = np.random.RandomState(None)

        # print(random_state)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='nearest')  # wrap,reflect, nearest
    return distored_image.reshape(image.shape)

if __name__ == "__main__":

    img = cv2.imread('example.JPEG')
    augmented_img = elastic_transform(img, 1000, 8, random_state=None)
    print(augmented_img.shape, augmented_img.dtype)

    augmented_img = np.concatenate([img, augmented_img], axis = 1)
    cv2.imshow('elastic', augmented_img)
    cv2.waitKey(0)
