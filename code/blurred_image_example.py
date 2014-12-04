import numpy as np
import scipy as sp
import scipy.ndimage
from skimage import data, img_as_float
import matplotlib.pyplot as plt

import echolect as el

camimg = img_as_float(data.camera())

b13 = np.asarray([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]).astype(np.float_)
b13amb = np.abs(el.autocorr(b13, 32))

K = np.fft.fftshift(b13amb, axes=0).T/np.max(b13amb)
def kernel(x):
    return np.sum(x*K)

blurredimg = sp.ndimage.convolve(camimg, K, mode='constant', cval=0.0)

plt.imsave('figures/blurred_image_example_orig.png', camimg,
           cmap=plt.cm.gray, dpi=camimg.shape[1]/1.3)

plt.imsave('figures/blurred_image_example_kernel.png', K,
           cmap=plt.cm.gray, dpi=K.shape[1]/1.3)

plt.imsave('figures/blurred_image_example_blurred.png', blurredimg,
           cmap=plt.cm.gray, dpi=blurredimg.shape[1]/1.3)
