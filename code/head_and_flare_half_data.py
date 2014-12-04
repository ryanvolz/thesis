import numpy as np
import scipy as sp
import scipy.signal
import cPickle
from bunch import Bunch

import echolect as el
import echolect.estimators.interpolators as interp

R = 2

with open('head_and_flare.pkl', 'rb') as f:
    data = cPickle.load(f)

data.vlt = data.vlt[:, ::R]
data.r = data.r[::R]
data.code_undersampling = R

with open('head_and_flare_half.pkl', 'wb') as f:
    cPickle.dump(data, f, protocol=-1)
