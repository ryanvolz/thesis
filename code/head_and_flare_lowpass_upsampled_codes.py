import numpy as np
import scipy as sp
import scipy.signal
import cPickle
from bunch import Bunch

import echolect as el
import echolect.estimators.interpolators as interp

with open('head_and_flare.pkl', 'rb') as f:
    data = cPickle.load(f)

n = 10
up = 2
n2 = n/up
n_lp = 50*n + 1
h_interp = interp.cubic_filter(n)
h_lowpass = sp.signal.remez(n_lp, bands=[0, 0.925/n, 1.075/n, 1],
                            desired=[1, 0], weight=[1, 23], Hz=2)
delays = []
for k, code in enumerate(data.codes):
    x_up = el.filtering.upsample(code, n)
    x_interp = el.filtering.filter(h_interp, x_up)
    x_lp = el.filtering.filter(h_lowpass, x_interp)
    filt_delay = (n_lp - 1)//2 - 1 # lowpass delay and part of interpolation delay
    code_lp = el.filtering.downsample(x_lp[filt_delay:(filt_delay + len(x_interp))], n2)
    data.codes[k] = code_lp
    delays.append(2*up) # rest of interpolation delay, //n2
data.code_delays = delays
data.code_undersampling = up

with open('head_and_flare_lowpass_up.pkl', 'wb') as f:
    cPickle.dump(data, f, protocol=-1)
