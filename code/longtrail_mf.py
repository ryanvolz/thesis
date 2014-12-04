import numpy as np
import scipy as sp
import scipy.constants
import cPickle
from bunch import Bunch

import echolect as el
import radarmodel

mfblksize = 5
mfvoters = [1, 2, 4]

basefilename = 'longtrail'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)

n = 128
m = data.vlt.shape[-1]
freqs = np.fft.fftfreq(int(n), data.ts/np.timedelta64(1, 's'))
v = freqs/data.f0*sp.constants.c/2
try:
    R = data.code_undersampling
except:
    R = 1

filts = []
for code, delay in zip(data.codes, data.code_delays):
    s = (code/np.linalg.norm(code)).astype(data.vlt.dtype)
    if R > 1:
        s = el.filtering.downsample(s, R)
        delay = delay/R
    filt = el.filtering.MatchedDoppler(s, n, m, xdtype=data.vlt.dtype)
    filt.nodelay = slice(filt.L - 1 - delay, filt.L - 1 - delay + filt.M)
    filts.append(filt)

vlt_mf = np.zeros_like(data.vlt)
freq = np.zeros(data.vlt.shape[0])
vlt_mf_all = np.zeros((mfblksize, n, m), data.vlt.dtype)
for kp in xrange(data.vlt.shape[0]):
    y = data.vlt[kp]
    filt = filts[kp % len(filts)]
    x = filt(y)
    vlt_mf_all[kp % mfblksize] = x[:, filt.nodelay]

    if ((kp + 1) % mfblksize) == 0:
        # get the frequency shift that gives max SNR for each pulse in data block
        shifts = np.zeros(len(mfvoters), 'int8')
        for ks, kmf in enumerate(mfvoters):
            vlt = vlt_mf_all[kmf]
            shifts[ks] = np.unravel_index(np.argmax(vlt.real**2 + vlt.imag**2), vlt.shape)[0]

        # wrap high positive shifts to negative, so median works near 0
        shifts = (shifts + n/2) % n - n/2
        shift = np.median(shifts)

        # store matched filter data for selected shift
        for ks in xrange(mfblksize):
            k = kp + 1 - mfblksize + ks
            vlt_mf[k] = vlt_mf_all[ks, shift]
            freq[k] = float(shift)/n*(np.timedelta64(1, 's')/data.ts)

mf = Bunch(vlt=vlt_mf, t=data.t, r=data.r, freq=freq, n=n, ts=data.ts,
           ipp=data.ipp, f0=data.f0, noise_sigma=data.noise_sigma)
with open(basefilename + '_mf.pkl', 'wb') as f:
    cPickle.dump(mf, f, protocol=-1)
