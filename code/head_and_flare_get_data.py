import numpy as np
import cPickle
from bunch import Bunch

import echolect as el
import echolect.jicamarca

datadir = '/storage/data/jicamarca/2011_meteor/data/d2011060_00'

def make_lfm(n, ts, bandwidth_hz):
    t = np.arange(n)*ts
    slope = bandwidth_hz/(n*ts)
    return np.exp(2*np.pi*1j*(0.5*slope*t - bandwidth_hz/2.)*t)

barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]).astype(np.float_)
msl = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1]).astype(np.float_)
uncoded = np.ones(40)
lfm = make_lfm(51, 1e-6, 1e6)
psrnd = np.array([-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1]).astype(np.float_)
codes = [barker13, msl, uncoded, lfm, psrnd]
code_delays = [0]*len(codes)

files = el.jicamarca.find_files(datadir)

vlt_o = el.jicamarca.voltage_reader(files[556])

rall = vlt_o.r
rslc = el.slice_by_value(rall, 80000, 110000)
r = rall[rslc]
vlt1 = vlt_o.read_from_block(13, 700, 0, sample_idx=rslc, chan_idx=0)
vlt2 = vlt_o.read_from_block(14, 0, 700, sample_idx=rslc, chan_idx=0)
vlt = np.concatenate((vlt1, vlt2), axis=0)
t = vlt_o.block_times[13] + (700 + np.arange(vlt.shape[0]))*vlt_o.ipp
ts = vlt_o.ts
ipp = vlt_o.ipp
f0 = 49920000.0

noise_vlt = vlt_o.read_from_block(13, 0, 0, sample_idx=el.slice_by_value(rall, None, 80000),
                                  chan_idx=0)
noise_sigma = np.sqrt(np.mean(noise_vlt.real**2 + noise_vlt.imag**2))

store = Bunch(vlt=vlt, r=r, t=t, codes=codes, code_delays=code_delays,
              ts=ts, ipp=ipp, f0=f0, noise_sigma=noise_sigma)
with open('head_and_flare.pkl', 'wb') as f:
    cPickle.dump(store, f, protocol=-1)
