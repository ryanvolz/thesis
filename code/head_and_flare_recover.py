import numpy as np
import scipy as sp
import scipy.constants
import cPickle
from bunch import Bunch

import echolect as el
import radarmodel
import prx

basefilename = 'head_and_flare'
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
r = data.r
if R > 1:
    diffr = np.diff(r)
    diffr = np.concatenate((diffr, [diffr[-1]]))
    rup = el.filtering.upsample(r, R)
    for k in xrange(1, R):
        rup[k::R] = r + diffr*k/R
    r = rup

lmbda = data.noise_sigma/np.sqrt(n)*2

As = []
Astars = []
nodelays = []
for k, code in enumerate(data.codes):
    s = (code/np.linalg.norm(code)).astype(data.vlt.dtype)
    A = radarmodel.point.fastest_forward(s, n, m, R)
    Astar = radarmodel.point.fastest_adjoint(s, n, m, R)

    As.append(A)
    Astars.append(Astar)

    try:
        code_delay = data.code_delays[k]
    except:
        code_delay = 0

    delay = Astar.delays + code_delay
    nodelays.append(slice(np.searchsorted(delay, 0), np.searchsorted(delay, R*m)))

vlt_sig = np.zeros((data.vlt.shape[0], n, R*m), data.vlt.dtype)
vlt_noise = np.zeros_like(vlt_sig)
h = np.zeros((data.vlt.shape[0], R*m), data.vlt.real.dtype)
h_sig = np.zeros_like(h)
h_noise = np.zeros((data.vlt.shape[0], R*m), data.vlt.dtype)
x0s = [np.zeros(A.inshape, A.indtype) for A in As]
for p in xrange(data.vlt.shape[0]):
    y = data.vlt[p]
    A = As[p % len(As)]
    Astar = Astars[p % len(Astars)]
    x = prx.l1rls(A, Astar, y, lmbda=lmbda, x0=x0s[p % len(As)], printrate=100)

    nz = Astar(y - A(x))

    # matched filter result with sidelobes removed is vlt_sig + vlt_noise
    nodelayslc = nodelays[p % len(nodelays)]
    vlt_sig[p] = x[:, nodelayslc]/np.sqrt(n)
    vlt_noise[p] = nz[:, nodelayslc]*np.sqrt(n)

    h_sig[p] = np.sqrt(np.sum(vlt_sig[p].real**2 + vlt_sig[p].imag**2, axis=0))
    # use zero Doppler noise since by definition noise is wideband with no Doppler shift
    h_noise[p] = np.abs(vlt_noise[p, 0])
    # sqrt(n) factor included in noise term by summing n terms of nz[0]
    h[p] = np.sqrt(np.sum(np.abs(vlt_sig[p] + nz[0, nodelayslc])**2, axis=0))

    x0s[p % len(As)] = x

fidx = np.abs(freqs) < 75000
rslc = el.slice_by_value(r, 80000, 110000)
recovered = Bunch(vlt_sig=vlt_sig[:, fidx, rslc], vlt_noise=vlt_noise[:, fidx, rslc],
                  t=data.t, f=freqs[fidx], v=v[fidx], r=r[rslc], n=n,
                  ts=data.ts, ipp=data.ipp, f0=data.f0,
                  noise_sigma=data.noise_sigma)
with open(basefilename + '_recovered.pkl', 'wb') as f:
    cPickle.dump(recovered, f, protocol=-1)

rec_range = Bunch(h=h, h_sig=h_sig, h_noise=h_noise, t=data.t, r=r, n=n,
                  ts=data.ts, ipp=data.ipp, f0=data.f0,
                  noise_sigma=data.noise_sigma)
with open(basefilename + '_recovered_range.pkl', 'wb') as f:
    cPickle.dump(rec_range, f, protocol=-1)
