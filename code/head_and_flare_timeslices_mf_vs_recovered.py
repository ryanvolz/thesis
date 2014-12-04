import numpy as np
import scipy as sp
import scipy.constants
from matplotlib import pyplot as plt
from matplotlib import animation
import cPickle
import copy
import os

import echolect as el
from echolect.tools.time import datetime_to_float

params = {#'figure.subplot.left': 0.01,
          #'figure.subplot.bottom': 0.01,
          #'figure.subplot.right': .99,
          #'figure.subplot.top': .99,
          #'figure.subplot.wspace': .025,
          #'figure.subplot.hspace': .025,
          'font.size': 10,
          'font.family': 'sans-serif',
          'font.sans-serif': ['TeX Gyre Heros', 'sans-serif'],
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          #'ps.usedistiller': 'pdftk',
          'axes.titlesize': 10,
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'lines.markersize': 1,
          'lines.linewidth': 0.45,
          'axes.linewidth': 0.45,
          'xtick.major.size': 2,
          'xtick.major.pad': 2,
          'ytick.major.size': 2,
          'ytick.major.pad': 3,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

basefilename = 'head_and_flare'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)

with open(basefilename + '_recovered.pkl', 'rb') as f:
    cs = cPickle.load(f)

n = 128
m = data.vlt.shape[-1]
freqs = np.fft.fftfreq(int(n), data.ts/np.timedelta64(1, 's'))
v = -freqs/data.f0*sp.constants.c/2

filts = []
for code in data.codes:
    s = (code/np.linalg.norm(code)).astype(data.vlt.dtype)
    filt = el.filtering.MatchedDoppler(s, n, m, xdtype=data.vlt.dtype)
    filts.append(filt)

# dimensions of EACH delay-frequency image
xinches = 1.875
yinches = 1.5

basedir = os.path.join('figures', basefilename, 'time_slices')
if not os.path.exists(basedir):
    os.makedirs(basedir)

cmap = copy.copy(plt.cm.coolwarm)
cmap.set_bad(cmap(0))

vidx = np.abs(v) <= np.max(np.abs(cs.v))

rslc = el.slice_by_value(data.r, 86000, 97000)
csrslc = el.slice_by_value(cs.r, data.r[rslc][0], data.r[rslc][-1])
# want an integer ratio for pixel aspect between cs data and non-cs data
cspixelaspect = int(round(len(cs.r[csrslc])/float(len(data.r[rslc]))))
# modify csrslc to make integer ratio exact
newstop = csrslc.stop + cspixelaspect*len(data.r[rslc]) - len(cs.r[csrslc])
csrslc = slice(csrslc.start, newstop, 1)
for kp in [4]:#xrange(5):
    pslc = slice(kp, None, 5)
    vlt = data.vlt[pslc]
    cs_sig = cs.vlt_sig[pslc]
    cs_noise = cs.vlt_noise[pslc]
    t = data.t[pslc]
    t0 = t[12]

    filt = filts[kp]

    numsteps = 4

    zses = []
    for k in xrange(numsteps):
        zs = np.zeros((len(v[vidx]), len(data.r[rslc])), np.float_)
        zses.append(zs)
    for k in xrange(numsteps):
        zscs = np.zeros((len(cs.v), len(cs.r[csrslc])), np.float_)
        zses.append(zscs)

    fig, axes = plt.subplots(numsteps, 2, sharey=True,
                             figsize=(2*xinches + 1, numsteps*yinches + 0.55))
    for ax in axes.ravel():
        el.make_axes_fixed(ax, xinches, yinches)

    imgs = []
    ttxts = []
    # matched filter plots
    for kax, ax in enumerate(axes[:, 0]):
        if kax == 0:
            title = 'Matched Filter'
        else:
            title = None
        if kax == numsteps - 1:
            xlabel = None#'Doppler range rate (km/s)'
        else:
            xlabel = None
            plt.setp(ax.get_xticklabels(), visible=False)
        mfimg = el.implot(
            zses[kax],
            np.fft.ifftshift(v[vidx][::-1])/1e3, data.r[rslc]/1e3,
            xlabel=xlabel, ylabel='Range (km)', title=title,
            cbar=False,
            exact_ticks=False, xbins=5,
            vmin=0, vmax=39,
            cmap=cmap,
            ax=ax,
        )
        imgs.append(mfimg)

        ttxt = ax.text(-0.4, 0.5,
                       't = {0:.0f} ms'.format(1e3*datetime_to_float(t0, t0)),
                       transform=ax.transAxes, ha='right', va='center')
        ttxts.append(ttxt)

    # waveform inversion plots
    for kax, ax in enumerate(axes[:, 1]):
        if kax == 0:
            title = 'Waveform Inversion'
        else:
            title = None
        if kax == numsteps - 1:
            xlabel = None#'Doppler range rate (km/s)'
        else:
            xlabel = None
            plt.setp(ax.get_xticklabels(), visible=False)
        csimg = el.implot(
            zses[numsteps + kax],
            np.fft.ifftshift(cs.v[::-1])/1e3, cs.r[csrslc]/1e3,
            xlabel=xlabel, clabel='SNR (dB)', title=title,
            exact_ticks=False, xbins=5,
            vmin=0, vmax=39,
            cmap=cmap, csize=0.0625, cpad=0.05,
            ax=ax,
        )
        imgs.append(csimg)

    ax = axes[-1, -1]
    ax.text(-0.05, -0.1, 'Doppler range rate (km/s)',
            transform=ax.transAxes, ha='center', va='top')

    plt.tight_layout(0.1)

    def init_frame():
        for img, zs in zip(imgs, zses):
            img.set_data(zs.T)
        return imgs, ttxts

    def animate(kfs):
        for ki, kf in enumerate(kfs):
            vlt_mf = filt(vlt[kf])[:, filt.validsame]
            mfimg = imgs[ki]
            mfimg.set_data(20*np.log10(np.abs(np.fft.ifftshift(vlt_mf[vidx, rslc][::-1,:],
                                                               axes=0))/data.noise_sigma).T)
            ttxt = ttxts[ki]
            ttxt.set_text('t = {0:.0f} ms'.format(1e3*datetime_to_float(t[kf], t0)))
        for ki, kf in enumerate(kfs):
            csimg = imgs[numsteps + ki]
            csimg.set_data(20*np.log10(np.abs(np.fft.ifftshift((cs_sig[kf, :, csrslc] +
                                                                cs_noise[kf, :, csrslc])[::-1,:],
                                                               axes=0))/cs.noise_sigma).T)
        return imgs, ttxts

    #for k in xrange(0, vlt.shape[0], numsteps):
        #animate([k, k+1, k+2, k+3])
        #path = os.path.join(basedir, basefilename +
                            #'_mf_vs_recovered_p{0}_t{1}.pdf').format(kp, k)
        #fig.savefig(path, bbox_inches='tight', pad_inches=0.01, transparent=True)

    for ks in [[12, 20, 32, 33], [50, 52, 136, 188]]:
        animate(ks)
        path = os.path.join(basedir, basefilename +
                            '_mf_vs_recovered_p{0}_ts{1}.pdf').format(kp, ks[1])
        fig.savefig(path, bbox_inches='tight', pad_inches=0.01, transparent=True)

#plt.show()
plt.close('all')
