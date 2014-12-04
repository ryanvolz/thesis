import numpy as np
import scipy as sp
import scipy.constants
from matplotlib import pyplot as plt
from matplotlib import animation
import cPickle
import copy
import os

import echolect as el

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

pixelaspect=1
basefilename = 'head_and_flare'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)
    r = data.r
    del data
with open(basefilename + '_recovered.pkl', 'rb') as f:
    cs = cPickle.load(f)

# maximum width and height in inches of figure (not counting savefig padding)
# one dimension will be trimmed to achieve desired pixel aspect ratio if specified
# (only width works well since plt.tight_layout() isn't actually tight vertically)
maxfigsize = (5.7, 2.3)
def plotter(z, t, r, **kwargs):
    fig = plt.figure(figsize=maxfigsize)
    ax = plt.subplot(111)
    img = el.rtiplot(z, t, r, ax=ax, csize=0.0625, cpad=0.05, **kwargs)
    plt.tight_layout(0.1)
    return fig

basedir = os.path.join('figures', basefilename, 'doppler_slices')
if not os.path.exists(basedir):
    os.makedirs(basedir)

cmap = copy.copy(plt.cm.coolwarm)
cmap.set_bad(cmap(0))

rslc = el.slice_by_value(r, 86000, 97000)
csrslc = el.slice_by_value(cs.r, r[rslc][0], r[rslc][-1])
# want an integer ratio for pixel aspect between cs data and non-cs data
cspixelaspect = int(round(len(cs.r[csrslc])/float(len(r[rslc]))))
# modify csrslc to make integer ratio exact
diff = cspixelaspect*len(r[rslc]) - len(cs.r[csrslc])
if diff != 0:
    newstop = csrslc.indices(len(cs.r))[1] + diff
    csrslc = slice(csrslc.start, newstop, 1)

for kp in xrange(5):
    pslc = slice(kp, None, 5)

    for kf in xrange(5):#cs.vlt_sig.shape[1]):
        fig0 = plotter(20*np.log10(np.abs((cs.vlt_sig[pslc, kf, csrslc]
                                           + cs.vlt_noise[pslc, kf, csrslc])/cs.noise_sigma)),
                       cs.t[pslc], cs.r[csrslc]/1e3, pixelaspect=cspixelaspect*pixelaspect,
                       exact_ticks=False, vmin=0, vmax=40, cmap=cmap,
                       ylabel='Range (km)', clabel='SNR (dB)')
        path = os.path.join(basedir, basefilename + '_recovered_doppler_rti_f{0}_p{1}.pdf')
        path = path.format(kf, pslc.start)

        fig0.savefig(path, bbox_inches='tight', pad_inches=0.05, transparent=True)

        plt.close(fig0)

        fig1 = plotter(20*np.log10(np.abs(cs.vlt_sig[pslc, kf, csrslc]/cs.noise_sigma)),
                       cs.t[pslc], cs.r[csrslc]/1e3, pixelaspect=cspixelaspect*pixelaspect,
                       exact_ticks=False, vmin=0, vmax=40, cmap=cmap,
                       ylabel='Range (km)', clabel='SNR (dB)')
        path = os.path.join(basedir, basefilename + '_recovered_sparse_doppler_rti_f{0}_p{1}.pdf')
        path = path.format(kf, pslc.start)

        fig1.savefig(path, bbox_inches='tight', pad_inches=0.05, transparent=True)

        plt.close(fig1)

        fig2 = plotter(20*np.log10(np.abs(cs.vlt_noise[pslc, kf, csrslc]/cs.noise_sigma)),
                       cs.t[pslc], cs.r[csrslc]/1e3, pixelaspect=cspixelaspect*pixelaspect,
                       exact_ticks=False, vmin=0, vmax=40, cmap=cmap,
                       ylabel='Range (km)', clabel='SNR (dB)')
        path = os.path.join(basedir, basefilename + '_recovered_noise_doppler_rti_f{0}_p{1}.pdf')
        path = path.format(kf, pslc.start)

        fig2.savefig(path, bbox_inches='tight', pad_inches=0.05, transparent=True)

        plt.close(fig2)
