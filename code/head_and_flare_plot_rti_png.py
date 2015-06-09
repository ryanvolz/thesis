import numpy as np
import matplotlib.pyplot as plt
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

with open(basefilename + '_recovered_range.pkl', 'rb') as f:
    cs = cPickle.load(f)

with open(basefilename + '_mf.pkl', 'rb') as f:
    mf = cPickle.load(f)

maxfigsize = (4.3, 1.8)
def plotter(z, t, r, **kwargs):
    nx = len(t)
    ny = len(r)
    width, height, dpi = el.size_dpi_nointerp(nx, ny, maxfigsize[0], maxfigsize[1])
    fig = plt.figure(figsize=(width + 1.5, height + 1))
    ax = plt.subplot(111)
    el.make_axes_fixed(ax, width, height)
    img = el.rtiplot(z, t, r, ax=ax, csize=0.0625, cpad=0.05, **kwargs)
    plt.tight_layout(0.1)
    return fig, dpi

basedir = os.path.join('figures', basefilename)
if not os.path.exists(basedir):
    os.makedirs(basedir)
cmap = copy.copy(plt.cm.coolwarm)
cmap.set_bad(cmap(0))

rslc = el.slice_by_value(data.r, 86000, 97000)
csrslc = el.slice_by_value(cs.r, data.r[rslc][0], data.r[rslc][-1])
# want an integer ratio for pixel aspect between cs data and non-cs data
cspixelaspect = int(round(len(cs.r[csrslc])/float(len(data.r[rslc]))))
# modify csrslc to make integer ratio exact
newstop = csrslc.stop + cspixelaspect*len(data.r[rslc]) - len(cs.r[csrslc])
csrslc = slice(csrslc.start, newstop, 1)
for pslc in [slice(0, None, 5), slice(1, None, 5), slice(2, None, 5),
             slice(3, None, 5), slice(4, None, 5), slice(None)]:
    fig1, dpi = plotter(
        20*np.log10(cs.h_sig[pslc, csrslc]/cs.noise_sigma),
        cs.t[pslc], cs.r[csrslc]/1e3, pixelaspect=cspixelaspect*pixelaspect,
        exact_ticks=False, vmin=0, vmax=40, cmap=cmap,
        ylabel='Range (km)', clabel='SNR (dB)',
    )
    savedpi = int(round(300/float(dpi)))*dpi;
    path = os.path.join(basedir, basefilename + '_recovered_rti_{0}.png')
    if pslc.start is not None:
        path = path.format(pslc.start)
    else:
        path = path.format('all')
    fig1.savefig(path, dpi=savedpi, bbox_inches='tight', pad_inches=0.05, transparent=True)

    fig2, dpi = plotter(
        20*np.log10(np.abs(cs.h_noise[pslc, csrslc])/cs.noise_sigma),
        cs.t[pslc], cs.r[csrslc]/1e3, pixelaspect=cspixelaspect*pixelaspect,
        exact_ticks=False, vmin=0, vmax=40,
        ylabel='Range (km)', clabel='SNR (dB)',
    )
    savedpi = int(round(300/float(dpi)))*dpi;
    path = os.path.join(basedir, basefilename + '_recovered_noise_{0}.png')
    if pslc.start is not None:
        path = path.format(pslc.start)
    else:
        path = path.format('all')
    fig2.savefig(path, dpi=savedpi, bbox_inches='tight', pad_inches=0.05, transparent=True)

    fig3, dpi = plotter(
        20*np.log10(cs.h[pslc, csrslc]/cs.noise_sigma),
        cs.t[pslc], cs.r[csrslc]/1e3, pixelaspect=cspixelaspect*pixelaspect,
        exact_ticks=False, vmin=0, vmax=40,
        ylabel='Range (km)', clabel='SNR (dB)',
    )
    savedpi = int(round(300/float(dpi)))*dpi;
    path = os.path.join(basedir, basefilename + '_recovered_rti_noise_{0}.png')
    if pslc.start is not None:
        path = path.format(pslc.start)
    else:
        path = path.format('all')
    fig3.savefig(path, dpi=savedpi, bbox_inches='tight', pad_inches=0.05, transparent=True)

    fig4, dpi = plotter(
        20*np.log10(np.abs(mf.vlt[pslc, rslc])/mf.noise_sigma),
        mf.t[pslc], mf.r[rslc]/1e3, pixelaspect=pixelaspect,
        exact_ticks=False, vmin=0, vmax=40,
        ylabel='Range (km)', clabel='SNR (dB)',
    )
    savedpi = int(round(300/float(dpi)))*dpi;
    path = os.path.join(basedir, basefilename + '_mf_rti_{0}.png')
    if pslc.start is not None:
        path = path.format(pslc.start)
    else:
        path = path.format('all')
    fig4.savefig(path, dpi=savedpi, bbox_inches='tight', pad_inches=0.05, transparent=True)

    fig5, dpi = plotter(
        20*np.log10(np.abs(data.vlt[pslc, rslc])/data.noise_sigma),
        data.t[pslc], data.r[rslc]/1e3, pixelaspect=pixelaspect,
        exact_ticks=False, vmin=0, vmax=40,
        ylabel='Range (km)', clabel='SNR (dB)',
    )
    savedpi = int(round(300/float(dpi)))*dpi;
    path = os.path.join(basedir, basefilename + '_rti_{0}.png')
    if pslc.start is not None:
        path = path.format(pslc.start)
    else:
        path = path.format('all')
    fig5.savefig(path, dpi=savedpi, bbox_inches='tight', pad_inches=0.05, transparent=True)

    plt.close('all')

plt.show()
