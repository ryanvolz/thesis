import numpy as np
import matplotlib.pyplot as plt

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

def make_lfm(n, ts, bandwidth_hz):
    t = np.arange(n)*ts
    slope = bandwidth_hz/(n*ts)
    return np.exp(2*np.pi*1j*(0.5*slope*t - bandwidth_hz/2.)*t)

barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]).astype(np.float_)
uncoded = np.ones(13)
msl = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1]).astype(np.float_)
lfm = make_lfm(51, 1e-6, 1e6)

d13 = np.fft.fftshift(np.fft.fftfreq(2*13 - 1)*(2*13 - 1))
d51 = np.fft.fftshift(np.fft.fftfreq(2*51 - 1)*(2*51 - 1))
f = np.fft.fftshift(np.fft.fftfreq(512, 1))

b13amb = np.abs(el.autocorr(barker13, 512))
uncamb = np.abs(el.autocorr(uncoded, 512))
mslamb = np.abs(el.autocorr(msl, 512))
lfmamb = np.abs(el.autocorr(lfm, 512))

dpi = 255
savedpi = dpi*1 # should be a multiple of dpi

def plotter(z, x, y, pixelaspect, **kwargs):
    xinches = len(x)/float(dpi)
    yinches = len(y)/float(dpi)*pixelaspect
    figsize = (xinches + 1.5, yinches + 1)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    el.make_axes_fixed(ax, xinches, yinches)
    img = el.implot(z, x, y, ax=ax, csize=0.0625, cpad=0.05, **kwargs)
    plt.tight_layout(0.01)
    return fig

plotter(np.fft.fftshift(uncamb, axes=0), f, d13, pixelaspect=16,
        xlabel='Normalized Frequency', ylabel='Delay (samples)',
        clabel='Relative Magnitude', exact_ticks=False, xbins=6, ybins=6, cbins=5)
plt.savefig('figures/ambiguity_uncoded.pdf', dpi=savedpi, bbox_inches='tight',
            pad_inches=0.05, transparent=True)

plotter(np.fft.fftshift(b13amb, axes=0), f, d13, pixelaspect=16,
        xlabel='Normalized Frequency', ylabel='Delay (samples)',
        clabel='Relative Magnitude', exact_ticks=False, xbins=6, ybins=6, cbins=5)
plt.savefig('figures/ambiguity_barker13.pdf', dpi=savedpi, bbox_inches='tight',
            pad_inches=0.05, transparent=True)

plotter(np.fft.fftshift(mslamb, axes=0), f, d51, pixelaspect=10,
        xlabel='Normalized Frequency', ylabel='Delay (samples)',
        clabel='Relative Magnitude', exact_ticks=False, xbins=6, ybins=6)
plt.savefig('figures/ambiguity_msl.pdf', dpi=savedpi, bbox_inches='tight',
            pad_inches=0.05, transparent=True)

plotter(np.fft.fftshift(lfmamb, axes=0), f, d51, pixelaspect=10,
        xlabel='Normalized Frequency', ylabel='Delay (samples)',
        clabel='Relative Magnitude', exact_ticks=False, xbins=6, ybins=6)
plt.savefig('figures/ambiguity_lfm.pdf', dpi=savedpi, bbox_inches='tight',
            pad_inches=0.05, transparent=True)

#plt.show()
plt.close('all')
