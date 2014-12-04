import numpy as np
import matplotlib as mpl
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

f = np.fft.fftshift(np.fft.fftfreq(32))
d = np.arange(25)
np.random.seed(6)
Z = np.random.exponential(scale=1.0,size=(len(f),len(d)))
Z[17,16] = 10

dpi = 15 # should be sized to match font size
savedpi = dpi*4 # should be a multiple of dpi
xinches = len(f)/float(dpi)
yinches = len(d)/float(dpi)
figsize = (xinches + 1.5, yinches + 1)
fig = plt.figure(figsize=figsize)
ax = plt.subplot(111)
el.make_axes_fixed(ax, xinches, yinches)
img = el.implot(Z, f, d,
                csize=0.0625, cpad=0.05,
                exact_ticks=False, xbins=6, ybins=6,
                vmin=0, vmax=10,
                interpolation='none', pixelaspect=1,
                xlabel='Normalized Frequency', ylabel='Delay (samples)',
                clabel='Reflectivity',
                ax=ax)
ax = img.axes
ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(1.0/len(f), 0))
ax.yaxis.set_minor_locator(mpl.ticker.IndexLocator(1.0, 0))
ax.xaxis.set_tick_params(which='minor', size=0)
ax.yaxis.set_tick_params(which='minor', size=0)
ax.grid(which='minor', color='m', linewidth=0.45, linestyle='-')
plt.tight_layout(0.1)

fig.savefig('figures/range_doppler_discretization.pdf', bbox_inches='tight',
            pad_inches=0.01, transparent=True)

plt.close(fig)
