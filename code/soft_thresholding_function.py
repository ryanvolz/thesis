import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

x = np.arange(0, 6 + 1) - 3
y = np.zeros_like(x)
y[x < -1] = x[x < -1] + 1
y[x > 1] = x[x > 1] - 1

fig = plt.figure(figsize=(3.5, 2.25))
ax = fig.add_subplot(111, aspect='equal')
ax.hlines(0, -3, 3)
ax.vlines(0, -2, 2)
ax.plot(x, y, linewidth=1)
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(1, 0))
ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter((r'$-3\tau$', r'$-2\tau$', r'$-\tau$', r'$0$', r'$\tau$', r'$2\tau$', r'$3\tau$')))
ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(1, 0))
ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter((r'$-2\tau$', r'$-\tau$', r'$0$', r'$\tau$', r'$2\tau$')))
ax.text(-1, 0.1, r'$-\tau$', verticalalignment='bottom', horizontalalignment='center')
ax.text(1, 0.1, r'$\tau$', verticalalignment='bottom', horizontalalignment='center')
ax.set_xlabel(r'$v$')
ax.set_ylabel(r'$\mathbf{soft}_\tau(v)$')
ax.yaxis.label.set_rotation('horizontal')
ax.yaxis.label.set_horizontalalignment('right')
ax.yaxis.label.set_verticalalignment('center')
plt.tight_layout(0.1)

fig.savefig('figures/soft_thresholding_function.pdf', bbox_inches='tight',
            pad_inches=0.01, transparent=True)

plt.show()
plt.close('all')
