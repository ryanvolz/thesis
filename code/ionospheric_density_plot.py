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
          'mathtext.fontset': 'custom',
          'mathtext.rm': ['TeX Gyre Heros', 'sans-serif'],
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

ne_night = np.genfromtxt('ionospheric_density_night.lst')
ne_day = np.genfromtxt('ionospheric_density_day.lst')

fig = plt.figure(figsize=(2.5, 3))
plt.loglog(ne_day[:, 1]/1e6, ne_day[:, 0], 'g-')
plt.loglog(ne_night[:, 1]/1e6, ne_night[:, 0], 'b-.')
plt.xlabel(r'Plasma density (cm$^{-\mathrm{3}}$)')
plt.ylabel('Altitude (km)')
plt.legend(('Day', 'Night'), bbox_to_anchor=(0, 0.8), loc='upper left',
           handlelength=3, frameon=False)
plt.xlim(50, 2e6)
plt.ylim(60, 2000)
ax = plt.gca()
ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=[0.1, 0.2, 0.4, 0.6, 1.0]))
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.text(0.2, 0.02, 'D Region', ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.39, 0.14, 'E Region', ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.425, 0.4, 'F Region', ha='left', va='bottom', transform=ax.transAxes)

plt.tight_layout(0.1)

fig.savefig('figures/ionospheric_density.pdf', bbox_inches='tight',
            pad_inches=0.01, transparent=True)
#plt.show()
plt.close(fig)
