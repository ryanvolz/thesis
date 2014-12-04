import numpy as np
import scipy as sp
import scipy.stats
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

phi = sp.stats.norm.pdf
Phi = sp.stats.norm.cdf

tau = np.concatenate((np.linspace(0, 1, 100)[:-1], np.logspace(0, 1, 100)))
delta = 2*phi(tau)/(tau + 2*(phi(tau) - tau*Phi(-tau)))
rho = 1 - tau*Phi(-tau)/phi(tau)

fig = plt.figure(figsize=(3, 2))
ax = fig.add_subplot(111)
ax.plot(delta, rho, 'k')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$\delta = M/N$')
ax.set_ylabel(r'$\rho = S/M$')
plt.fill_between(delta, rho, 0, color='g', alpha=0.1)
plt.fill_between(delta, rho, 1, color='r', alpha=0.1)
ax.text(delta[75], rho[75], r'$\rho_\mathrm{MSE}(\delta)$', ha='right', va='bottom')
plt.tight_layout(0.1)

fig.savefig('figures/l1_phase_transition.pdf', bbox_inches='tight',
            pad_inches=0.01, transparent=True)

plt.show()
plt.close('all')
