import numpy as np
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

N = 16
f = np.fft.fftshift(np.fft.fftfreq(N*100))
h = np.sinc(N*f)/np.sinc(f)

fig = plt.figure(figsize=(4,1.75))
plt.plot(f, h)#, color=(0, 0.5, 0))
plt.xlim(-0.5, 0.5)
plt.ylim(-0.25, 1)
plt.xlabel(r'Normalized Frequency ($1/\tau_b$)')
plt.ylabel(r'$\mathrm{psinc}_{16}(2\pi\tau_b f)$')
plt.tight_layout(0.1)

fig.savefig('figures/periodic_sinc.pdf', bbox_inches='tight',
            pad_inches=0.01, transparent=True)

plt.show()
plt.close('all')
