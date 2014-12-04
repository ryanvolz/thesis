import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
from mpl_toolkits.axes_grid1 import Divider, axes_size, make_axes_locatable

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
          'lines.markersize': 1.5,
          'lines.linewidth': 0.45,
          'axes.linewidth': 0.45,
          'xtick.major.size': 2,
          'xtick.major.pad': 2,
          'ytick.major.size': 2,
          'ytick.major.pad': 3,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

def plot(t, tx, h, rx):
    tstep = t[1] - t[0]
    M = int(t[-1] + tstep)
    samp = np.searchsorted(t, np.arange(M + 1), 'right') - 1

    fig = plt.figure(figsize=(3, 2.5))
    horiz = [axes_size.Scaled(10)]
    vert = [axes_size.Scaled(1), axes_size.Fixed(0.05),
            axes_size.Scaled(2), axes_size.Fixed(0.01),
            axes_size.Scaled(2), axes_size.Fixed(0.15),
            axes_size.Fixed(0.1), axes_size.Fixed(0.15),
            axes_size.Scaled(2), axes_size.Fixed(0.01),
            axes_size.Scaled(2), axes_size.Fixed(0.05),
            axes_size.Scaled(1), axes_size.Fixed(0.1),
            axes_size.Fixed(0.1)]
    div = Divider(fig, (0.1, 0, 0.85, 0.875), horiz, vert)
    ax0 = SubplotZero(fig, 111)
    fig.add_subplot(ax0)

    ax0.set_axes_locator(div.new_locator(nx=0, ny=len(vert)-1))
    axs = [ax0]
    for k in range(2, len(vert), 2):
        ax = fig.add_axes((0, 0, 1, 1), label="%d"%k, sharex=ax0)
        ax.set_axes_locator(div.new_locator(nx=0, ny=len(vert)-1-k))
        axs.append(ax)
    ax0.set_xlim(-0.5, t[-1] + 0.5)
    for ax in axs:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_ticks([])
        ax.set_frame_on(False)
        ax.set_ylim(-1.1, 1.1)

    axs[0].axis['left','bottom','top','right'].set_visible(False)
    axs[0].set_xticks(range(0,M), minor=True)
    xzero = axs[0].axis['xzero']
    xzero.set_axisline_style('->')
    xzero.set_visible(True)
    xzero.set_axis_direction('top')
    xzero.set_label('Delay (samples)')

    axs[1].bar(t[samp], np.abs(tx[samp]), width=0.8,
               linewidth=0.45, color='k', edgecolor='k')
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].text(-0.5, 0.5, r'$s[l]$', va='center', ha='right', size=8)

    txon = np.abs(tx) > 0
    axs[2].bar(t[txon], 2*np.abs(tx[txon]), width=tstep, bottom=-np.abs(tx[txon]),
               linewidth=0, color=(0.9, 0.9, 0.9))
    axs[2].plot(t, tx.real, 'b')
    axs[2].plot(t[samp], tx[samp].real, 'ko')
    axs[3].bar(t[txon], 2*np.abs(tx[txon]), width=tstep, bottom=-np.abs(tx[txon]),
               linewidth=0, color=(0.9, 0.9, 0.9))
    axs[3].plot(t, tx.imag, 'g')
    axs[3].plot(t[samp], tx[samp].imag, 'ko')
    axs[3].text(-0.5, 1.05, 'TX', va='center', ha='right', size=8)

    t_tgts = t[np.abs(h) > 0]
    h_tgts = h[np.abs(h) > 0]
    axs[4].scatter(t_tgts, np.zeros(len(t_tgts)), s=25*np.abs(h_tgts),
                   c='r', marker='x', linewidths=1)
    axs[4].text(-2.5, 0, 'Targets', va='center', ha='left', size=8)

    rxon = np.abs(rx) > 0
    axs[5].bar(t[rxon], 2*np.abs(rx[rxon]), width=tstep, bottom=-np.abs(rx[rxon]),
               linewidth=0, color=(0.9, 0.9, 0.9))
    axs[5].plot(t, rx.real, 'b')
    axs[5].plot(t[samp], rx[samp].real, 'ko')
    axs[6].bar(t[rxon], 2*np.abs(rx[rxon]), width=tstep, bottom=-np.abs(rx[rxon]),
               linewidth=0, color=(0.9, 0.9, 0.9))
    axs[6].plot(t, rx.imag, 'g')
    axs[6].plot(t[samp], rx[samp].imag, 'ko')
    axs[6].text(-0.5, 1.05, 'RX', va='center', ha='right', size=8)

    axs[7].bar(t[samp], np.abs(rx[samp]), width=0.8,
               linewidth=0.45, color='k', edgecolor='k')
    axs[7].set_ylim(-0.05, 1.05)
    axs[7].text(-0.5, 0.5, r'$|y[m]|$', va='center', ha='right', size=8)

    return fig

f0 = 2.3
points_per_sample = np.round(50.*f0)
tstep = 1/points_per_sample

M = 20
t = np.arange(0, M + tstep, tstep)
carrier = np.exp(2*np.pi*1j*f0*t)

h = np.zeros_like(t)
h[np.searchsorted(t, 5.2)] = 1
h[np.searchsorted(t, 12.73)] = 0.5
h[np.searchsorted(t, 15.32)] = 0.5

ss = np.zeros_like(t)
ss[t < 1] = 1
txs = ss*carrier
rxs = np.convolve(txs, h, 'full')[:len(h)]

sl = np.zeros_like(t)
sl[t < 4] = 1
txl = sl*carrier
rxl = np.convolve(txl, h, 'full')[:len(h)]

figs = plot(t, txs, h, rxs)
figl = plot(t, txl, h, rxl)

figs.savefig('figures/shortpulse_return.pdf', transparent=True)
figl.savefig('figures/longpulse_return.pdf', transparent=True)

#plt.show()
plt.close(figs)
plt.close(figl)
