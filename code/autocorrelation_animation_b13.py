import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
import os

params = {
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['TeX Gyre Heros', 'sans-serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.markersize': 2,
    'lines.linewidth': 0.45,
}
plt.rcParams.update(params)

savedpi = 72.27*4

b13 = np.asarray([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]).astype(np.float_)

s = b13
L = len(s)
ts = np.arange(L)

acorr = np.correlate(s, s, 'full')
M = len(acorr)
ta = np.fft.fftshift(np.fft.fftfreq(M)*M)

tpad = np.arange(-(L-1), 2*L - 1)
spad = np.zeros(2*(L-1) + L, s.dtype)
spad[L-1:2*L-1] = s

ks = range(-4, M + 7)

fig = plt.figure(figsize=(5, 1.5))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 3],
                       left=0.21, bottom=0, right=1, top=1)
axes = []
ax3 = fig.add_subplot(gs[3])
ax2 = fig.add_subplot(gs[2])
ax1 = fig.add_subplot(gs[1], sharex=ax2)
ax0 = fig.add_subplot(gs[0], sharex=ax2)
axes = [ax0, ax1, ax2, ax3]
axes[0].axes.get_xaxis().set_visible(False)
axes[0].axes.get_yaxis().set_visible(False)
axes[0].set_frame_on(False)
axes[0].tick_params(bottom=False, top=False, left=False, right=False)
axes[1].axes.get_xaxis().set_visible(False)
axes[1].axes.get_yaxis().set_visible(False)
axes[1].set_frame_on(False)
axes[1].tick_params(bottom=False, top=False, left=False, right=False)
axes[2].axes.get_xaxis().set_visible(False)
axes[2].axes.get_yaxis().set_visible(False)
axes[2].set_frame_on(False)
axes[2].tick_params(bottom=False, top=False, left=False, right=False)
axes[3].axes.get_xaxis().set_visible(False)
axes[3].axes.get_yaxis().set_visible(False)
axes[3].set_frame_on(False)
axes[3].tick_params(bottom=False, top=False)

axes[0].plot((ts[0] - 0.5, ts[-1] + 1.5), (0, 0), color=(0.5, 0.5, 0.5), zorder=-10)
axes[0].bar(ts, s, width=1, color='b', linewidth=0.45)
axes[0].text(ts[0] - 1, 0, r'$s[l]$', va='center', ha='right')
ax1line, = axes[1].plot((ts[0] - L + 0.5, ts[-1] - L + M + 0.5), (0, 0), color=(0.5, 0.5, 0.5), zorder=-10)
sliderects = axes[1].bar(ts - L + 1, s, width=1, color='b', linewidth=0.45)
rectxt = axes[1].text(ta[0] - 1, 0, r'$s^*[l-p+(L-1)]$', va='center', ha='right', zorder=-10)
ax2line, = axes[2].plot((ts[0] - 0.5, ts[-1] + 1.5), (0, 0), color=(0.5, 0.5, 0.5), zorder=-10)
plusrects = axes[2].bar(ts, np.ones_like(s), width=1, color=(0.75, 0.75, 1), linewidth=0.45)
minusrects = axes[2].bar(ts, -np.ones_like(s), width=1, color=(0.25, 0.25, 0.5), linewidth=0.45)
sumtxt = axes[2].text(ts[0] - 1, -1, r'$\sum$', va='bottom', ha='right')
sumtxt.set_visible(False)
restxt = axes[2].text(ts[-1] + 2, -1, '', va='bottom', ha='left', color='r')
restxt.set_visible(False)
axes[2].set_xlim(-L + 0.5, 2*L - 1 + 0.5)

axes[3].plot(ta, acorr, color=(0.5, 0.5, 0.5))
acorrline, = axes[3].plot([], [], 'r-')
acorrpts, = axes[3].plot([], [], marker='o', mfc='r', mec='r')
mftxt = axes[3].text(ta[0] - 1, 0, r'$\chi_s[p]$', va='bottom', ha='right')
axes[3].set_xlim(-L + 0.5 - L/2.0, L - 0.5 + L/2.0)
#axes[3].set_yticks([0, L])
axes[3].set_ylim(-0.5, 1.1*L)
acorrtxt = axes[3].text(0, 0, '', va='bottom', ha='center', color='r')
acorrtxt.set_visible(False)

#plt.tight_layout(0.1)

plt.draw() # need draw to update axes position
# need the resolution to be multiples of 2 for libx264
savesize = np.floor(savedpi*fig.get_size_inches())
if np.any(np.mod(savesize, 2)):
    newsize = np.mod(savesize, 2)/savedpi + fig.get_size_inches()
    fig.set_size_inches(newsize, forward=True)
    savesize = np.floor(savedpi*newsize)

def init_frame():
    ax1line.set_visible(False)
    for rect in sliderects:
        rect.set_visible(False)
    ax2line.set_visible(False)
    for rect in plusrects:
        rect.set_visible(False)
    for rect in minusrects:
        rect.set_visible(False)
    acorrline.set_data([], [])
    acorrpts.set_data([], [])
    acorrtxt.set_visible(False)
    sumtxt.set_visible(False)
    restxt.set_visible(False)
    #rectxt.set_visible(False)

    return ax1line, sliderects, ax2line, plusrects, minusrects, acorrline, acorrpts, acorrtxt, sumtxt, restxt, rectxt, mftxt

def animate(k):
    init_frame()
    if k < 0:
        rcvtxt.set_visible(True)
    elif k < M:
        #if k >= L-1:
            #rectxt.set_visible(True)
        rectxt.set_position((ta[k] - 1, 0))
        rectxt.set_text(r'$s^*[l-{0}+(L-1)]$'.format(k))
        ax1line.set_visible(True)
        ax1line.set_xdata((ts[0] - L + 0.5 + k, ts[0] + 1.5 + k))
        for kr, rect in enumerate(sliderects):
            rect.set_x(ts[kr] - L + 1 + k)
            rect.set_visible(True)
        ax2line.set_visible(True)
        tslc = tpad[k:(k+L)]
        spadslc = spad[k:(k+L)]
        mlt = s*spadslc
        for m, t in zip(mlt, tslc):
            if (t >= 0) and (t < L):
                if m > 0:
                    plusrects[t].set_visible(True)
                else:
                    minusrects[t].set_visible(True)
        acorrtxt.set_position((ta[k], acorr[k] + 1))
        acorrtxt.set_text(r'${0}$'.format(int(acorr[k])))
        acorrtxt.set_visible(True)
        sumtxt.set_visible(True)
        restxt.set_text(r'$={0}$'.format(int(acorr[k])))
        restxt.set_visible(True)
        mftxt.set_text(r'$\chi_s[{0}]$'.format(k))
        acorrline.set_data(ta[:(k + 1)], acorr[:(k + 1)])
        acorrpts.set_data(ta[k], acorr[k])
    else:
        acorrline.set_data(ta, acorr)
        acorrpts.set_data(ta, acorr)

    return ax1line, sliderects, ax2line, plusrects, minusrects, acorrline, acorrpts, acorrtxt, sumtxt, restxt, rectxt, mftxt

basedir = os.path.join('figures', 'autocorrelation_b13')
if not os.path.exists(basedir):
    os.makedirs(basedir)

for k in xrange(2*L-1):
    animate(k)
    fig.savefig(os.path.join(basedir, 'autocorrelation_b13_{0}.pdf'.format(k)),
                transparent=True)

#plt.show()
plt.close('all')
