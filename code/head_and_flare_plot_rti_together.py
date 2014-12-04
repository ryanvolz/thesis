import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
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

# size of block of images only, not counting axes and labels, etc.
blocksize = (5, 1.5)
def plot_block(z, z_unc, t, r, pixelaspect=None, **kwargs):
    # eat pixelaspect argument since it messes things up with fixed size axes

    # approximate size for figure
    # (doesn't matter if saving with tight bbox)
    fig = plt.figure(figsize=(blocksize[0] + 1, blocksize[1] + 0.75))

    # size for between upper and lower plots
    # we will add sizes for labels and titles later
    h = [axes_grid1.Size.Fixed(blocksize[0]/5.0)]*5
    v = [axes_grid1.Size.Fixed(blocksize[1])]
    gs = matplotlib.gridspec.GridSpec(1, 1, left=0.0875, bottom=0.25, right=1, top=1)
    div = axes_grid1.SubplotDivider(fig, gs[0],
                                    horizontal=h, vertical=v)
    loc0 = div.new_locator(nx=0, ny=0)
    loc1 = div.new_locator(nx=1, ny=0)
    loc2 = div.new_locator(nx=2, ny=0)
    loc3 = div.new_locator(nx=3, ny=0)
    loc4 = div.new_locator(nx=4, ny=0)

    ax0 = fig.add_axes(loc0(None, None), label='ax0')
    ax1 = fig.add_axes(loc1(None, None), label='ax1', sharey=ax0)
    ax2 = fig.add_axes(loc2(None, None), label='ax2', sharey=ax0)
    ax3 = fig.add_axes(loc3(None, None), label='ax3', sharey=ax0)
    ax4 = fig.add_axes(loc4(None, None), label='ax4', sharey=ax0)

    # turn off unwanted (duplicate) tick labels
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    # locate the axes in the divider
    ax0.set_axes_locator(loc0)
    ax1.set_axes_locator(loc1)
    ax2.set_axes_locator(loc2)
    ax3.set_axes_locator(loc3)
    ax4.set_axes_locator(loc4)

    # also have to override get_subplotspec after setting locator
    # so tight_layout works
    ax0.get_subplotspec = loc0.get_subplotspec
    ax1.get_subplotspec = loc1.get_subplotspec
    ax2.get_subplotspec = loc2.get_subplotspec
    ax3.get_subplotspec = loc3.get_subplotspec
    ax4.get_subplotspec = loc4.get_subplotspec

    # plot the frequency shift
    # plot the images
    img0 = el.rtiplot(z[0::5, :], t[0::5], r/1e3, title='Barker 13',
                      ylabel='Range (km)',
                      ax=ax0, cbar=False, xbins=6,
                      exact_ticks=False, interpolation='none', **kwargs)
    img1 = el.rtiplot(z[1::5, :], t[1::5], r/1e3, title='MSL',
                      ax=ax1, cbar=False, xbins=6,
                      exact_ticks=False, interpolation='none', **kwargs)
    img2 = el.rtiplot(z_unc, t[2::5], r/1e3, title='Uncoded',
                      ax=ax2, cbar=False, xbins=6,
                      exact_ticks=False, interpolation='none', **kwargs)
    img3 = el.rtiplot(z[3::5, :], t[3::5], r/1e3, title='LFM',
                      ax=ax3, cbar=False, xbins=6,
                      exact_ticks=False, interpolation='none', **kwargs)
    img4 = el.rtiplot(z[4::5, :], t[4::5], r/1e3, title='PSRND',
                      clabel='SNR (dB)',
                      ax=ax4, xbins=6,
                      exact_ticks=False, interpolation='none', **kwargs)

    # erase all but one xlabel on plots for separate codes so they don't overlap
    ax0.set_xlabel('')
    ax1.set_xlabel('')
    ax3.set_xlabel('')
    ax4.set_xlabel('')

    # tight layout
    gs.tight_layout(fig)
    plt.draw()

    return fig

basefilename = 'head_and_flare'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)

with open(basefilename + '_mf.pkl', 'rb') as f:
    mf = cPickle.load(f)

basedir = os.path.join('figures', basefilename)
if not os.path.exists(basedir):
    os.makedirs(basedir)
cmap = copy.copy(plt.cm.coolwarm)
cmap.set_bad(cmap(0))

pslc = slice(0, len(mf.t) - 375)
rslc = el.slice_by_value(mf.r, 80000, 108000)

fig = plot_block(20*np.log10(np.abs(mf.vlt[pslc, rslc])/mf.noise_sigma),
                 20*np.log10(np.abs(data.vlt[pslc, rslc][2::5, :])/data.noise_sigma),
                 mf.t[pslc], mf.r[rslc],
                 vmin=0, vmax=40,
                 csize=0.0625, cpad=0.05)
fpath = os.path.join(basedir, basefilename + '_mf_rti_block.pdf')
fig.savefig(fpath, bbox_inches='tight', pad_inches=0.01, transparent=True)
plt.close(fig)

plt.show()
