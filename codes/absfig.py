#! /usr/bin/env python
"""
This script generates the recurrence plot figure for the extended abstract
==========================================================================

"""

# Created: Fri Aug 31, 2018  04:22pm
# Last modified: Fri Aug 31, 2018  05:17pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl
import datetime as dt
import recurrence as rc


if __name__ == "__main__":
    # generate white noise data and get RP
    Xnoise = np.random.randn(500)
    Tnoise = np.arange(1, 501)
    Rnoise = rc.rp(Xnoise, m=1, tau=1, e=0.1,
                   metric="euclidean", threshold_by="frr")

    # load the Nino 3.4 data, estimate embedding parameters and get RP
    # load Nino 3.4 index data
    D = np.loadtxt("../data/enso/nino.txt", delimiter=",", skiprows=5)
    Y, M = D[:, 0], D[:, 1]
    Xnino = D[:, -1]
    # convert time info to datetime array
    Tnino = []
    for y, m in zip(Y, M):
        Tnino.append(dt.datetime(int(y), int(m), 15))
    Tnino = np.array(Tnino)
    mi, lags = rc.mi(Xnino, maxlag=100)
    i = rc.first_minimum(mi)
    tau = lags[i]
    fnn, dims = rc.fnn(Xnino, tau, maxdim=20, r=0.01)
    i = rc.first_zero(fnn)
    m = dims[i]
    Rnino = rc.rp(Xnino, m, tau, e=0.1,
                  metric="euclidean", threshold_by="frr")

    # load the FordA data, estimate embedding parameters and get RP
    D = np.loadtxt("../data/fordA/FordA_TEST.txt", delimiter=",")
    k = 1325
    Xford = D[k, 1:]
    Tford = np.arange(Xford.shape[0])
    mi, lags = rc.mi(Xford, maxlag=100)
    i = rc.first_minimum(mi)
    tau = lags[i]
    fnn, dims = rc.fnn(Xford, tau, maxdim=20, r=0.01)
    i = rc.first_zero(fnn)
    m = dims[i]
    Rford = rc.rp(Xford, m, tau, e=0.1,
                  metric="euclidean", threshold_by="frr")

    # set up figure
    fig = pl.figure(figsize=[7.68, 3.93], dpi=320)
    b1, h1 = 0.75, 0.15
    l_ = [0.08, 0.41, 0.72]
    b2, h2= 0.12, 0.480
    w = 0.260
    ax_top = []
    ax_bot = []
    for l in l_:
        ax_top.append(fig.add_axes([l, b1, w, h1]))
        ax_bot.append(fig.add_axes([l, b2, w, h2]))
    splabfs, axlabfs, tiklabfs = 12, 10, 7
    col_hdr = ["Random Process", r"Ni$\mathsf{\~n}$o 3.4", "Engine Noise"]
    signal = [Xnoise, Xnino, Xford]
    time = [Tnoise, Tnino, Tford]
    RPs = [Rnoise, Rnino, Rford]
    splab_top = ["a", "b", "c"]
    splab_bot = ["d", "e", "f"]

    # plot
    for ax, x, t, hdr, lab in zip(ax_top, signal, time, col_hdr, splab_top):
        ax.plot(t, x, "-", c="SteelBlue", lw=0.75)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Time", fontsize=axlabfs)
        if ax == ax_top[0]:
            ax.set_ylabel("Signal", fontsize=axlabfs)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(hdr, fontsize=splabfs)
        ax.text(-0.15, 1.10,
                lab,
                fontsize=splabfs, fontweight="bold",
                ha="left", va="center", rotation=0,
                transform=ax.transAxes)

    for ax, R, t, lab in zip(ax_bot, RPs, time, splab_bot):
        t = t[:R.shape[0]]
        tx, ty = np.meshgrid(t, t)
        ax.pcolormesh(tx, ty, R,
                      cmap=pl.cm.gray_r, rasterized=True
                      )
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(t[0], t[-1])
        if ax == ax_bot[-1]:
            ax.set_xticks(np.arange(0., 501., 100.))
            ax.set_yticks(np.arange(0., 501., 100.))
        ax.set_xlabel("Time", fontsize=axlabfs)
        if ax == ax_bot[0]:
            ax.set_ylabel("Time", fontsize=axlabfs)
        ax.text(-0.15, 1.10,
                lab,
                fontsize=splabfs, fontweight="bold",
                ha="left", va="center", rotation=0,
                transform=ax.transAxes)

    # prettify
    for ax in fig.axes:
        ax.tick_params(labelsize=tiklabfs)

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True)
    print("figure saved to: %s" % FN)

