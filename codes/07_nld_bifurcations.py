#! /usr/bin/env python
"""
Plots bifurcation diagrams for Lorenz-63 and Henon
==================================================

"""

# Created: Sat Dec 15, 2018  03:41pm
# Last modified: Thu Feb 21, 2019  08:04pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl

from scipy.signal import find_peaks_cwt

import toymodels as tm
from utils import _progressbar_start, _progressbar_update, _progressbar_finish

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]

if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    # fig = pl.figure(figsize=[7.480315, 5.314961])     # 190 mm wide, 135 mm tall 
    # fig = pl.figure(figsize=[7.480315, 4.724409])     # 190 mm wide, 120 mm tall 
    fig = pl.figure(figsize=[7.480315, 3.937008])     # 190 mm wide, 100 mm tall 
    lm, bm, wd, ht = 0.10, 0.150, 0.360, 0.75
    kh = 1.40
    ax1 = fig.add_axes([lm, bm, wd, ht])
    ax2 = fig.add_axes([lm + kh * wd, bm, wd, ht])
    axlabfs, tiklabfs = 12, 11
    clrs = ["MediumTurquoise", "GoldenRod", "IndianRed"]


    # Roessler
    # ---------
    print("Roessler ...")
    TR = np.linspace(0., 10000., 100000.)
    aR = np.arange(0.001, 0.550001, 0.001)
    bR, cR = 2., 4.
    nR = len(aR)
    mR = 5000
    yR_eq = []
    kR = 5
    x0R = (-1. + 2. * np.random.rand(nR, kR),
           -1. + 2. * np.random.rand(nR, kR),
           1. * np.random.rand(nR, kR))
    count = 0
    TOL = 1E-3
    pbar_on = True
    pb = _progressbar_start(max_value=nR * kR, pbar_on=pbar_on)
    for i in range(nR):
        params = (aR[i], bR, cR)
        yR_eq_ = []
        for l in range(kR):
            x0R_ = (x0R[0][i, l], x0R[1][i, l], x0R[2][i, l])
            XR = tm.roessler(x0R_, TR, params)
            xR_ = XR[-mR:, 0]
            yR_ = XR[-mR:, 1]
            if np.all((yR_[1:] - yR_[:-1]) < TOL):
                yR_eq_.extend(yR_)
            else:
                d2y = np.diff(np.sign(np.diff(yR_)))
                iipos = np.where(d2y == -2.)[0] + 1
                if len(iipos) > 0:
                    yR_eq_.extend(yR_[iipos])
                iineg = np.where(d2y == 2.)[0] + 1
                if len(iineg) > 0:
                    yR_eq_.extend(yR_[iineg])
            _progressbar_update(pb, count)
            count += 1
        np.random.shuffle(yR_eq_)
        yR_eq.append(yR_eq_[-250:])
    yR_eq = np.array(yR_eq)
    _progressbar_finish(pb)

    # Henon
    # -----
    print("Henon ...")
    TH = np.arange(0, 5000, 1)
    aH = np.arange(0.0, 1.40001, 0.005)
    nH = len(aH)
    mH = 100
    xH_eq = []
    kH = 10
    pb = _progressbar_start(max_value=nH * kH, pbar_on=True)
    count = 0
    for i in range(nH):
        params = (aH[i], 0.30)
        xH_eq_ = []
        for l in range(kH):
            x0H = (-0.3 + 0.6 * np.random.rand(),
                   -0.3 + 0.6 * np.random.rand()
                   )
            XH = tm.henon(x0H, TH[-1], params)
            xH_eq_.extend(XH[-mH:, 0])
            _progressbar_update(pb, count)
            count += 1
        xH_eq.append(xH_eq_)
    _progressbar_finish(pb)
    xH_eq = np.array(xH_eq)

    # plot
    print("plot ...")
    print("\tRoessler ...")
    for i in range(nR):
        mR = len(yR_eq[i])
        ax1.plot(aR[i] * np.ones(mR), yR_eq[i], "o",
                 mec="none", mfc="k", ms=0.50,
                 alpha=1.0, rasterized=True,
                )
    print("\tHenon ...")
    for i in range(nH):
        mH = len(xH_eq[i])
        ax2.plot(aH[i] * np.ones(mH), xH_eq[i], "o",
                 mec="none", mfc="k", ms=0.50,
                 alpha=1.0, rasterized=True,
                )

    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs)
    # ## Roessler
    ax1.set_xlim(0., 0.55)
    ax1.set_ylim(-6.5, 2.5)
    ax1.set_xticks(np.arange(0., 0.55, 0.10))
    ax1.set_yticks(np.arange(-6., 2.01, 2.))
    ax1.set_xlabel(r"$a$", fontsize=axlabfs, labelpad=5.)
    ax1.set_ylabel(r"$y$", fontsize=axlabfs, labelpad=5.)
    ax1.axvline(0.432, ls="--", color="IndianRed", lw=1.00, zorder=20)
    ## Henon map
    ax2.set_xlim(0., 1.41)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xticks(np.arange(0., 1.4001, 0.35))
    ax2.set_yticks(np.arange(-1.5, 1.51, 0.5))
    ax2.set_xlabel(r"$a$", fontsize=axlabfs, labelpad=5.)
    ax2.set_ylabel(r"$x_t$", fontsize=axlabfs, labelpad=5.)
    ax2.axvline(1.4, ls="--", color="IndianRed", lw=1.00, zorder=20)


    # subplot labels
    ax1.text(-0.20, 0.97,
             "A",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax1.transAxes
             )
    ax2.text(-0.20, 0.97,
             "B",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax2.transAxes
             )

    # save figure
    FN = "../figs/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

