#! /usr/bin/env python
"""
Plots consequences of transitions
=================================

"""

# Created: Sat Dec 15, 2018  03:41pm
# Last modified: Thu Feb 21, 2019  08:00pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

import toymodels as tm

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]

if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    # fig = pl.figure(figsize=[7.480315, 5.314961])     # 190 mm wide, 135 mm tall 
    # fig = pl.figure(figsize=[7.480315, 4.724409])     # 190 mm wide, 120 mm tall 
    fig = pl.figure(figsize=[7.480315, 3.937008])     # 190 mm wide, 100 mm tall 
    lm, bm, wd, ht = 0.10, 0.750, 0.80, 0.225
    ax1 = fig.add_axes([lm, bm, wd, ht])
    kh = 2.70
    ax2 = fig.add_axes([lm - 0.05, bm - kh * ht - 0.10, wd / 2, kh * ht],
                       projection="3d")
    ax3 = fig.add_axes([lm + wd / 2 + 0.05, bm - kh * ht - 0.10, wd / 2, kh * ht],
                       projection="3d")
    axlabfs, tiklabfs = 12, 11
    splabs = {
                ax1: "a",
                ax2: "b",
            }
    legbba = [[0.38, 0.825], [0.10, 0.825]]
    main_clr, err_clr = "MediumTurquoise", "GoldenRod"
    clr3 = "IndianRed"


    # time vector
    T = np.linspace(0., 600., 600001)
    n = len(T)
    t1 = T[:n / 2]
    t2 = T[n / 2:]
    neq = 50000

    # spiral-type chaos
    # -----------------
    print("spiral-type chaos ...")
    params = (0.32, 0.3, 4.5)
    x0 = (0.0, -1., 1.)
    pos = tm.roessler(x0, t1, params)
    x1, y1, z1 = pos[neq:, 0], pos[neq:, 1], pos[neq:, 2]

    # screw-type chaos
    # ----------------
    print("screw-type chaos ...")
    params = (0.38, 0.4, 4.820)
    x0 = (x1[-1], y1[-1], z1[-1])
    pos = tm.roessler(x0, t2, params)
    x2, y2, z2 = pos[neq:, 0], pos[neq:, 1], pos[neq:, 2]

    # get full time series and the mean and variance filter
    print("mean and variance ...")
    X = np.r_[x1, x2]
    T = T[:-2 * neq]
    n = len(T)
    k = 25000
    Xav = np.array([X[i-k/2:i+k/2].mean() for i in range(k / 2, n - k / 2)])
    # Xav = np.r_[X[:k/2], Xav, X[n - k/2:]]
    Xsd = np.array([X[i-k/2:i+k/2].var() for i in range(k / 2, n - k / 2)])
    T_ = T[(k / 2) : (n - k / 2)]
    # Xsd = np.r_[X[:k/2], Xsd, X[n - k/2:]]
    # Xsd = np.array([X[i-k/2:i+k/2].std() for i in range(k / 2, n - k / 2)])

    # plot
    print("plot ...")
    ax1.plot(T, X, "-", c=main_clr, label="Signal")
    ax1.plot(T_, Xav, "-", c=clr3, label="Moving average", alpha=0.5)
    ax1.axvline(T[n / 2], linestyle="--", color="0.5")
    ax1_ = ax1.twinx()
    ax1_.plot(T_, Xsd, "-", c=err_clr, label="Moving variance", alpha=0.5)
    ax2.plot(x1, y1, z1, "-", c=main_clr, lw=1.0, alpha=0.85)
    ax3.plot(x2, y2, z2, "-", c=main_clr, lw=1.0, alpha=0.85)


    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs)
    ax1.set_ylim(-15., 25.)
    ax1.set_yticks(np.arange(-10., 10.1, 10))
    leg = ax1.legend(loc="upper left", ncol=2,
                     bbox_to_anchor=[0.01, 1.05])
    ax1.set_xlabel("Time", fontsize=axlabfs, labelpad=5.)
    ax1.set_ylabel(r"$x$", fontsize=axlabfs, labelpad=5.)
    ax1_.set_ylabel("Variance", fontsize=axlabfs, labelpad=1.)
    leg_ = ax1_.legend(loc="upper right",
                      bbox_to_anchor=[0.99, 1.05])
    for txt in leg.get_texts():
        txt.set_size(tiklabfs)
    for txt in leg_.get_texts():
        txt.set_size(tiklabfs)
    ax1.set_xlim(T[0], T[-1])

    for ax in [ax2, ax3]:
        ax.set_xlim(-15., 15.)
        ax.set_ylim(-15., 15.)
        ax.set_zlim(0., 50.)
        ax.view_init(elev=20., azim=245.)
        # ax.view_init(elev=20., azim=60.)
        ax.xaxis.set_pane_color((1., 1. ,1., 1.))
        ax.zaxis.gridlines.set_alpha(0.5)
        ax.yaxis.set_pane_color((1., 1. ,1., 1.))
        ax.zaxis.gridlines.set_alpha(0.5)
        ax.zaxis.set_pane_color((1., 1. ,1., 1.))
        ax.zaxis.gridlines.set_alpha(0.5)
        ax.set_xlabel(r"$x$", fontsize=axlabfs, labelpad=5.)
        ax.set_ylabel(r"$y$", fontsize=axlabfs, labelpad=5.)
        ax.set_zlabel(r"$z$", fontsize=axlabfs, labelpad=5.)
        ax.set_xticks(np.arange(-10., 10.1, 10.))
        ax.set_yticks(np.arange(-10., 10.1, 10.))
    ax1_.set_yticks(np.arange(0., 50.1, 10.))
    ax1_.set_ylim(0., 50.)

    # subplot labels
    ax1.text(-0.10, 1.0,
             "A",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax1.transAxes
             )
    fig.text(0.05, 0.55,
             "B",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             )
    fig.text(0.55, 0.55,
             "C",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             )

    # save figure
    FN = "../figs/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

