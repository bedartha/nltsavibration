#! /usr/bin/env python3
"""
Plots consequences of predictability
====================================

"""

# Created: Sat Dec 15, 2018  01:08pm
# Last modified: Mon Sep 02, 2019  02:50pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl

import toymodels as tm

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]

if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    fig = pl.figure(figsize=[7.480315, 1.771654])     # 190 mm wide, 45 mm tall 
    lm, bm, wd, ht = 0.10, 0.350, 0.40, 0.625
    ax1 = fig.add_axes([lm, bm, wd, ht])
    ax2 = fig.add_axes([lm + wd, bm, wd, ht])
    ax2.set_facecolor("none")
    axlabfs, tiklabfs = 12, 11
    splabs = {
                ax1: "A",
                ax2: "B",
            }
    legbba = [[0.38, 0.825], [0.10, 0.825]]
    main_clr, err_clr = "MediumTurquoise", "GoldenRod"


    # time vector
    t = np.linspace(0., 500., 501)
    n = len(t)
    m = 100             # ensemble size for future prediction
    E = 0.000001        # noise level (as a multiplier of STD)

    # linear example
    # --------------
    print("linear example ...")
    # 1) get data
    pi = np.pi
    T1, T2, T3 = 11., 87, 210.
    w1, w2, w3 = 1. / T1, 1. / T2, 1. / T3
    A1, A2, A3 = 1.5, 4.5, 7.5
    x1 = A1 * np.sin(2. * pi * w1 * t)
    x2 = A2 * np.sin(2. * pi * w2 * t)
    x3 = A3 * np.sin(3. * pi * w3 * t)
    x = x1 + x2 + x3 + np.random.randn(n) * (A1 + A2 + A3) * 0.25
    # 2) get noise limit based on variance of main trajectory
    std = np.std(x)
    # 3) get upper and lower bounds on trajectory using erroneous IC
    ne = int(n / 2)
    te = t[ne:]
    xe = np.zeros((m, len(te)))
    re = np.random.randn(m)
    for k in range(m):
        xe[k] = x[ne:] + E * std * re[k]
    xhi = np.percentile(xe, 97.5, axis=0)
    xlo = np.percentile(xe, 2.5, axis=0)
    # 2) plot
    ax1.plot(t, x, "-", c=main_clr, zorder=10, alpha=0.75)
    ax1.fill_between(te, xhi, xlo,
                     color=err_clr)
    ax1.axvline(te[0], color="IndianRed", linestyle="--")

    # nonlinear example
    # -----------------
    print("nonlinear example ...")
    # 1) get primary time series
    params = (0.3, 0.4, 7.5)
    x0 = (0.0, -1., 1.)
    pos = tm.roessler(x0, t, params)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # 2) get noise limit based on variance of main trajectory
    std = np.std(x)
    # 3) get upper and lower bounds on trajectory using erroneous IC
    ne = int(n / 2)
    te = t[ne:]
    xe = np.zeros((m, len(te)))
    re = np.random.randn(m)
    for k in range(m):
        xe0 = (x[ne] + E * std * re[k], y[ne], z[ne])
        pose = tm.roessler(xe0, t[ne:], params)
        xe[k] = pose[:, 0]
    xhi = np.percentile(xe, 97.5, axis=0)
    xlo = np.percentile(xe, 2.5, axis=0)
    # 2) plot
    ax2.plot(t, x, "-", c=main_clr, zorder=10, alpha=0.75)
    ax2.fill_between(te, xhi, xlo,
                     color=err_clr)
    ax2.axvline(te[0], color="IndianRed", linestyle="--")

    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs,
                       left="on", right="on")
        ax.set_ylabel("Signal (au)", fontsize=axlabfs, labelpad=10)
        ax.set_ylim(-25., 25.)
        ax.set_xlim(0., t[-1])
        ax.set_yticks(np.arange(-20., 20.01, 10.))
        ax.text(0.08, 0.90,
                splabs[ax],
                ha="right", va="center",
                fontsize=axlabfs, fontweight="heavy", family="sans-serif",
                usetex=False,
                transform=ax.transAxes
                )
    ax2.yaxis.set_label_position("right")
    ax1.set_xlim(ax1.get_xlim()[::-1])
    ax2.tick_params(right="on",
                    labelleft="off", labelright="on")
    fig.text(0.5, 0.05,
             "Time (au)",
             ha="center", va="center",
             fontsize=axlabfs,
             )

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

