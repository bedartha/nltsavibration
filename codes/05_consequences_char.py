#! /usr/bin/env python3
"""
Plots consequences on characterization
======================================

"""

# Created: Sat Dec 15, 2018  05:13pm
# Last modified: Mon Sep 02, 2019  03:45pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

import toymodels as tm
import recurrence as rc

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]

if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    # fig = pl.figure(figsize=[7.480315, 5.314961])     # 190 mm wide, 135 mm tall 
    fig = pl.figure(figsize=[7.480315, 4.724409])     # 190 mm wide, 120 mm tall 
    # fig = pl.figure(figsize=[7.480315, 3.937008])     # 190 mm wide, 100 mm tall 
    lm, bm, wd, ht = 0.10, 0.775, 0.80, 0.200
    ax1 = fig.add_axes([lm, bm, wd, ht])
    kh = 2.70
    ax2 = fig.add_axes([lm,
                        bm - kh * ht - 0.10,
                        wd / 2 - 0.05,
                        kh * ht - 0.05])
    ax3 = fig.add_axes([lm + wd / 2 + 0.03,
                        bm - kh * ht - 0.15,
                        wd / 2 + 0.05,
                        kh *ht + 0.10],
                        projection="3d", fc="none")
    axlabfs, tiklabfs = 12, 11
    sig_clr, sur_clr = "MediumTurquoise", "GoldenRod"


    # get the Roessler trajectory
    print("Roessler trajectory ...")
    t = np.linspace(0., 1000., 10001)
    x0 = (.0, -1., 1.)
    params = (0.38, 0.4, 4.820)
    pos = tm.roessler(x0, t, params)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # get a phase randomised surrogate
    print("iAAFT surrogate ...")
    s = rc.surrogates(x.copy(), ns=1, method="iaaft").squeeze()

    # get the power spectra
    print("Power spectra ...")
    px = np.abs(np.fft.rfft(x)) ** 2
    fx = np.fft.rfftfreq(n=x.size, d=np.mean(np.diff(t)))
    ps = np.abs(np.fft.rfft(s)) ** 2
    fs = np.fft.rfftfreq(n=s.size, d=np.mean(np.diff(t)))

    # embed measured signal
    print("embed ...")
    m, tau = 9, 30
    ex = rc.embed(x, m, tau)
    es = rc.embed(s, m, tau)

    # plot
    print("plot ...")
    ax1.plot(t, x, "-",
             c=sig_clr, alpha=0.75, label="Signal 1",
             )
    ax1.plot(t, s, "-",
             c=sur_clr, alpha=0.75, label="Signal 2",
             )
    ax2.plot(fx, px, ".", ms=0.75,
             c=sig_clr, alpha=0.50, label="Signal 1",
             rasterized=True,
             )
    ax2.plot(fs, ps, ".", ms=0.75,
             c=sur_clr, alpha=0.50, label="Signal 2",
             rasterized=True,
             )
    ax3.plot(ex[:, 0], ex[:, 1], ex[:, 2], ".", c=sig_clr,
             label="Signal 1", alpha=0.5, ms=0.5,
             rasterized=True,
             )
    ax3.plot(es[:, 0], es[:, 1], es[:, 2], ".", c=sur_clr,
             label="Signal 2", alpha=0.5, ms=0.5,
             rasterized=True,
             )

    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs)
    # ax1
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(-15., 25.1)
    ax1.set_yticks(np.arange(-10., 10.1, 10))
    leg = ax1.legend(loc="upper left", ncol=2,
                     bbox_to_anchor=[0.60, 1.05])
    for txt in leg.get_texts():
        txt.set_size(tiklabfs)
    ax1.set_xlabel("Time $t$ (au)", fontsize=axlabfs, labelpad=5.)
    ax1.set_ylabel("Signal (au)", fontsize=axlabfs, labelpad=5.)
    ax1.text(0.01, 0.87,
             "A",
             ha="left", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax1.transAxes
             )
    # ax2
    ax2.set_yscale("log")
    ax2.set_ylim(1E-1, 1E9)
    ax2.set_xlim(fx[0], fx[-1])
    ax2.set_xlabel("Frequency (au)", fontsize=axlabfs, labelpad=5.)
    ax2.set_ylabel("Power (au)", fontsize=axlabfs, labelpad=5.)
    ax2.text(0.01, 0.95,
             "B",
             ha="left", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax2.transAxes
             )
    # ax3
    ax3.set_xlim(-15., 15.)
    ax3.set_ylim(-15., 15.)
    ax3.set_zlim(-10., 10.)
    ax3.view_init(elev=20., azim=245.)
    ax3.xaxis.set_pane_color((1., 1. ,1., 1.))
    ax3.zaxis.gridlines.set_alpha(0.5)
    ax3.yaxis.set_pane_color((1., 1. ,1., 1.))
    ax3.zaxis.gridlines.set_alpha(0.5)
    ax3.zaxis.set_pane_color((1., 1. ,1., 1.))
    ax3.zaxis.gridlines.set_alpha(0.5)
    ax3.set_xlabel(r"X-component $x$ (au)", fontsize=axlabfs, labelpad=5.)
    ax3.set_ylabel(r"Y-component $y$ (au)", fontsize=axlabfs, labelpad=5.)
    ax3.set_zlabel(r"Z-component $z$ (au)", fontsize=axlabfs, labelpad=5.)
    ax3.set_xticks(np.arange(-10., 10.1, 10.))
    ax3.set_yticks(np.arange(-10., 10.1, 10.))
    fig.text(0.50, 0.60,
             "C",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             )

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

