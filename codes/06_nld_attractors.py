#! /usr/bin/env python3
"""
Plots the Lorenz butterfly and the Henon strange attractors
===========================================================

"""

# Created: Sat Dec 15, 2018  03:41pm
# Last modified: Mon Sep 02, 2019  04:10pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

import toymodels as tm

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]
pl.rcParams["font.serif"] = ["cm10"]

if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    # fig = pl.figure(figsize=[7.480315, 5.314961])     # 190 mm wide, 135 mm tall 
    # fig = pl.figure(figsize=[7.480315, 4.724409])     # 190 mm wide, 120 mm tall 
    fig = pl.figure(figsize=[7.480315, 3.937008])     # 190 mm wide, 100 mm tall 
    lm, bm, wd, ht = 0.05, 0.150, 0.400, 0.80
    kh = 1.40
    ax1 = fig.add_axes([lm, bm - 0.025, wd, ht], projection="3d")
    ax2 = fig.add_axes([lm + kh * wd, bm, wd - 0.050, ht - 0.050])
    axlabfs, tiklabfs = 12, 11
    clrs = ["MediumTurquoise", "GoldenRod", "IndianRed"]


    # Lorenz butterfly
    # ----------------
    print("Roessler ...")
    Tx = np.linspace(0., 1000., 100000.)
    params = (0.432, 2., 4.)
    x0 = {
            1: (1., 3., 5.),
            2: (7.5, -4.5, 1.),
            3: (-2.5, -4.5, 0.),
            }
    X = {}
    for i in range(1, 4):
        print("\tinitial condition #%d" % i)
        X[i] = tm.roessler(x0[i], Tx, params)

    # Henon map
    # ---------
    print("Henon map ...")
    Ty = np.arange(0, 10000, 1)
    params = (1.4, 0.30)
    y0 = {
            1: (-1., 0.),
            2: (1., -0.1),
            3: (0.5, 0.1),
            }
    Y = {}
    for i in range(1, 4):
        print("\tinitial condition #%d" % i)
        Y[i] = tm.henon(y0[i], Ty[-1], params)

    # plot
    print("plot ...")
    print("\tRoessler ...")
    for i in range(1,4):
        pos = X[i]
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                 c=clrs[i - 1], ls="-", lw=0.35, alpha=0.75,
                 clip_on=False
                 )
        ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2],
                    s=50, c=clrs[i - 1], alpha=0.75,
                    marker="x",
                    )
    print("\tHenon map ...")
    for i in range(1,4):
        pos = Y[i]
        ax2.plot(pos[:, 0], pos[:, 1], "o",
                 mfc=clrs[i - 1], mec="none", alpha=0.75, ms=0.50,
                 rasterized=True,
                 )
        ax2.scatter(pos[0, 0], pos[0, 1],
                    s=50, c=clrs[i - 1], alpha=0.75,
                    marker="x",
                    )



    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs)
    ## Roessler
    # ax1.set_xlim(-15., 15.)
    # ax1.set_ylim(-15., 15.)
    ax1.set_zlim(0., 10.)
    # ax1.set_xticks(np.arange(-15., 15.1, 7.5))
    # ax1.set_yticks(np.arange(-15., 15.1, 7.5))
    # ax1.set_zticks(np.arange(0., 20.1, 5.))
    ax1.view_init(elev=15., azim=-135.)
    ax1.xaxis.set_pane_color((1., 1. ,1., 1.))
    ax1.zaxis.gridlines.set_alpha(0.5)
    ax1.yaxis.set_pane_color((1., 1. ,1., 1.))
    ax1.zaxis.gridlines.set_alpha(0.5)
    ax1.zaxis.set_pane_color((1., 1. ,1., 1.))
    ax1.zaxis.gridlines.set_alpha(0.5)
    ax1.set_xlabel(r"R{\"o}ssler $x$ (au)",
                   fontsize=axlabfs, labelpad=5.)
    ax1.set_ylabel(r"R{\"o}ssler $y$ (au)",
                   fontsize=axlabfs, labelpad=5.)
    ax1.set_zlabel(r"R{\"o}ssler $z$ (au)",
                   fontsize=axlabfs, labelpad=5.)
    ## Henon map
    ax2.set_xlabel(r"H{\'e}non $x_t$ (au)", fontsize=axlabfs, labelpad=5.)
    ax2.set_ylabel(r"H{\'e}non $y_t$ (au)", fontsize=axlabfs, labelpad=5.)


    # subplot labels
    labA = fig.text(0.025, 0.87,
             "A",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             )
    ax2.text(-0.20, 0.97,
             r"B",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax2.transAxes
             )

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

