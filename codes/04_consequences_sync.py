#! /usr/bin/env python3
"""
Plots consequences of synchronization
=====================================

"""

# Created: Sat Dec 15, 2018  05:13pm
# Last modified: Tue Nov 19, 2019  02:47pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import hilbert

import toymodels as tm

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
    ax3 = fig.add_axes([lm + wd / 2 + 0.05,
                        bm - kh * ht - 0.10,
                        wd / 2 - 0.05,
                        kh *ht - 0.05])
    axlabfs, tiklabfs = 12, 11
    splabs = {
                ax1: "A",
                ax2: "B",
                ax3: "C",
            }
    sys1_clr, sys2_clr = "MediumTurquoise", "GoldenRod"


    # time vector
    t = np.linspace(0., 500., 500001)
    n = len(t)
    neq = 50000

    # coupled Roessler
    # -----------------
    print("coupled Roessler ...")
    a = 0.15
    b = 0.2
    c = 8.5
    nu = 0.020
    mu = 0.035
    # get integrated coupled-roessler trajectories (as in the paper by
    # Rosenblum, Pikovsky, and Kurths)
    params = (a, b, c, nu, mu)
    X0 = (0.0, -1., 1., 0.1, -2., 5,)
    pos = tm.coupled_roessler(X0, t, params)
    x1, y1, z1 = pos[neq:, 0], pos[neq:, 1], pos[neq:, 2]
    x2, y2, z2 = pos[neq:, 3], pos[neq:, 4], pos[neq:, 5]
    t = t[neq:]

    # get the Hilbert transform and the instantaneous phases
    print("phases from Hilbert transform ...")
    phi1 = np.unwrap(np.angle(hilbert(x1[::1])))
    phi2 = np.unwrap(np.angle(hilbert(x2[::1])))
    # pl.hist(phi1-phi2, bins="fd")
    # pl.show()
    # import sys
    # sys.exit()

    # correlation of amplitudes
    r = (((x1 - x1.mean()) / x1.std()) * ((x2 - x2.mean()) / x2.std())).mean()
    # plot
    print("plot ...")
    ax1.plot(t, x1, "-", c=sys1_clr, alpha=0.75, lw=0.75,
             label=r"R{\"o}ssler $x_1$")
    ax1.plot(t, x2, "-", c=sys2_clr, alpha=0.75, lw=0.75,
             label=r"R{\"o}ssler $x_2$")
    ax2.plot(x1, x2, ".", c="IndianRed",
             ms=0.75, mec="none", alpha=0.50,
             rasterized=True)
    ax3.plot(phi1, phi2, ".", c="IndianRed",
             ms=0.75, mec="none", alpha=0.25,
             rasterized=True)

    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs)
        if ax == ax1:
            xpos = 0.05
            ypos = 0.85
        else:
            xpos = 0.10
            ypos = 0.90
        ax.text(xpos, ypos,
                splabs[ax],
                ha="right", va="center",
                fontsize=axlabfs, fontweight="bold", family="sans-serif",
                usetex=False,
                transform=ax.transAxes
                )
    ax1.set_ylim(-15., 25.)
    ax1.set_yticks(np.arange(-10., 10.1, 10))
    leg = ax1.legend(loc="upper left", ncol=2,
                     bbox_to_anchor=[0.55, 1.05])
    for txt in leg.get_texts():
        txt.set_size(tiklabfs)
    ax1.set_xlabel(r"Time $t$ (au)", fontsize=axlabfs, labelpad=5.)
    ax1.set_ylabel(r"R{\"o}ssler $x$ (au)",
                   fontsize=axlabfs, labelpad=5.)
    ax1.set_xlim(t[0], t[-1])
    ax2.set_xlabel(r"$x$-component of system 1 $x_1$ (au)",
                   fontsize=axlabfs, labelpad=5.)
    ax2.set_ylabel(r"$x$-component of system 2 $x_2$ (au)",
                   fontsize=axlabfs, labelpad=5.)
    ax2.set_xticks(np.arange(-10., 10.1, 5))
    ax2.set_yticks(np.arange(-10., 10.1, 5))
    ax2.set_xlim(-15., 15.)
    ax2.set_ylim(-15., 15.)
    ax3.set_xlabel(r"Phase of system 1 $\phi_1$ (rad)",
                   fontsize=axlabfs, labelpad=5.)
    ax3.set_ylabel(r"Phase of system 2 $\phi_2$ (rad)",
                   fontsize=axlabfs, labelpad=5.)
    ax3.set_xticks(np.arange(0., 501., 100.))
    ax3.set_yticks(np.arange(0., 501., 100.))
    ax3.set_xlim(0., 500.)
    ax3.set_ylim(0., 500.)

    # add text box for correlation between amplitudes
    ax2.text(0.175, 0.10,
             "{\it r} = %.3f" % r,
             ha="center", va="center",
             fontsize=tiklabfs, fontweight="normal", color="k",
             transform=ax2.transAxes,
             bbox={"facecolor": "w", "pad": 5., "alpha": 0.50}
             )

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

