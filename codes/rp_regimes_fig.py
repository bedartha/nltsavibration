#! /usr/bin/env python3
"""
Plots consequences of synchronization
=====================================

"""

# Created: Tue Jan 22, 2019  12:32pm
# Last modified: Wed Dec 04, 2019  02:45pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl

import toymodels as tm
import recurrence as rc
import rqa
import igraph as ig
import utils
from utils import _progressbar_start, _progressbar_update, _progressbar_finish

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]


if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    fig = pl.figure(figsize=[7.480315, 3.937008])     # 140 mm wide, 100 mm tall 
    # fig = pl.figure(figsize=[7.480315, 5.314961])     # 190 mm wide, 135 mm tall 
    # fig = pl.figure(figsize=[7.480315, 4.724409])     # 190 mm wide, 120 mm tall 
    # fig = pl.figure(figsize=[7.480315, 5.905512])     # 190 mm wide, 150 mm tall 
    # fig = pl.figure(figsize=[7.480315, 3.937008])     # 190 mm wide, 100 mm tall 
    lm1, lm2 = 0.10, 0.30
    wd1, wd2 = 0.20, 0.60
    bm, ht = 0.12, 0.80
    ax1 = fig.add_axes([lm1, bm, wd1, ht])
    ax2 = fig.add_axes([lm2, bm, wd2, ht])
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"

    # triple-well potential
    # ---------------------
    # U(x) = x^2 (bx^2 - c)^2 + ax^2
    # b = 0.1, c = 1, a = 1.
    #
    # Langevin equation
    # -----------------
    # dx = -U'(x) dt + sig dW
    #
    # Euler-marayama method for integration
    # -------------------------------------
    # x_{n+1} = x_{n} - U'(x_{n}) dt + sig dt^0.5 eps; eps ~ N(0,1)
    #
    # References
    # ----------
    # aip.scitation.org/doi/full/10.1063/1.4768729
    # ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
    print("triple-well potential time series ...")
    a, b, c = 0., 0.5, 2.
    sig = 1.5
    dt = 0.001
    sqrt_dt = np.sqrt(dt)
    T = 1000.
    N = int(T / dt)
    t = np.linspace(0., T, N)
    x = np.zeros(N)
    eps = np.random.randn(N)
    for n in range(N - 1):
        dU = 2. * x[n] * (b * x[n] ** 2 - c) * (3.* b * x[n] ** 2 - c) + 2. * a
        x[n + 1] = x[n] - dU * dt + sig * sqrt_dt * eps[n]

    # sample the true time series
    every = 250
    xs = x[::every]
    ts = t[::every]

    # get recurrence network adjacency
    print("recurrence network ...")
    A = rc.rn(xs, m=1, tau=1, e=0.20,
              norm="euclidean", threshold_by="frr", normed=True)

    # optimize modularity
    print("optimize modularity ...")
    G = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
    dendro = G.community_fastgreedy()
    clust = dendro.as_clustering()
    mem = clust.membership
    clust_ids = np.unique(mem)

    # plot
    print("plot ...")
    # plot potential
    x_ = np.linspace(-3., 3., 1000)
    U = x_ ** 2 * (b * x_ ** 2 - c) ** 2 + a * x_ ** 2
    ax1.plot(U, x_, "-", c="0.5")
    # plot time series with time points color-coded by community membership
    clrs = [clr1, clr2, clr3, "Fuchsia", "Indigo", "Peru", "Salmon"]
    ax2.plot(t, x, "-", c="0.5", alpha=0.5, zorder=1)
    for k, clust_id in enumerate(clust_ids):
        i = mem == clust_id
        ax2.plot(ts[i], xs[i], "o", label="Cluster % d", ms=3, c=clrs[k])

    # prettify
    for ax in fig.axes:
        ax.tick_params(labelsize=tiklabfs, size=6)
        ax.set_ylim(-3.0, 3.0)
        ax.axhline(np.sqrt(c / (3. * b)), ls="--", c="k")
        ax.axhline(-np.sqrt(c / (3. * b)), ls="--", c="k")
    for ax, lab in zip([ax1, ax2], ["a", "b"]):
        ax.text(0.05, 0.98,
                lab,
                ha="left", va="top",
                fontsize=axlabfs, fontweight="bold", family="sans-serif",
                usetex=False,
                transform=ax.transAxes
                )
    ax1.set_xlim(10., 0.)
    ax1.set_xlabel(r"Potential $U(x)$", fontsize=axlabfs)
    ax1.set_ylabel(r"Displacement $x$", fontsize=axlabfs)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel("Time $t$", fontsize=axlabfs)
    ax2.set_ylabel(r"Displacement $x$", fontsize=axlabfs)
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(labelleft="off", right="on", labelright="on")

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

