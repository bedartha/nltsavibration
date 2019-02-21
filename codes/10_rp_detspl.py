#! /usr/bin/env python
"""
Plots consequences of synchronization
=====================================

"""

# Created: Thu Jan 17, 2019  03:28pm
# Last modified: Thu Feb 21, 2019  08:08pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import argparse
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


def _get_data():
    """
    Estimates Lyapunov, DET, and SPL for Henon map.
    """
    # Henon map time series
    print("Henon map time series ...")
    t = np.arange(0, 10000, 1)
    a = np.linspace(1.0, 1.4, NA).reshape(NA, 1)
    b = 0.30
    nt= len(t)
    x, y = [np.zeros((nt, NA, NS)) for i in range(2)]
    x[0, :, :] = 1E-1 * np.random.rand(NA, NS)
    y[0, :, :] = 1E-1 * np.random.rand(NA, NS)
    pb = _progressbar_start(max_value=nt, pbar_on=True)
    LPV = np.zeros((NA, NS))
    for i in range(1, nt):
        x[i, :, :] = 1. - a * x[i - 1, :, :] ** 2 + y[i - 1, :, :]
        y[i, :, :] = b * x [i - 1, :, :]
        if i >= nt / 2:
            LPV[:, :] += np.log(np.fabs(-2. * a * x[i - 1, :, :]))
        _progressbar_update(pb, i)
    _progressbar_finish(pb)
    xH_eq = x[-NEQ:, :, :]
    LPV /= float(nt)

    # estimate embedding parameters
    print("embedding parameters ...")
    tau, m = np.ones(NA, dtype="int"), 2 * np.ones(NA, dtype="int")

    # DET
    print("DET ...")
    RR = 0.30
    DET = np.zeros((NA, NS))
    pb = _progressbar_start(max_value=NS * NA, pbar_on=True)
    k = 0
    for j in range(NS):
        for i in range(NA):
            R = rc.rp(xH_eq[:, i, j], m=m[i], tau=tau[i], e=RR,
                      norm="euclidean", threshold_by="frr")
            DET[i, j] = rqa.det(R, lmin=2, hist=None, verb=False)
            del R
            _progressbar_update(pb, k)
            k += 1
    _progressbar_finish(pb)

    # SPL
    print("SPL ...")
    SPL = np.zeros((NA, NS))
    pb = _progressbar_start(max_value=NS * NA, pbar_on=True)
    k = 0
    for j in range(NS):
        for i in range(NA):
            A = rc.rn(xH_eq[:, i, j], m=m[i], tau=tau[i], e=RR,
                      norm="euclidean", threshold_by="frr")
            G = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
            pl_hist = G.path_length_hist(directed=False)
            SPL[i, j] = pl_hist.mean
            del A, G
            _progressbar_update(pb, k)
            k += 1
    _progressbar_finish(pb)

    # save output
    FN = DATPATH + "det_spl_lpv_NA%d_NS%s_NEQ%d" % (NA, NS, NEQ)
    np.savez(FN, DET=DET, SPL=SPL, LPV=LPV, t=t, a=a, b=b, x=x, y=y)
    print("saved to %s.npz" % FN)

    return None


def _get_fig():
    """
    Loads the data from analysis and plots the figure.
    """
    # load data
    FN = DATPATH + "det_spl_lpv_na%d_ns%s_neq%d.npz" % (NA, NS, NEQ)
    DAT = np.load(FN)
    DET = DAT["DET"]
    SPL = DAT["SPL"]
    LPV = DAT["LPV"]
    a = DAT["a"]

    # set up figure
    print("set up figure ...")
    fig = pl.figure(figsize=[7.480315, 3.937008])     # 140 mm wide, 100 mm tall 
    lm, bm, wd, ht = 0.10, 0.12, 0.65, 0.80
    ax1 = fig.add_axes([lm, bm, wd, ht])
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    # Offset the right spine of ax3.  The ticks and label have already been
    # placed on the right by twinx above.
    ax3.spines["right"].set_position(("axes", 1.15))
    # Having been created by twinx, ax3 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(ax3)
    # Second, show the right spine.
    ax3.spines["right"].set_visible(True)
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"

    # plot
    LPVlo = np.percentile(LPV, 25., axis=1)
    LPVhi = np.percentile(LPV, 75., axis=1)
    DETlo = np.percentile(DET, 25., axis=1)
    DEThi = np.percentile(DET, 75., axis=1)
    SPLlo = np.percentile(SPL, 25., axis=1)
    SPLhi = np.percentile(SPL, 75., axis=1)

    # ax1: Lyapunov exponent
    k = 1.
    a = a.flatten()
    hi, lo = LPVhi, LPVlo
    ax1.fill_between(a, hi, lo,
                     color=clr3, alpha=0.5,
                     )
    ax1.axhline(0., c=clr3, ls="--", lw=0.75)
    ax1.set_ylabel(r"Lyapunov exponent $\Lambda$",
                   fontsize=axlabfs, color=clr3)
    # ax2: DET
    hi, lo = DEThi, DETlo
    ax2.fill_between(a, hi, lo,
                     color=clr1, alpha=0.5,
                     )
    ax2.set_ylabel(r"$DET$", fontsize=axlabfs, color=clr1)

    # ax3: SPL
    hi, lo = SPLhi, SPLlo
    ax3.fill_between(a, hi, lo,
                     color=clr2, alpha=0.5,
                     )
    ax3.set_ylabel(r"$SPL$", fontsize=axlabfs, color=clr2)

    # prettify
    for ax in fig.axes:
        ax.tick_params(labelsize=tiklabfs, size=6)
    for ax, clr in zip([ax1, ax2, ax3], [clr3, clr1, clr2]):
        ax.tick_params(color=clr, labelcolor=clr, axis="y")
    ax2.tick_params(which="y", labelleft="off", labelright="on",
                    left="off", right="on")
    ax2.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax3.set_yticks(np.arange(1., 5.01, 1.))
    lims = [(-0.6, 0.5), (0.35, 1.05), (1., 5.)]
    for ax, lim in zip([ax1, ax2, ax3], lims):
        ax.set_ylim(lim)
    ax1.set_xlim(0.95, 1.45)
    ax1.set_xlabel(r"$a$", fontsize=axlabfs)

    # save figure
    FN = "../figs/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

    return None


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return None


def _parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--get",
                        type=str, choices=[
                                           "data",
                                           "fig",
                                           ],
                        help="Specify what to get")
    parser.add_argument("-ns", "--nsamp",
                        type=int, default=100,
                        help="Sample size, i.e., number of initial conditions")
    parser.add_argument("-na",
                        type=int, default=500,
                        help="Number of points in parameter array")
    parser.add_argument("-neq",
                        type=int, default=1000,
                        help="Number of points from end to analyse")
    parser.add_argument("-v", "--verbose",
                        type=bool, default=False,
                        help="Print verbose messages and progress bars")
    parser.add_argument("-ft", "--figtype",
                        type=str, default="png",
                        help="File type of output figure")
    return parser.parse_args()


if __name__ == "__main__":
    DATPATH = "../data/detspl/"
    args = _parse_args()
    NA, NS, NEQ = args.na, args.nsamp, args.neq
    _func = eval("_get_%s" % args.get)
    _func()
