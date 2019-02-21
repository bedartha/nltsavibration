#! /usr/bin/env python
"""
Script to generate Figure 11: Inferring dependencies using recurrences
======================================================================

"""

# Created: Thu Jan 17, 2019  03:28pm
# Last modified: Thu Feb 21, 2019  08:14pm
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


# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]


def _get_timeseries():
    """Generates time series for coupled Roessler system and saves output.
    """
    # coupled Roessler systems
    # -----------------
    utils._printmsg("coupled Roessler ...", args.verbose)
    # time vector
    t = np.linspace(0., 500., 100001)
    # t = np.linspace(0.,1000., 500001)
    n = len(t)
    # parameters
    a = 0.15
    b = 0.2
    c = 8.5
    nu = 0.02
    neq = 50000
    every = 100
    # coupling constant array
    mu = np.linspace(0., 0.12, NM)

    # get time series
    utils._printmsg("coupled Roessler time series ...", args.verbose)
    X1, X2 = [np.zeros((NM, NS, neq / every)) for i in range(2)]
    pb = utils._progressbar_start(max_value=(NM * NS), pbar_on=args.verbose)
    k = 0
    X00 = np.array([0.0, -1., 1., 0.1, -2., 5])
    randmult = np.random.rand(NM, NS, 6)
    for j in range(NS):
        for i in range(NM):
            # get integrated coupled-roessler trajectories (as in the paper by
            # Rosenblum, Pikovsky, and Kurths)
            params = (a, b, c, nu, mu[i])
            X0 = X00 * randmult[i, j]
            pos = tm.coupled_roessler(X0, t, params)
            X1[i, j, :], X2[i, j, :] = pos[-neq::every, 0], pos[-neq::every, 3]
            utils._progressbar_update(pb, k)
            k += 1
    utils._progressbar_finish(pb)

    # save output
    FN = DATPATH + __file__[5:-3]
    FN += "_NM%d_NS%d_timeseries.npz" % (NM, NS)
    np.savez(FN, X1=X1, X2=X2, mu=mu)
    print("data saved to: %s" % FN)

    return None


def _get_embedding():
    """
    Loads coupled time series data and estimates embedding parameters.
    """
    # load data
    utils._printmsg("load data ...", args.verbose)
    FN = DATPATH + __file__[5:-3]
    FN += "_NM%d_NS%d_timeseries.npz" % (NM, NS)
    DAT = np.load(FN)
    X1, X2 = DAT["X1"], DAT["X2"]
    mu = DAT["mu"]

    # embedding parameters
    utils._printmsg("embedding parameters ...", args.verbose)
    m, tau = [np.zeros((NM, NS), dtype="int") for i in range(2)]
    maxlag = 100
    maxdim = 10
    R = 0.025
    pb = utils._progressbar_start(max_value=(NM * NS), pbar_on=args.verbose)
    k = 0
    for j in range(NS):
        for i in range(NM):
            # get mi
            mi1, mi_lags1 = rc.mi(X1[i, j, :], maxlag, pbar_on=False)
            mi_filt1, _ = utils.boxfilter(mi1, filter_width=3, estimate="mean")
            tau1 = rc.first_minimum(mi_filt1)
            mi2, mi_lags2 = rc.mi(X2[i, j, :], maxlag, pbar_on=False)
            mi_filt2, _ = utils.boxfilter(mi2, filter_width=3, estimate="mean")
            tau2 = rc.first_minimum(mi_filt2)
            tau[i, j] = int(max(tau1, tau2))
            # FNN
            fnn1, dims1 = rc.fnn(X1[i, j, :], tau[i, j],
                                 maxdim=maxdim, r=R, pbar_on=False)
            m1 = dims1[rc.first_zero(fnn1)]
            fnn2, dims2 = rc.fnn(X2[i, j, :], tau[i, j],
                                 maxdim=maxdim, r=R, pbar_on=False)
            m2 = dims2[rc.first_zero(fnn2)]
            m[i, j] = int(max(m1, m2))
            utils._progressbar_update(pb, k)
            k += 1
    utils._progressbar_finish(pb)

    # save output
    utils._printmsg("save output ...", args.verbose)
    FN = DATPATH + __file__[5:-3]
    FN += "_NM%d_NS%d_embedding.npz" % (NM, NS)
    np.savez(FN, X1=X1, X2=X2, m=m, tau=tau, mu=mu)
    utils._printmsg("data saved to: %s" % FN, args.verbose)

    return None


def _get_cprrmd():
    """
    Estimates CPR, RMD, and PCC from time series and embedding parameters.
    """
    # load data
    utils._printmsg("load data ...", args.verbose)
    FN = DATPATH + __file__[5:-3]
    FN += "_NM%d_NS%d_embedding.npz" % (NM, NS)
    DAT = np.load(FN)
    X1, X2 = DAT["X1"], DAT["X2"]
    m, tau = DAT["m"], DAT["tau"]
    mu = DAT["mu"]

    # recurrence plots and CPR
    utils._printmsg("RPs and CPR and RMD ...", args.verbose)
    CPR = np.zeros((NM, NS))
    RMD = np.zeros((NM, NS))
    PCC = np.zeros((NM, NS))
    e_cpr = 0.20
    e_rmd = 0.25
    pb = utils._progressbar_start(max_value=(NM * NS), pbar_on=args.verbose)
    k = 0
    for j in range(NS):
        for i in range(NM):
            R1 = rc.rp(X1[i, j, :], m=m[i, j], tau=tau[i, j], e=e_cpr,
                       norm="euclidean", threshold_by="frr", normed=True)
            R2 = rc.rp(X2[i, j, :], m=m[i, j], tau=tau[i, j], e=e_cpr,
                       norm="euclidean", threshold_by="frr", normed=True)
            CPR[i, j] = rqa.cpr(R1, R2)
            del R1, R2
            R1 = rc.rp(X1[i, j, :], m=m[i, j], tau=tau[i, j], e=e_rmd,
                       norm="euclidean", threshold_by="distance", normed=True)
            R2 = rc.rp(X2[i, j, :], m=m[i, j], tau=tau[i, j], e=e_rmd,
                       norm="euclidean", threshold_by="distance", normed=True)
            RMD[i, j] = rqa.rmd(R1, R2)
            del R1, R2
            PCC[i, j] = np.corrcoef(X1[i, j, :], X2[i, j, :])[0, 1]
            utils._progressbar_update(pb, k)
            k += 1
    utils._progressbar_finish(pb)

    # save output
    utils._printmsg("save output ...", args.verbose)
    FN = DATPATH + __file__[5:-3]
    FN += "_NM%d_NS%d_cprrmd.npz" % (NM, NS)
    np.savez(FN, CPR=CPR, RMD=RMD, PCC=PCC, mu=mu)
    utils._printmsg("data saved to: %s" % FN, args.verbose)

    return None


def _get_fig():
    """
    Plots Figure 11 based on using esimtated CPR, RMD, and PCC arrays.
    """
    # load data
    utils._printmsg("load data ...", args.verbose)
    FN = DATPATH + __file__[5:-3]
    FN += "_NM%d_NS%d_cprrmd.npz" % (NM, NS)
    DAT = np.load(FN)
    CPR, RMD, PCC  = DAT["CPR"], DAT["RMD"], DAT["PCC"]
    mu = DAT["mu"]

    # get percentiles
    utils._printmsg("get percentiles ...", args.verbose)
    CPRlo = np.nanpercentile(CPR, 25., axis=1)
    CPRhi = np.nanpercentile(CPR, 75., axis=1)
    RMDlo = np.nanpercentile(RMD, 25., axis=1)
    RMDhi = np.nanpercentile(RMD, 75., axis=1)
    PCClo = np.nanpercentile(PCC, 25., axis=1)
    PCChi = np.nanpercentile(PCC, 75., axis=1)

    # set up figure
    utils._printmsg("set up figure ...", args.verbose)
    fig = pl.figure(figsize=[7.480315, 3.937008])   # 140 mm wide, 100 mm tall 
    lm, bm, wd, ht = 0.15, 0.12, 0.65, 0.80
    ax1 = fig.add_axes([lm, bm, wd, ht])
    ax2 = ax1.twinx()
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"

    # plot
    utils._printmsg("plot ...", args.verbose)
    ax1.fill_between(mu, PCClo, PCChi, color=clr3, label=r"$PCC$", alpha=0.5)
    ax1.fill_between(mu, CPRlo, CPRhi, color=clr1, label=r"$CPR$", alpha=0.5)
    ax2.fill_between(mu, RMDlo, RMDhi, color=clr2, label=r"$RMD$", alpha=0.5)
    ax2.axvline(0.042, ls="--", c="0.5", zorder=10)
    ax2.axvline(0.075, ls="--", c="0.5", zorder=10)
    ax2.axvline(0.102, ls="--", c="0.5", zorder=10)

    # prettify
    for ax in fig.axes:
        ax.tick_params(labelsize=tiklabfs, size=6)
    ax1.text(-0.1, 0.42, r"$CPR$,", color=clr1, fontsize=axlabfs,
             ha="center", va="center", rotation=90, transform=ax1.transAxes)
    ax1.text(-0.1, 0.57, r"$PCC$", color=clr3, fontsize=axlabfs,
             ha="center", va="center", rotation=90, transform=ax1.transAxes)
    ax2.set_ylabel(r"$RMD$", fontsize=axlabfs, color=clr2)
    ax1.tick_params(axis="y", labelcolor=clr1, color=clr1)
    ax2.tick_params(axis="y", labelcolor=clr2, color=clr2)
    ax1.set_xlabel(r"Coupling strength $\mu$", fontsize=axlabfs)

    # save figure
    utils._printmsg("save figure ...", args.verbose)
    FN = "../figs/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    utils._printmsg("figure saved to: %s" % FN, args.verbose)

    return None


def _parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--get",
                        type=str, choices=[
                                           "timeseries",
                                           "embedding",
                                           "cprrmd",
                                           "fig",
                                           ],
                        help="Specify what to get")
    parser.add_argument("-ns", "--nsamp",
                        type=int, default=100,
                        help="Sample size, i.e., number of initial conditions")
    parser.add_argument("-nm", "--n-mu",
                        type=int, default=100,
                        help="Number of points in coupling constant array")
    parser.add_argument("-v", "--verbose",
                        type=bool, default=False,
                        help="Print verbose messages and progress bars")
    parser.add_argument("-ft", "--figtype",
                        type=str, default="png",
                        help="File type of output figure")
    return parser.parse_args()


if __name__ == "__main__":
    DATPATH = "../data/cprrmd/"
    args = _parse_args()
    NS, NM = args.nsamp, args.n_mu
    _func = eval("_get_%s" % args.get)
    _func()
