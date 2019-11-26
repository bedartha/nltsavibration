#! /usr/bin/env python3
"""
Plots consequences of synchronization
=====================================

"""

# Created: Thu Jan 17, 2019  03:28pm
# Last modified: Tue Nov 26, 2019  12:16pm
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
    lm, bm, wd, ht = 0.15, 0.12, 0.65, 0.80
    ax1 = fig.add_axes([lm, bm, wd, ht])
    ax2 = ax1.twinx()
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"

    # # coupled Roessler systems
    # # -----------------
    # print("coupled Roessler ...")
    # # time vector
    # t = np.linspace(0., 500., 100001)
    # # t = np.linspace(0.,1000., 500001)
    # n = len(t)
    # # parameters
    # a = 0.15
    # b = 0.2
    # c = 8.5
    # nu = 0.02
    # neq = 50000
    # every = 100
    # # coupling constant array
    # NS = 100        # number of samples
    # N = 100
    # mu = np.linspace(0., 0.12, N)

    # # get time series
    # print("coupled Roessler time series ...")
    # X1, X2 = [np.zeros((N, NS, neq / every)) for i in range(2)]
    # pb = _progressbar_start(max_value=(N * NS), pbar_on=True)
    # k = 0
    # X00 = np.array([0.0, -1., 1., 0.1, -2., 5])
    # randmult = np.random.rand(N, NS, 6)
    # for j in range(NS):
    #     for i in range(N):
    #         # get integrated coupled-roessler trajectories (as in the paper by
    #         # Rosenblum, Pikovsky, and Kurths)
    #         params = (a, b, c, nu, mu[i])
    #         X0 = X00 * randmult[i, j]
    #         # X0 = (0.0, -1., 1., 0.1, -2., 5,)
    #         pos = tm.coupled_roessler(X0, t, params)
    #         # x1, y1, z1 = pos[:, 0], pos[:, 1], pos[:, 2]
    #         # x2, y2, z2 = pos[:, 3], pos[:, 4], pos[:, 5]
    #         X1[i, j, :], X2[i, j, :] = pos[-neq::every, 0], pos[-neq::every, 3]
    #         _progressbar_update(pb, k)
    #         k += 1
    # _progressbar_finish(pb)

    # # embedding parameters
    # print("embedding parameters ...")
    # m, tau = [np.zeros((N, NS), dtype="int") for i in range(2)]
    # maxlag = 100
    # maxdim = 10
    # R = 0.025
    # pb = _progressbar_start(max_value=(N * NS), pbar_on=True)
    # k = 0
    # for j in range(NS):
    #     for i in range(N):
    #         # get mi
    #         mi1, mi_lags1 = rc.mi(X1[i, j, :], maxlag, pbar_on=False)
    #         mi_filt1, _ = utils.boxfilter(mi1, filter_width=3, estimate="mean")
    #         tau1 = rc.first_minimum(mi_filt1)
    #         mi2, mi_lags2 = rc.mi(X2[i, j, :], maxlag, pbar_on=False)
    #         mi_filt2, _ = utils.boxfilter(mi2, filter_width=3, estimate="mean")
    #         tau2 = rc.first_minimum(mi_filt2)
    #         tau[i, j] = int(max(tau1, tau2))
    #         # FNN
    #         fnn1, dims1 = rc.fnn(X1[i, j, :], tau[i, j],
    #                              maxdim=maxdim, r=R, pbar_on=False)
    #         m1 = dims1[rc.first_zero(fnn1)]
    #         fnn2, dims2 = rc.fnn(X2[i, j, :], tau[i, j],
    #                              maxdim=maxdim, r=R, pbar_on=False)
    #         m2 = dims2[rc.first_zero(fnn2)]
    #         m[i, j] = int(max(m1, m2))
    #         _progressbar_update(pb, k)
    #         k += 1
    # _progressbar_finish(pb)

    # # save this data temporarily
    # FN = "../data/cpr_rmd/rp_dependence_N%d_NS%d.npz" % (N, NS)
    # np.savez(FN, X1=X1, X2=X2, m=m, tau=tau, mu=mu)
    # print("data saved to: %s" % FN)
    # sys.exit()

    # load data
    N, NS = 100, 100
    FN = "../data/cpr_rmd/rp_dependence_N%d_NS%d.npz" % (N, NS)
    dat = np.load(FN)
    X1, X2 = dat["X1"], dat["X2"]
    m, tau = dat["m"], dat["tau"]
    mu = dat["mu"]
    N = len(mu)

    # recurrence plots and CPR
    print("RPs and CPR and RMD ...")
    CPR = np.zeros((N, NS))
    RMD = np.zeros((N, NS))
    PCC = np.zeros((N, NS))
    e_cpr = 0.20
    e_rmd = 0.25
    pb = _progressbar_start(max_value=(N * NS), pbar_on=True)
    k = 0
    for j in range(NS):
        for i in range(N):
            # pl.clf()
            # pl.plot(X1[i], X2[i], "k.", alpha=0.5)
            # # X = rc.embed(X1[i], m=m[i], tau=tau[i])
            # # pl.plot(X[:, 0],X[:, 1])
            # pl.show()
            # sys.exit()
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
            _progressbar_update(pb, k)
            k += 1
    _progressbar_finish(pb)

    CPRlo = np.nanpercentile(CPR, 25., axis=1)
    CPRhi = np.nanpercentile(CPR, 75., axis=1)
    RMDlo = np.nanpercentile(RMD, 25., axis=1)
    RMDhi = np.nanpercentile(RMD, 75., axis=1)
    PCClo = np.nanpercentile(PCC, 25., axis=1)
    PCChi = np.nanpercentile(PCC, 75., axis=1)

    # plot
    ax1.fill_between(mu, PCClo, PCChi, color=clr3, label=r"$PCC$", alpha=0.5)
    ax1.fill_between(mu, CPRlo, CPRhi, color=clr1, label=r"$CPR$", alpha=0.5)
    ax2.fill_between(mu, RMDlo, RMDhi, color=clr2, label=r"$RMD$", alpha=0.5)
    # ax1.plot(mu, CPR, "-", c=clr1, label=r"$CPR$")
    # ax2.plot(mu, RMD, "-", c=clr2, label=r"$RMD$")
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
    # leg1 = ax1.legend(loc="upper right")
    # leg2 = ax2.legend(loc="lower right")
    # for leg in [leg1, leg2]:
    #     for txt in leg.get_texts():
    #         txt.set_size(tiklabfs)
    # for ax, clr in zip([ax1, ax2, ax3], [clr3, clr1, clr2]):
    #     ax.tick_params(color=clr, labelcolor=clr, axis="y")
    # ax2.tick_params(which="y", labelleft="off", labelright="on",
    #                 left="off", right="on")
    # lims = [(-0.6, 0.5), (0.4, 1.0), (1., 9.)]
    # for ax, lim in zip([ax1, ax2, ax3], lims):
    #     ax.set_ylim(lim)
    # ax3.set_yticks(np.arange(1., 10.01, 2.))
    # ax1.set_xlim(0.95, 1.45)
    ax1.set_xlabel(r"Coupling strength $\mu$ (au)", fontsize=axlabfs)

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

