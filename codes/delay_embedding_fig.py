#! /usr/bin/env python3
"""
Plots consequences of synchronization
=====================================

"""

# Created: Sat Dec 15, 2018  05:13pm
# Last modified: Wed Dec 04, 2019  02:15pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

import toymodels as tm
import recurrence as rc
import utils

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]


def _get_data():
    """
    Runs the analysis and saves the data to be used later for plotting.
    """
    # get the Roessler trajectory
    print("Roessler trajectory ...")
    t = np.linspace(0., 1000., 100000.)
    params = (0.432, 2., 4.)
    x0 = (1., 3., 5.)
    pos = tm.roessler(x0, t, params)
    i_eq = 90000
    x, y, z = pos[i_eq:, 0], pos[i_eq:, 1], pos[i_eq:, 2]
    t = t[i_eq:]

    # set the X component as our measured signal
    s = x.copy()

    # get mi
    print("MI ...")
    maxlag = np.where(t <= (t[0] + 10.))[0][-1].astype("int")
    mi, mi_lags = rc.mi(s, maxlag)
    mi_filt, _ = utils.boxfilter(mi, filter_width=25, estimate="mean")
    tau_mi = rc.first_minimum(mi_filt)
    print("FNN ...")
    M = 10
    R = 0.50
    fnn_mi, dims_mi = rc.fnn(s, tau_mi, maxdim=M, r=R)
    m_mi = dims_mi[rc.first_zero(fnn_mi)]

    # save data
    print("save output ...")
    FN = "../data/delay_embedding/results"
    np.savez(FN,
            x=x, y=y, z=z, t=t, params=params, x0=x0, i_eq=i_eq,
            s=s, maxlag=maxlag, mi=mi, mi_lags=mi_lags, mi_filt=mi_filt,
            tau_mi=tau_mi,
            M=M, R=R, fnn_mi=fnn_mi, dims_mi=dims_mi,
            m_mi=m_mi
            )
    print("saved to: %s.npz" % FN)
    return None


def _get_fig():
    """
    Loads saved data and plots the figure.
    """
    print("load data ...")
    data = np.load("../data/delay_embedding/results.npz")
    s = data["s"]
    x, y, z, t = data["x"], data["y"], data["z"], data["t"]
    mi_lags, mi, tau_mi = data["mi_lags"], data["mi"], data["tau_mi"]
    M, R = data["M"], data["R"]
    dims_mi, fnn_mi, m_mi = data["dims_mi"], data["fnn_mi"], data["m_mi"]


    # set up figure
    print("set up figure ...")
    fig = pl.figure(figsize=[7.480315, 5.905512])     # 190 mm wide, 150 mm tall 
    wd, ht = 0.375, 0.375
    lm1, lm2 = 0.10, 0.55
    bm1, bm2 = 0.55, 0.05
    ax1 = fig.add_axes([lm1 + 0.025, bm1, wd - 0.050, ht])
    ax2 = fig.add_axes([lm2 + 0.040, bm1, wd - 0.050, ht])
    ax3 = fig.add_axes([lm1, bm2, wd, ht], projection="3d")
    ax4 = fig.add_axes([lm2, bm2, wd, ht], projection="3d")
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"



    # get mi
    print("MI ...")
    ax1.plot(mi_lags, mi, "-", c=clr1)
    ax1.plot(mi_lags[tau_mi], mi[tau_mi], "o", c=clr3)
    ax1.axvline(mi_lags[tau_mi], ls="--", c=clr2)
    ax1.text(mi_lags[tau_mi] + 20, 4.80,
             r"$\tau_e = %d$" % int(mi_lags[tau_mi]),
             fontsize=tiklabfs,
             )
    ax1_ = ax1.twiny()
    sampres = np.mean(np.diff(t))
    ax1_.plot(mi_lags * sampres, mi, "-", color="none")
    ax1.text(-0.21, 0.50,
             r"Self mutual information",
             fontsize=axlabfs, ha="center", va="center", rotation=90,
             transform=ax1.transAxes
             )
    ax1.text(-0.13, 0.50,
             r"$I(\tau)$ (bits)",
             fontsize=axlabfs, ha="center", va="center", rotation=90,
             transform=ax1.transAxes
             )
    ax1.set_xlabel(r"Delay $\tau$ (per unit sampling time)", fontsize=axlabfs)
    ax1_.set_xlabel(r"$\tau \Delta t$", fontsize=axlabfs)


    # FNN
    print("FNN ...")
    ax2.plot(dims_mi, fnn_mi, "o-", c=clr1, mfc="none")
    ax2.plot(3, fnn_mi[2], "o", c=clr3)
    ax1.axvline(mi_lags[tau_mi], ls="--", c=clr2)
    ax2.axvline(3, ls="--", c=clr2)
    ax2.text(3.5, 0.9,
             r"$m_e = 3$",
             fontsize=tiklabfs
             )
    ttl1 = r"Fraction false neighbours"
    ttl2 = r"$FNN(m)$"
    ax2.text(-0.25, 0.5,
             ttl1,
             fontsize=axlabfs, ha="center", va="center", rotation=90,
             transform=ax2.transAxes
             )
    ax2.text(-0.17, 0.5,
             ttl2,
             fontsize=axlabfs, ha="center", va="center", rotation=90,
             transform=ax2.transAxes
             )
    ax2.set_xlabel(r"Embedding dimension $m$", fontsize=axlabfs)

    # true attractor
    ax3.plot(x, y, z, "-", color=clr2, lw=0.75)

    # reconstructed attractor
    print("reconstructed attractor ...")
    print("\tembedding parameters: m = %d, tau = %d" % (m_mi, tau_mi))
    ra = rc.embed(s, m_mi, tau_mi)
    ax4.plot(ra[:, 0], ra[:, 1], ra[:, 2], "-", color=clr1, lw=0.75)


    # prettify figure
    for ax in fig.axes:
        ax.tick_params(size=4, labelsize=tiklabfs)
    # ax1
    ax1.text(-0.20, 0.99,
             "a",
             ha="left", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax1.transAxes
             )
    # ax2
    ax2.text(-0.20, 0.99,
             "b",
             ha="left", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             transform=ax2.transAxes
             )
    fig.text(0.075, 0.450,
             "c",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             )
    # ax4
    fig.text(0.55, 0.450,
             "d",
             ha="right", va="center",
             fontsize=axlabfs, fontweight="bold", family="sans-serif",
             usetex=False,
             )
    # ax3 and ax4
    for ax in [ax3, ax4]:
        ax.xaxis.set_pane_color((1., 1. ,1., 1.))
        ax.zaxis.gridlines.set_alpha(0.5)
        ax.yaxis.set_pane_color((1., 1. ,1., 1.))
        ax.zaxis.gridlines.set_alpha(0.5)
        ax.zaxis.set_pane_color((1., 1. ,1., 1.))
        ax.zaxis.gridlines.set_alpha(0.5)
        ax.set_xlabel(r"R{\"o}ssler $x$", fontsize=axlabfs, labelpad=5.)
        ax.set_ylabel(r"R{\"o}ssler $y$", fontsize=axlabfs, labelpad=5.)
        ax.set_zlabel(r"R{\"o}ssler $z$", fontsize=axlabfs, labelpad=5.)

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

    return None


if __name__ == "__main__":
    arg = sys.argv[1]
    _func = eval("_get_%s" % arg)
    _func()
