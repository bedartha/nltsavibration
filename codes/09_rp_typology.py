#! /usr/bin/env python
"""
Plots consequences of synchronization
=====================================

"""

# Created: Thu Jan 17, 2019  10:48am
# Last modified: Thu Jan 17, 2019  03:28pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl

import toymodels as tm
import recurrence as rc
import utils

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]

if __name__ == "__main__":
    # set up figure
    print("set up figure ...")
    fig = pl.figure(figsize=[7.480315, 5.314961])     # 190 mm wide, 135 mm tall 
    # fig = pl.figure(figsize=[7.480315, 4.724409])     # 190 mm wide, 120 mm tall 
    # fig = pl.figure(figsize=[7.480315, 5.905512])     # 190 mm wide, 150 mm tall 
    # fig = pl.figure(figsize=[7.480315, 3.937008])     # 190 mm wide, 100 mm tall 
    wd, ht = 0.375, 0.375
    lm1, lm2 = 0.05, 0.55
    bm1, bm2 = 0.585, 0.08
    ax1 = fig.add_axes([lm1, bm1, wd, ht])
    ax2 = fig.add_axes([lm2, bm1, wd, ht])
    ax3 = fig.add_axes([lm1, bm2, wd, ht])
    ax4 = fig.add_axes([lm2, bm2, wd, ht])
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"


    # A. Gaussian white noise RP
    print("A: Gaussian white noise ...")
    N = 500
    x = np.random.rand(N)
    R = rc.rp(x, m=1, tau=1, e=0.10, norm="euclidean", threshold_by="distance")
    ax1.imshow(R, cmap=pl.cm.gray_r,
               interpolation="none", origin="lower",
               rasterized=True)
    del R
    ax1.set_title(r"$\tau_e = 1, m_e = 1$, \varepsilon = 0.10",
                  fontsize=axlabfs)

    # B. Superposed harmonics RP
    print("B. Superposed harmonics RP ...")
    N = 500
    t = np.arange(N)
    T1, T2, T3 = 10., 50., 75.
    A1, A2, A3 = 1., 1.5, 2.
    # T1, T2 = 10., 50.
    # A1, A2 = 1., 1.5
    twopi = 2. * np.pi
    H1 = A1 * np.sin((twopi * t) / T1)
    H2 = A2 * np.sin((twopi * t) / T2)
    H3 = A3 * np.sin((twopi * t) / T3)
    x =  H1 + H2 + H3
    # get mi
    maxlag = 150
    mi, mi_lags = rc.mi(x, maxlag)
    mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
    tau_mi = rc.first_minimum(mi_filt)
    # FNN
    M = 20
    R = 0.025
    fnn_mi, dims_mi = rc.fnn(x, tau_mi, maxdim=M, r=R)
    m_mi = dims_mi[rc.first_zero(fnn_mi)]
    R = rc.rp(x, m=m_mi, tau=tau_mi, e=0.15,
              norm="euclidean", threshold_by="frr", normed=True)
    ax2.imshow(R, cmap=pl.cm.gray_r,
               interpolation="none", origin="lower",
               rasterized=True)
    del R
    ax2.set_title(r"$\tau_e = %d, m_e = %d, \varepsilon = 0.15$" % (tau_mi, m_mi),
                  fontsize=axlabfs)

    # C. Chaotic Roessler
    print("C. Chaotic Roessler ...")
    t = np.linspace(0., 1000., 100000.)
    params = (0.432, 2., 4.)
    x0 = (1., 3., 5.)
    pos = tm.roessler(x0, t, params)
    i_eq = 50000
    sample_every = 10
    t = t[i_eq::sample_every]
    # set the X component as our measured signal
    x = pos[i_eq::sample_every, 0]
    # get mi
    maxlag = np.where(t <= (t[0] + 10.))[0][-1].astype("int")
    mi, mi_lags = rc.mi(x, maxlag)
    mi_filt, _ = utils.boxfilter(mi, filter_width=25, estimate="mean")
    tau_mi = rc.first_minimum(mi_filt)
    # FNN
    M = 10
    R = 0.025
    fnn_mi, dims_mi = rc.fnn(x, tau_mi, maxdim=M, r=R)
    m_mi = dims_mi[rc.first_zero(fnn_mi)]
    R = rc.rp(x, m=m_mi, tau=tau_mi, e=0.50,
              norm="euclidean", threshold_by="distance")
    ax3.imshow(R, cmap=pl.cm.gray_r,
               interpolation="none", origin="lower",
               rasterized=True)
    del R
    ax3.set_title(r"$\tau_e = %d, m_e = %d, \varepsilon = 0.50$" % (tau_mi, m_mi),
                  fontsize=axlabfs)

    # D. Geometric Brownian Motion
    # https://jtsulliv.github.io/stock-movement/
    print("D. Geometric Brownian Motion ...")
    N = 500
    mu, sd = 0.25, 0.5       # drift coefficient, diffusion coefficient
    x0 = 100.               # initial condition of GBM
    t = np.linspace(0, 1., N + 1)     # time span
    dt = 1. / float(N)       # time increment
    b = np.random.normal(0., 1., N) * np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)                # Wiener process
    x = [x0]
    for i in xrange(1, N + 1):
        drift = (mu + 0.5 * sd ** 2) * t[i]
        diffusion = sd * W[i - 1]
        x_ = x0 * np.exp(drift + diffusion)
        x.append(x_)
    x = np.array(x)
    # get mi
    maxlag = 100
    mi, mi_lags = rc.mi(x, maxlag)
    mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
    tau_mi = rc.first_minimum(mi_filt)
    # FNN
    M = 10
    R = 0.025
    fnn_mi, dims_mi = rc.fnn(x, tau_mi, maxdim=M, r=R)
    m_mi = dims_mi[rc.first_zero(fnn_mi)]
    R = rc.rp(x, m=1, tau=1, e=0.35,
              norm="euclidean", threshold_by="distance")
    ax4.imshow(R, cmap=pl.cm.gray_r,
               interpolation="none", origin="lower",
               rasterized=True)
    del R
    ax4.set_title(r"$\tau_e = %d, m_e = %d, \varepsilon = 0.35$" % (tau_mi, m_mi),
                  fontsize=axlabfs)


    # prettify figure
    labs = ["A", "B", "C", "D"]
    for i, ax in enumerate(fig.axes):
        ax.tick_params(size=4, labelsize=tiklabfs)
        ax.set_xlabel("Time index", fontsize=axlabfs)
        ax.set_ylabel("Time index", fontsize=axlabfs)
        ax.text(-0.25, 0.95,
                labs[i],
                ha="right", va="center",
                fontsize=axlabfs, fontweight="bold", family="sans-serif",
                usetex=False,
                transform=ax.transAxes
                )
        if ax != ax3:
            ax.set_xticks(np.arange(0., 501., 100.))
            ax.set_xticks(np.arange(0., 501., 100.))
            ax.set_xlim(0., 500.)
            ax.set_ylim(0., 500.)
        else:
            ax.set_xticks(np.arange(0., 5001., 1000.))
            ax.set_xticks(np.arange(0., 5001., 1000.))
            ax.set_xlim(0., 5000.)
            ax.set_ylim(0., 5000.)

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

