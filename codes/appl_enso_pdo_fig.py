#! /usr/bin/env python3
"""
Plots consequences of synchronization
=====================================

"""

# Created: Tue Jan 22, 2019  12:32pm
# Last modified: Mon Jan 11, 2021  09:03pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import argparse
import datetime as dt
import numpy as np
import matplotlib.pyplot as pl
from itertools import chain
from scipy.stats import percentileofscore as pos
import matplotlib.dates as mdates
from matplotlib.patches import Polygon

import recurrence as rc
import rqa
import igraph as ig
import utils
from utils import _progressbar_start, _progressbar_update, _progressbar_finish

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]


def _load_indices():
    """Loads the ENSO and PDO index data"""
    # load data
    # enso
    enso = np.loadtxt(DATPATH + "nino34_long.csv",
                      delimiter=",", skiprows=2)
    t_enso, x_enso = enso[:, 1], enso[:, 0]
    t_enso = np.array([dt.datetime.fromordinal(int(date)) for date in t_enso])
    # pdo
    pdo = np.loadtxt(DATPATH + "pdo.csv",
                      delimiter=",", skiprows=2)
    t_pdo, x_pdo = pdo[:, 1], pdo[:, 0]
    t_pdo = np.array([dt.datetime.fromordinal(int(date)) for date in t_pdo])

    # get common time scale
    t_enso_set = set(t_enso.tolist())
    t_pdo_set = set(t_pdo.tolist())
    t_common = t_enso_set.intersection(t_pdo_set)
    t_common = np.array(sorted(list(t_common)))
    i_enso = np.in1d(t_enso, t_common)
    x_enso = x_enso[i_enso]
    i_pdo = np.in1d(t_pdo, t_common)
    x_pdo = x_pdo[i_pdo]
    t = t_common

    return t, x_enso, x_pdo


def _get_embedding():
    """Estimates and saves the embedding parameters for all windows"""
    t, x_enso, x_pdo = _load_indices()
    WS, SS = args.window_size, args.step_size
    if args.embed_solo:
        tm, xw_enso, m_enso, tau_enso = _embed_solo(t, x_enso, WS, SS)
        tm, xw_pdo, m_pdo, tau_pdo = _embed_solo(t, x_pdo, WS, SS)
        m = {"enso": m_enso,
             "pdo": m_pdo
             }
        tau = {"enso": tau_enso,
               "pdo": tau_pdo
               }
        xw = {"enso": xw_enso,
              "pdo": xw_pdo
              }
        FN = DATPATH + "embed_solo_WS%d_SS%d" % (WS, SS)
        #save output
        np.savez(FN, tm=tm, xw=xw, m=m, tau=tau)
    elif args.embed_pair:
        tm, xw, yw, m, tau = _embed_pair(t, x_enso, x_pdo, WS, SS)
        FN = DATPATH + "embed_pair_WS%d_SS%d" % (WS, SS)
        #save output
        np.savez(FN, tm=tm, xw=xw, yw=yw, m=m, tau=tau)
    else:
        print("Please provide --embed-solo or --embed-pair")
        sys.exit()
    if args.verbose: print("output saved to: %s.npz" % FN)
    return None


def _embed_solo(t, x, ws, ss):
    """Embed the time series for each window"""
    n = len(t)
    nw = int(np.floor(float(n - ws) / float(ss)))
    tm = np.empty(nw, dtype="object")
    m, tau = [np.zeros(nw, dtype="int") for i in range(2)]
    xw = np.zeros((nw, ws), dtype="float")
    maxlag = 150
    maxdim = 10
    R = 0.025
    pb = _progressbar_start(max_value=nw, pbar_on=args.verbose)
    for i in range(nw):
        start = i * ss
        end = start + ws
        x_ = x[start:end]
        xw[i] = x_
        # get mi
        mi, mi_lags = rc.mi(x_, maxlag, pbar_on=False)
        mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
        try:
            tau[i] = rc.first_minimum(mi_filt)
        except ValueError:
            tau[i] = 1
        # FNN
        fnn, dims = rc.fnn(x_, tau[i], maxdim=maxdim, r=R, pbar_on=False)
        m[i] = dims[rc.first_zero(fnn)]
        tm[i] = t[start] + (t[end] - t[start]) / 2
        _progressbar_update(pb, i)
    _progressbar_finish(pb)
    return tm, xw, m, tau


def _embed_pair(t, x, y, ws, ss):
    """Determines common embedding parameters for both time series"""
    n = len(t)
    nw = int(np.floor(float(n - ws) / float(ss)))
    tm = np.empty(nw, dtype="object")
    m, tau = [np.zeros(nw, dtype="int") for i in range(2)]
    xw, yw = [np.zeros((nw, ws), dtype="float") for i in range(2)]
    maxlag = 150
    maxdim = 10
    R = 0.025
    pb = _progressbar_start(max_value=nw, pbar_on=args.verbose)
    for i in range(nw):
        start = i * ss
        end = start + ws
        x_ = x[start:end]
        y_ = y[start:end]
        xw[i] = x_
        yw[i] = y_
        # get mi
        mi1, mi_lags1 = rc.mi(x_, maxlag, pbar_on=False)
        mi_filt1, _ = utils.boxfilter(mi1, filter_width=3, estimate="mean")
        tau1 = rc.first_minimum(mi_filt1)
        mi2, mi_lags2 = rc.mi(y_, maxlag, pbar_on=False)
        mi_filt2, _ = utils.boxfilter(mi2, filter_width=3, estimate="mean")
        tau2 = rc.first_minimum(mi_filt2)
        tau[i] = int(max(tau1, tau2))
        # FNN
        fnn1, dims1 = rc.fnn(x_, tau[i], maxdim=maxdim, r=R, pbar_on=False)
        m1 = dims1[rc.first_zero(fnn1)]
        fnn2, dims2 = rc.fnn(y_, tau[i], maxdim=maxdim, r=R, pbar_on=False)
        m2 = dims2[rc.first_zero(fnn2)]
        m[i] = int(max(m1, m2))
        tm[i] = t[start] + (t[end] - t[start]) / 2
        _progressbar_update(pb, i)
    _progressbar_finish(pb)
    return tm, xw, yw, m, tau


def _get_det():
    """
    Estimates the determinism DET for the indices.
    """
    # load data
    utils._printmsg("load data ...", args.verbose)
    t, x_enso, x_pdo = _load_indices()
    x = {"enso": x_enso,
         "pdo": x_pdo,
         }
    names = ["enso", "pdo"]

    # get surrogates
    utils._printmsg("iAAFT surrogates ...", args.verbose)
    ns = args.nsurr
    SURR = {}
    for name in names:
        utils._printmsg("\t for %s" % name.upper(), args.verbose)
        SURR[name] = rc.surrogates(x[name], ns, "iaaft", verbose=args.verbose)

    # recurrence plot parameters
    EPS, LMIN = 0.30, 3
    thrby = "frr"

    # get DET for original data
    utils._printmsg("DET for original data ...", args.verbose)
    n = len(t)
    ws, ss = args.window_size, args.step_size
    nw = int(np.floor(float(n - ws) / float(ss)))
    tm = np.empty(nw, dtype="object")
    m, tau = {}, {}
    R = {}
    maxlag = 150
    maxdim = 20
    r_fnn = 0.0010
    DET = {}
    for name in names:
        if args.verbose: print("\t for %s" % name.upper())
        # get embedding parameters
        ## get mi
        mi, mi_lags = rc.mi(x[name], maxlag, pbar_on=False)
        # mi, mi_lags = rc.acf(x[name], maxlag)
        mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
        try:
            tau[name] = rc.first_minimum(mi_filt)
        except ValueError:
            tau[name] = 1
        ## FNN
        fnn, dims = rc.fnn(x[name], tau[name],
                           maxdim=maxdim, r=r_fnn,
                           pbar_on=False)
        m[name] = dims[rc.first_zero(fnn)]
        R[name] = rc.rp(x[name], m=m[name], tau=tau[name], e=EPS,
                     norm="euclidean", threshold_by=thrby,
                     )
        R_ = R[name]
        nw = len(tm)
        det = np.zeros(nw)
        pb = _progressbar_start(max_value=nw, pbar_on=args.verbose)
        for i in range(nw):
            start = i * ss
            end = start + ws
            Rw = R_[start:end, start:end]
            det[i] = rqa.det(Rw, lmin=LMIN, hist=None, verb=False)
            tm[i] = t[start] + (t[end] - t[start]) / 2
            _progressbar_update(pb, i)
        _progressbar_finish(pb)
        DET[name] = det

    # get DET for surrogate data
    utils._printmsg("DET for surrogates ...", args.verbose)
    DETSURR = {}
    for name in names:
        utils._printmsg("\tfor %s" % name.upper(), args.verbose)
        xs = SURR[name]
        y = np.diff(xs, axis=0)
        detsurr = np.zeros((ns, nw), dtype="float")
        pb = _progressbar_start(max_value=ns, pbar_on=args.verbose)
        for k in range(ns):
            Rs = rc.rp(xs[k], m=m[name], tau=tau[name], e=EPS,
                       norm="euclidean", threshold_by=thrby,
                       )
            for i in range(nw):
                start = i * ss
                end = start + ws
                Rw = Rs[start:end, start:end]
                detsurr[k, i] = rqa.det(Rw, lmin=LMIN, hist=None, verb=False)
            _progressbar_update(pb, k)
        _progressbar_finish(pb)
        DETSURR[name] = detsurr

    # get each individual array out of dict to avoid  NumPy import error
    DET_enso = DET["enso"]
    DET_pdo = DET["pdo"]
    DETSURR_enso = DETSURR["enso"]
    DETSURR_pdo = DETSURR["pdo"]
    SURR_enso = SURR["enso"]
    SURR_pdo = SURR["pdo"]
    tm = np.array([date.toordinal() for date in tm])

    # save output
    EPS = int(EPS * 100)
    FN = DATPATH + "det_WS%d_SS%d_EPS%dpc_LMIN%d_NSURR%d" \
                   % (ws, ss, EPS, LMIN, ns)
    np.savez(FN,
             DET_enso=DET_enso, DET_pdo=DET_pdo,
             DETSURR_enso=DETSURR_enso, DETSURR_pdo=DETSURR_pdo,
             SURR_enso=SURR_enso, SURR_pdo=SURR_pdo,
             tm=tm
            )
    if args.verbose: print("output saved to: %s.npz" % FN)

    return None


def _get_spl():
    """
    Estimates the average shortest path length SPL for the indices.
    """
    # load data
    utils._printmsg("load data ...", args.verbose)
    t, x_enso, x_pdo = _load_indices()
    x = {"enso": x_enso,
         "pdo": x_pdo,
         }
    names = ["enso", "pdo"]

    # get surrogates
    utils._printmsg("iAAFT surrogates ...", args.verbose)
    ns = args.nsurr
    SURR = {}
    for name in names:
        utils._printmsg("\t for %s" % name.upper(), args.verbose)
        SURR[name] = rc.surrogates(x[name], ns, "iaaft", verbose=args.verbose)

    # recurrence plot parameters
    EPS, LMIN = 0.30, 3
    thrby = "frr"

    # get SPL for original data
    utils._printmsg("SPL for original data ...", args.verbose)
    n = len(t)
    ws, ss = args.window_size, args.step_size
    nw = int(np.floor(float(n - ws) / float(ss)))
    tm = np.empty(nw, dtype="object")
    m, tau = {}, {}
    A = {}
    maxlag = 150
    maxdim = 20
    r_fnn = 0.0010
    SPL = {}
    for name in names:
        if args.verbose: print("\t for %s" % name.upper())
        # get embedding parameters
        ## get mi
        mi, mi_lags = rc.mi(x[name], maxlag, pbar_on=False)
        # mi, mi_lags = rc.acf(x[name], maxlag)
        mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
        try:
            tau[name] = rc.first_minimum(mi_filt)
        except ValueError:
            tau[name] = 1
        ## FNN
        fnn, dims = rc.fnn(x[name], tau[name],
                           maxdim=maxdim, r=r_fnn,
                           pbar_on=False)
        m[name] = dims[rc.first_zero(fnn)]
        A[name] = rc.rn(x[name], m=m[name], tau=tau[name], e=EPS,
                     norm="euclidean", threshold_by=thrby,
                     )
        A_ = A[name]
        G_ = ig.Graph.Adjacency(A_.tolist(), mode=ig.ADJ_UNDIRECTED)
        nw = len(tm)
        spl = np.zeros(nw)
        pb = _progressbar_start(max_value=nw, pbar_on=args.verbose)
        for i in range(nw):
            start = i * ss
            end = start + ws
            Gw = G_.subgraph(vertices=G_.vs[start:end])
            pl_hist = Gw.path_length_hist(directed=False)
            spl[i] = pl_hist.mean
            tm[i] = t[start] + (t[end] - t[start]) / 2
            _progressbar_update(pb, i)
        _progressbar_finish(pb)
        SPL[name] = spl

    # get SPL for surrogate data
    utils._printmsg("SPL for surrogates ...", args.verbose)
    SPLSURR = {}
    for name in names:
        utils._printmsg("\tfor %s" % name.upper(), args.verbose)
        xs = SURR[name]
        y = np.diff(xs, axis=0)
        splsurr = np.zeros((ns, nw), dtype="float")
        pb = _progressbar_start(max_value=ns, pbar_on=args.verbose)
        for k in range(ns):
            As = rc.rp(xs[k], m=m[name], tau=tau[name], e=EPS,
                       norm="euclidean", threshold_by=thrby,
                       )
            Gs = ig.Graph.Adjacency(As.tolist(), mode=ig.ADJ_UNDIRECTED)
            for i in range(nw):
                start = i * ss
                end = start + ws
                Gw = Gs.subgraph(vertices=Gs.vs[start:end])
                pl_hist = Gw.path_length_hist(directed=False)
                splsurr[k, i] = pl_hist.mean
            _progressbar_update(pb, k)
        _progressbar_finish(pb)
        SPLSURR[name] = splsurr

    # get each individual array out of dict to avoid  NumPy import error
    SPL_enso = SPL["enso"]
    SPL_pdo = SPL["pdo"]
    SPLSURR_enso = SPLSURR["enso"]
    SPLSURR_pdo = SPLSURR["pdo"]
    SURR_enso = SURR["enso"]
    SURR_pdo = SURR["pdo"]
    tm = np.array([date.toordinal() for date in tm])

    # save output
    EPS = int(EPS * 100)
    FN = DATPATH + "spl_WS%d_SS%d_EPS%dpc_LMIN%d_NSURR%d" \
                   % (ws, ss, EPS, LMIN, ns)
    np.savez(FN,
             SPL_enso=SPL_enso, SPL_pdo=SPL_pdo,
             SPLSURR_enso=SPLSURR_enso, SPLSURR_pdo=SPLSURR_pdo,
             SURR_enso=SURR_enso, SURR_pdo=SURR_pdo,
             tm=tm
            )
    if args.verbose: print("output saved to: %s.npz" % FN)

    return None


def _get_cpr():
    """Estimates the CPR between ENSO and PDO"""
    # load data
    utils._printmsg("load data ...", args.verbose)
    t, x_enso, x_pdo = _load_indices()
    x = {"enso": x_enso,
         "pdo": x_pdo,
         }
    names = ["enso", "pdo"]

    # recurrence plot parameters
    EPS = 0.30
    thrby = "frr"

    # embedding parameters
    utils._printmsg("embedding parameters ...", args.verbose)
    n = len(t)
    m, tau = {}, {}
    R = {}
    maxlag = 150
    maxdim = 20
    r_fnn = 0.0010
    for name in names:
        if args.verbose: print("\t for %s" % name.upper())
        # get embedding parameters
        ## get mi
        mi, mi_lags = rc.mi(x[name], maxlag, pbar_on=False)
        # mi, mi_lags = rc.acf(x[name], maxlag)
        mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
        try:
            tau[name] = rc.first_minimum(mi_filt)
        except ValueError:
            tau[name] = 1
        ## FNN
        fnn, dims = rc.fnn(x[name], tau[name],
                           maxdim=maxdim, r=r_fnn,
                           pbar_on=False)
        m[name] = dims[rc.first_zero(fnn)]
    # take the maximum delay and the maximum embedding dimension
    tau = np.max([tau["enso"], tau["pdo"]]).astype("int")
    m = np.max([m["enso"], m["pdo"]]).astype("int")

    # get surrogates
    utils._printmsg("surrogates ...", args.verbose)
    ns = args.nsurr
    SURR = {}
    params = {
              "m": m,
              "tau": tau,
              "eps": EPS,
              "norm": "euclidean",
              "thr_by": thrby,
              "tol": 2.
              }
    ns = args.nsurr
    SURR = {}
    for name in names:
        utils._printmsg("\t for %s" % name.upper(), args.verbose)
        SURR[name] = rc.surrogates(x[name], ns, "twins", params,
                                   verbose=args.verbose)


    # get CPR for original data
    utils._printmsg("CPR for original data ...", args.verbose)
    ws, ss = args.window_size, args.step_size
    nw = int(np.floor(float(n - ws) / float(ss)))
    tm = np.empty(nw, dtype="object")
    for name in names:
        R[name] = rc.rp(x[name], m=m, tau=tau, e=EPS,
                     norm="euclidean", threshold_by=thrby,
                     )
    cpr = np.zeros(nw)
    pb = _progressbar_start(max_value=nw, pbar_on=args.verbose)
    for i in range(nw):
        start = i * ss
        end = start + ws
        Rw_enso = R["enso"][start:end, start:end]
        Rw_pdo = R["pdo"][start:end, start:end]
        cpr[i] = rqa.cpr(Rw_enso, Rw_pdo)
        tm[i] = t[start] + (t[end] - t[start]) / 2
        _progressbar_update(pb, i)
    _progressbar_finish(pb)

    # get CPR for surrogate data
    utils._printmsg("CPR for surrogates ...", args.verbose)
    Rs = {}
    cprsurr = np.zeros((ns, nw), dtype="float")
    pb = _progressbar_start(max_value=ns, pbar_on=args.verbose)
    for k in range(ns):
        for name in names:
            xs = SURR[name][k]
            Rs[name] = rc.rp(xs, m=m, tau=tau, e=EPS,
                             norm="euclidean", threshold_by=thrby,
                             )
        for i in range(nw):
            start = i * ss
            end = start + ws
            Rsw_enso = Rs["enso"][start:end, start:end]
            Rsw_pdo = Rs["pdo"][start:end, start:end]
            cprsurr[k, i] = rqa.cpr(Rsw_enso, Rsw_pdo)
        _progressbar_update(pb, k)
    _progressbar_finish(pb)

    # get each individual array out of dict to avoid  NumPy import error
    SURR_enso = SURR["enso"]
    SURR_pdo = SURR["pdo"]
    tm = np.array([date.toordinal() for date in tm])


    # save output
    EPS = int(EPS * 100)
    FN = DATPATH + "cpr_WS%d_SS%d_EPS%dpc_NSURR%d" \
                   % (ws, ss, EPS, ns)
    np.savez(FN,
             cpr=cpr, tm=tm, cprsurr=cprsurr,
             SURR_enso=SURR_enso, SURR_pdo=SURR_pdo,
             )
    if args.verbose: print("output saved to: %s.npz" % FN)

    return None



def _get_rmd():
    """Estimates the RMD between ENSO and PDO"""
    # load data
    utils._printmsg("load data ...", args.verbose)
    t, x_enso, x_pdo = _load_indices()
    x = {"enso": x_enso,
         "pdo": x_pdo,
         }
    names = ["enso", "pdo"]

    # recurrence plot parameters
    EPS = 0.30
    thrby = "frr"

    # embedding parameters
    utils._printmsg("embedding parameters ...", args.verbose)
    n = len(t)
    m, tau = {}, {}
    R = {}
    maxlag = 150
    maxdim = 20
    r_fnn = 0.0010
    for name in names:
        if args.verbose: print("\t for %s" % name.upper())
        # get embedding parameters
        ## get mi
        mi, mi_lags = rc.mi(x[name], maxlag, pbar_on=False)
        # mi, mi_lags = rc.acf(x[name], maxlag)
        mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
        try:
            tau[name] = rc.first_minimum(mi_filt)
        except ValueError:
            tau[name] = 1
        ## FNN
        fnn, dims = rc.fnn(x[name], tau[name],
                           maxdim=maxdim, r=r_fnn,
                           pbar_on=False)
        m[name] = dims[rc.first_zero(fnn)]
    # take the maximum delay and the maximum embedding dimension
    tau = np.max([tau["enso"], tau["pdo"]]).astype("int")
    m = np.max([m["enso"], m["pdo"]]).astype("int")

    # get surrogates
    utils._printmsg("surrogates ...", args.verbose)
    ns = args.nsurr
    SURR = {}
    params = {
              "m": m,
              "tau": tau,
              "eps": EPS,
              "norm": "euclidean",
              "thr_by": thrby,
              "tol": 2.
              }
    for name in names:
        utils._printmsg("\t for %s" % name.upper(), args.verbose)
        # SURR[name] = rc.surrogates(x[name], ns, "iaaft", verbose=args.verbose)
        SURR[name] = rc.surrogates(x[name], ns, "twins", params,
                                   verbose=args.verbose)

    # get RMD for original data
    utils._printmsg("RMD for original data ...", args.verbose)
    ws, ss = args.window_size, args.step_size
    nw = int(np.floor(float(n - ws) / float(ss)))
    tm = np.empty(nw, dtype="object")
    for name in names:
        R[name] = rc.rp(x[name], m=m, tau=tau, e=EPS,
                        norm="euclidean", threshold_by=thrby,
                        )
    rmd = np.zeros(nw)
    pb = _progressbar_start(max_value=nw, pbar_on=args.verbose)
    for i in range(nw):
        start = i * ss
        end = start + ws
        Rw_enso = R["enso"][start:end, start:end]
        Rw_pdo = R["pdo"][start:end, start:end]
        rmd[i] = rqa.rmd(Rw_enso, Rw_pdo)
        tm[i] = t[start] + (t[end] - t[start]) / 2
        _progressbar_update(pb, i)
    _progressbar_finish(pb)

    # get RMD for surrogate data
    utils._printmsg("RMD for surrogates ...", args.verbose)
    Rs = {}
    rmdsurr = np.zeros((ns, nw), dtype="float")
    pb = _progressbar_start(max_value=ns, pbar_on=args.verbose)
    for k in range(ns):
        for name in names:
            xs = SURR[name][k]
            Rs[name] = rc.rp(xs, m=m, tau=tau, e=EPS,
                             norm="euclidean", threshold_by=thrby,
                             )
        for i in range(nw):
            start = i * ss
            end = start + ws
            Rsw_enso = Rs["enso"][start:end, start:end]
            Rsw_pdo = Rs["pdo"][start:end, start:end]
            rmdsurr[k, i] = rqa.rmd(Rsw_enso, Rsw_pdo)
        _progressbar_update(pb, k)
    _progressbar_finish(pb)


    # get each individual array out of dict to avoid  NumPy import error
    SURR_enso = SURR["enso"]
    SURR_pdo = SURR["pdo"]
    tm = np.array([date.toordinal() for date in tm])

    # save output
    EPS = int(EPS * 100)
    FN = DATPATH + "rmd_WS%d_SS%d_EPS%dpc_NSURR%d" \
                   % (ws, ss, EPS, ns)
    np.savez(FN,
             rmd=rmd, tm=tm, rmdsurr=rmdsurr,
             SURR_enso=SURR_enso, SURR_pdo=SURR_pdo,
             )
    if args.verbose: print("output saved to: %s.npz" % FN)

    return None


def _get_communities():
    """
    Identifies the optimal community structure based on modularity.
    """
    # load data
    utils._printmsg("load data ...", args.verbose)
    t, x_enso, x_pdo = _load_indices()
    x = {"enso": x_enso,
         "pdo": x_pdo,
         }
    names = ["enso", "pdo"]

    # recurrence plot parameters
    EPS = 0.30
    thrby = "frr"

    # embedding parameters
    utils._printmsg("embedding parameters ...", args.verbose)
    n = len(t)
    m, tau = {}, {}
    R = {}
    maxlag = 150
    maxdim = 20
    r_fnn = 0.0010
    for name in names:
        if args.verbose: print("\t for %s" % name.upper())
        # get embedding parameters
        ## get mi
        mi, mi_lags = rc.mi(x[name], maxlag, pbar_on=False)
        # mi, mi_lags = rc.acf(x[name], maxlag)
        mi_filt, _ = utils.boxfilter(mi, filter_width=3, estimate="mean")
        try:
            tau[name] = rc.first_minimum(mi_filt)
        except ValueError:
            tau[name] = 1
        ## FNN
        fnn, dims = rc.fnn(x[name], tau[name],
                           maxdim=maxdim, r=r_fnn,
                           pbar_on=False)
        m[name] = dims[rc.first_zero(fnn)]

    # # print out embedding dimensions for documentation in the paper
    # print m
    # print tau
    # sys.exit()

    # identify communities using modularity
    utils._printmsg("communities based on modularity ...", args.verbose)
    COMM = {}
    for name in names:
        utils._printmsg("\tfor %s" % name.upper(), args.verbose)
        A = rc.rn(x[name], m=m[name], tau=tau[name], e=EPS,
                  norm="euclidean", threshold_by="frr", normed=True)

        # optimize modularity
        utils._printmsg("\toptimize modularity ...", args.verbose)
        G = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
        dendro = G.community_fastgreedy()
        # dendro = G.community_edge_betweenness(directed=False)
        clust = dendro.as_clustering()
        # clust = G.community_multilevel()
        mem = clust.membership
        COMM[name] = mem

    # get each individual array out of dict to avoid  NumPy import error
    x_enso = x["enso"]
    x_pdo = x["pdo"]
    COMM_enso = COMM["enso"]
    COMM_pdo = COMM["pdo"]
    t = np.array([date.toordinal() for date in t])

    # save output
    EPS = int(EPS * 100)
    FN = DATPATH + "communities_EPS%d" \
                   % EPS
    np.savez(FN,
             x_enso=x_enso, x_pdo=x_pdo, t=t,
             COMM_enso=COMM_enso, COMM_pdo=COMM_pdo,
             m=m, tau=tau, e=EPS, thrby=thrby
             )
    if args.verbose: print("output saved to: %s.npz" % FN)

    return None


def _holm(pvalues, alpha=0.05, corr_type="dunn"):
    """
    Returns indices of p-values using Holm's method for multiple testing.
    """
    n = len(pvalues)
    sortidx = np.argsort(pvalues)
    p_ = pvalues[sortidx]
    j = np.arange(1, n + 1)
    if corr_type == "bonf":
            corr_factor = alpha / (n - j + 1)
    elif corr_type == "dunn":
        corr_factor = 1. - (1. - alpha) ** (1. / (n - j + 1))
    try:
        idx = np.where(p_ <= corr_factor)[0][-1]
        idx = sortidx[:idx]
    except IndexError:
        idx = []
    return idx


def _get_fig():
    """Plots the final figure"""
    # set up figure
    utils._printmsg("set up figure ...", args.verbose)
    fig = pl.figure(figsize=[7.480315, 7.874016])     # 190 mm wide, 200 mm tall 
    l = 0.10
    b_ = [0.74, 0.58, 0.42, 0.26, 0.10]
    w, h = 0.75, 0.16
    axes = [fig.add_axes([l, b, w, h]) for b in b_]
    axlabfs, tiklabfs = 12, 11
    clr1, clr2, clr3 = "MediumTurquoise", "GoldenRod", "IndianRed"
    mksz = 2

    # parameters
    names = ["enso", "pdo"]
    clrs = [clr1, clr2]
    SIG_ALPHA = 0.50
    ws, ss, EPS = 120, 1, 30
    LMIN, ns = 3, 1000
    every = 1

    # DET
    utils._printmsg("DET ...", args.verbose)
    FN = DATPATH + "det_WS%d_SS%d_EPS%dpc_LMIN%d_NSURR%d.npz" \
                   % (ws, ss, EPS, LMIN, ns)
    DAT = np.load(FN)
    DET = {}
    DET["enso"] = DAT["DET_enso"]
    DET["pdo"] = DAT["DET_pdo"]
    DETSURR = {}
    DETSURR["enso"] = DAT["DETSURR_enso"]
    DETSURR["pdo"] = DAT["DETSURR_pdo"]
    tm = np.array([dt.datetime.fromordinal(int(date)) for date in DAT["tm"]])
    nw = len(tm)
    for i, name in enumerate(names):
        det = DET[name][::every]
        detsurr = DETSURR[name][:, ::every]
        lo = np.percentile(detsurr, 25., axis=0)
        hi = np.percentile(detsurr, 75, axis=0)
        ax = axes[0]
        # ax.fill_between(tm, lo, hi, color=clrs[i], alpha=0.25)
        ax.plot(tm, lo, "--", c=clrs[i])
        ax.plot(tm, hi, "--", c=clrs[i])
        ax.plot(tm, det, ":", c=clrs[i],
                alpha=0.75, lw=1.0)
        k = (det <= lo) + (det >= hi)
        ax.plot(tm[k], det[k], ".", c=clrs[i],
                alpha=0.75, ms=mksz)
        # proxy artist for legend
        ax.fill_between(tm, 1.5, 2., color=clrs[i], label=name.upper())

    # SPL
    utils._printmsg("SPL ...", args.verbose)
    FN = DATPATH + "spl_WS%d_SS%d_EPS%dpc_LMIN%d_NSURR%d.npz" \
                   % (ws, ss, EPS, LMIN, ns)
    DAT = np.load(FN)
    SPL = {}
    SPL["enso"] = DAT["SPL_enso"]
    SPL["pdo"] = DAT["SPL_pdo"]
    SPLSURR = {}
    SPLSURR["enso"] = DAT["SPLSURR_enso"]
    SPLSURR["pdo"] = DAT["SPLSURR_pdo"]
    tm = [dt.datetime.fromordinal(int(date)) for date in DAT["tm"]]
    tm = np.array(tm)
    nw = len(tm)
    for i, name in enumerate(names):
        spl = SPL[name][::every]
        splsurr = SPLSURR[name][:, ::every]
        lo = np.percentile(splsurr, 25., axis=0)
        hi = np.percentile(splsurr, 75, axis=0)
        ax = axes[1]
        # ax.fill_between(tm, lo, hi, color=clrs[i], alpha=0.25)
        ax.plot(tm, lo, "--", c=clrs[i])
        ax.plot(tm, hi, "--", c=clrs[i])
        ax.plot(tm, spl, ":", c=clrs[i],
                alpha=0.75, lw=1.0)
        k = (spl <= lo) + (spl >= hi)
        ax.plot(tm[k], spl[k], ".", c=clrs[i],
                alpha=0.75, ms=mksz)
        # proxy artist for legend
        ax.fill_between(tm, 3.5, 4., color=clrs[i], label=name.upper())

    # CPR
    utils._printmsg("CPR ...", args.verbose)
    FN = DATPATH + "cpr_WS%d_SS%d_EPS%dpc_NSURR%d.npz" \
                   % (ws, ss, EPS, ns)
    DAT = np.load(FN)
    cpr = DAT["cpr"][::every]
    cprsurr = DAT["cprsurr"][:, ::every]
    tm = [dt.datetime.fromordinal(int(date)) for date in DAT["tm"]]
    tm = np.array(tm)
    tm = tm[::every]
    nw = len(tm)
    lo = np.nanpercentile(cprsurr, 0., axis=0)
    hi = np.nanpercentile(cprsurr, 95., axis=0)
    ax = axes[2]
    # ax.fill_between(tm, lo, hi, color=clr1, alpha=0.25)
    ax.plot(tm, lo, "--", c=clr1)
    ax.plot(tm, hi, "--", c=clr1)
    ax.plot(tm, cpr, ":", c=clr1,
            alpha=0.75, lw=1.0)
    nans = np.isnan(cprsurr)
    # k1 = np.less_equal(cpr, lo, where=~np.isnan(cpr))
    # k2 = np.greater_equal(cpr, hi, where=~np.isnan(cpr))
    # k = k1 + k2
    pvals = np.zeros(nw)
    for i in range(nw):
        pvals[i] = 1. - pos(cprsurr[:, i], cpr[i], kind="weak") / 100.
    k = _holm(pvals, alpha=0.05)
    ax.plot(tm[k], cpr[k], ".", c=clr1,
            alpha=0.75, ms=mksz)
    # proxy artist for legend
    ax.fill_between(tm, 1.5, 2., color=clr1, label=r"$CPR$")

    # RMD
    utils._printmsg("RMD ...", args.verbose)
    FN = DATPATH + "rmd_WS%d_SS%d_EPS%dpc_NSURR%d.npz" \
                   % (ws, ss, EPS, ns)
    DAT = np.load(FN)
    rmd = DAT["rmd"][::every]
    rmdsurr = DAT["rmdsurr"][:, ::every]
    tm = [dt.datetime.fromordinal(int(date)) for date in DAT["tm"]]
    tm = np.array(tm)
    tm = tm[::every]
    nw = len(tm)
    lo = np.nanpercentile(rmdsurr, 0., axis=0)
    hi = np.nanpercentile(rmdsurr, 95., axis=0)
    ax_ = axes[2].twinx()
    ax_.set_facecolor("none")
    # ax_.fill_between(tm, lo, hi, color=clr2, alpha=0.25)
    ax_.plot(tm, lo, "--", c=clr2)
    ax_.plot(tm, hi, "--", c=clr2)
    ax_.plot(tm, rmd, ":", c=clr2,
             alpha=0.75, lw=1.0)
    # k = (rmd <= lo) + (rmd >= hi)
    pvals = np.zeros(nw)
    for i in range(nw):
        pvals[i] = 1. - pos(rmdsurr[:, i], cpr[i], kind="weak") / 100.
    k = _holm(pvals, alpha=0.05)
    ax_.plot(tm[k], rmd[k], ".", c=clr2,
             alpha=0.75, ms=mksz)
    # proxy artist for legend
    ax_.fill_between(tm, 150., 200., color=clr2, label=r"$RMD$")

    # communities
    utils._printmsg("communities ...", args.verbose)
    FN = DATPATH + "communities_EPS%d.npz" \
                   % EPS
    DAT = np.load(FN)
    x_enso, x_pdo, t = DAT["x_enso"], DAT["x_pdo"], DAT["t"]
    x = {}
    x["enso"] = x_enso
    x["pdo"] = x_pdo
    t = np.array([dt.datetime.fromordinal(int(date)) for date in DAT["t"]])
    COMM_enso = DAT["COMM_enso"]
    COMM_pdo = DAT["COMM_pdo"]
    COMM = {}
    COMM["enso"] = COMM_enso
    COMM["pdo"] = COMM_pdo
    for ax, name in zip([axes[3], axes[4]], names):
        comm = COMM[name]
        ne = len(comm)
        t_, x_ = t[:ne], x[name][:ne]
        if name == "enso":
            x_ = (x_ - x_.mean()) / x_.std()
        clust_ids = np.unique(comm)
        clrs = [clr1, clr3, clr2, "Fuchsia", "Indigo", "Peru", "Salmon"]
        # if name == "pdo":
        #     clrs[0], clrs[1] = clrs[1], clrs[0]
        ax.plot(t_, x_, "-", c="0.5", alpha=0.5)
        for k, clust_id in enumerate(clust_ids):
            i = comm == clust_id
            ax.plot(t_[i], x_[i], ".",
                    label="Cluster % d", ms=mksz + 2, c=clrs[k])
        if name == "pdo":
            # plot W and C corresponding to cold and warm phases
            coldx = [dt.datetime(1884, 1, 1),
                     dt.datetime(1919, 1, 1),
                     dt.datetime(1960, 1, 1),
                     dt.datetime(2005, 1, 1)
                    ]
            for xcoord in coldx:
                ax.text(xcoord, -4.25,
                        "C",
                        ha="center", va="center",
                        fontsize=tiklabfs,
                        )
            warmx = [dt.datetime(1906, 1, 1),
                     dt.datetime(1936, 1, 1),
                     dt.datetime(1986, 1, 1),
                     dt.datetime(2016, 1, 1),
                    ]
            for xcoord in warmx:
                ax.text(xcoord, -4.25,
                        "W",
                        ha="center", va="center",
                        fontsize=tiklabfs,
                        )

    del ax

    ## prettify
    # Y-axis
    # DET
    ax = axes[0]
    ax.set_ylim(0.25, 0.80)
    ax.set_yticks(np.arange(0.35, 0.76, 0.10))
    ax.legend(loc="upper left")
    ax.set_ylabel(r"$DET$", fontsize=axlabfs)
    ax.set_xlabel("Time (years AD)", fontsize=axlabfs)
    ax.xaxis.set_label_position("top")
    ax.text(1.09, 0.5, r"$DET$",
            ha="center", va="center", rotation=90,
            fontsize=axlabfs, transform=ax.transAxes)
    # SPL
    ax = axes[1]
    ax.set_ylim(1.50, 3.50)
    ax.set_yticks(np.arange(1.80, 3.21, 0.40))
    ax.legend(loc="upper left")
    ax.set_ylabel(r"$SPL$", fontsize=axlabfs)
    ax.text(1.09, 0.5, r"$SPL$",
            ha="center", va="center", rotation=90,
            fontsize=axlabfs, transform=ax.transAxes)
    # CPR
    ax = axes[2]
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax.legend(loc="upper left")
    ax.set_ylabel(r"$CPR$", fontsize=axlabfs)
    # RMD
    ax_.set_ylim(-10, 25)
    ax_.set_yticks(np.arange(-5., 20.01, 5.))
    ax_.legend(loc="upper left", bbox_to_anchor=[0.0, 0.80])
    ax_.set_ylabel(r"$RMD$", fontsize=axlabfs)
    # communities
    # ENSO communities
    ax = axes[3]
    ax.hlines([0.], t[0], t[-1], colors="0.5", linestyles="--")
    ax.set_ylabel(r"Ni{\~n}o 3.4 Index", fontsize=axlabfs)
    ax.text(1.10, 0.5, r"Ni{\~n}o 3.4 Index",
            ha="center", va="center", rotation=90,
            fontsize=axlabfs, transform=ax.transAxes)
    # PDO communities
    ax = axes[4]
    ax.hlines([0.], t[0], t[-1], colors="0.5", linestyles="--")
    ylo, yhi = ax.get_ylim()
    ax.vlines([
                dt.datetime(1895, 1, 1),
                dt.datetime(1915, 1, 1),
                dt.datetime(1924, 1, 1),
                dt.datetime(1947, 1, 1),
                dt.datetime(1976, 1, 1),
                dt.datetime(1998, 1, 1),
                dt.datetime(2013, 1, 1),
               ],
               ylo, yhi,
               colors="0.5", linestyles="--")
    ax.set_ylabel(r"PDO Index", fontsize=axlabfs)
    ax.text(1.10, 0.5, r"PDO Index",
            ha="center", va="center", rotation=90,
            fontsize=axlabfs, transform=ax.transAxes)
    ax.set_xlabel("Time (years AD)", fontsize=axlabfs)
    # ticks
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.tick_params(labelsize=tiklabfs, size=6,
                       top=True, bottom=True,
                       right=True, labelright=True)
        ax.tick_params(which="minor", size=3., top=True, bottom=True)
        if ax is not axes[4]:
            ax.tick_params(labelbottom=False)
        if ax is axes[0]:
            ax.tick_params(top=True, labeltop=True)
        ax.set_xlim(dt.datetime(1850, 1, 1), dt.datetime(2019,1,1))
        XMajLoc = mdates.YearLocator(base=10, month=1, day=1)
        ax.xaxis.set_major_locator(XMajLoc)
        XMinLoc = mdates.YearLocator(base=2, month=1, day=1)
        ax.xaxis.set_minor_locator(XMinLoc)
        XMajFmt = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_formatter(XMajFmt)
        if ax is axes[2]:
            ax.tick_params(right=False, labelright=False)
    for ax, lab in zip(axes, ["a", "b", "c", "d", "e"]):
        ax.text(-0.125, 1.00,
                lab,
                ha="left", va="top",
                fontsize=axlabfs, fontweight="bold", family="sans-serif",
                usetex=False,
                transform=ax.transAxes
                )
    # hatching in areas with no data
    for ax in axes[:3]:
        ylo, yhi = ax.get_ylim()
        poly1 = Polygon(xy=[
                            [dt.datetime(1850, 1, 1).toordinal(), ylo],
                            [dt.datetime(1876, 1, 1).toordinal(), ylo],
                            [dt.datetime(1876, 1, 1).toordinal(), yhi],
                            [dt.datetime(1850, 1, 1).toordinal(), yhi],
                            ],
                        closed=True, hatch="xxx", fill=False,
                        color="LightGray", lw=0.5,
                        )
        hatch1 = ax.add_patch(poly1)
        poly2 = Polygon(xy=[
                            [dt.datetime(2014, 1, 1).toordinal(), ylo],
                            [dt.datetime(2019, 1, 1).toordinal(), ylo],
                            [dt.datetime(2019, 1, 1).toordinal(), yhi],
                            [dt.datetime(2014, 1, 1).toordinal(), yhi],
                            ],
                        closed=True, hatch="xxx", fill=False,
                        color="LightGray", lw=0.5,
                        )
        hatch2 = ax.add_patch(poly2)
    for ax in axes[3:]:
        ylo, yhi = ax.get_ylim()
        poly1 = Polygon(xy=[
                            [dt.datetime(1850, 1, 1).toordinal(), ylo],
                            [dt.datetime(1870, 1, 1).toordinal(), ylo],
                            [dt.datetime(1870, 1, 1).toordinal(), yhi],
                            [dt.datetime(1850, 1, 1).toordinal(), yhi],
                            ],
                        closed=True, hatch="xxx", fill=False,
                        color="LightGray", lw=0.5,
                        )
        hatch1 = ax.add_patch(poly1)

    # save figure
    FT = "." + args.figtype
    FN = "../plots/" + __file__[2:-3] +  FT
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

    return None


def _parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--get",
                        type=str, choices=[
                                           "embedding",
                                           "det",
                                           "spl",
                                           "cpr",
                                           "rmd",
                                           "communities",
                                           "fig",
                                           ],
                        help="Specify what to get")
    parser.add_argument("-ws", "--window-size",
                        type=int, default=360,
                        help="Window size (months)")
    parser.add_argument("-ss", "--step-size",
                        type=int, default=120,
                        help="Window size (months)")
    parser.add_argument("-ns", "--nsurr",
                        type=int, default=100,
                        help="Number of surrogates in each window")
    parser.add_argument("--embed-solo",
                        type=bool, default=False,
                        help="Whether to embed each time series individually")
    parser.add_argument("--embed-pair",
                        type=bool, default=False,
                        help="Whether to embed both time series as pairs")
    parser.add_argument("-v", "--verbose",
                        type=bool, default=False,
                        help="Print verbose messages and progress bars")
    parser.add_argument("-ft", "--figtype",
                        type=str, default="png",
                        help="File type of output figure")
    return parser.parse_args()


if __name__ == "__main__":
    DATPATH = "../data/appl/"
    args = _parse_args()
    _func = eval("_get_%s" % args.get)
    _func()
