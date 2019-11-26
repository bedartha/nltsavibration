"""
Module for recurrence-based nonlinear time series analysis
==========================================================

"""

# Created: Fri Aug 31, 2018  12:26am
# Last modified: Tue Sep 24, 2019  02:59pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np

from scipy.spatial.distance import pdist, squareform

from utils import _progressbar_start, _progressbar_update, _progressbar_finish
from utils import _printmsg

# disable dive by zero warnings
np.seterr(divide="ignore")


def mi(x, maxlag, binrule="fd", pbar_on=True):
    """
    Returns the self mutual information of a time series up to max. lag.
    """
    # initialize variables
    n = len(x)
    lags = np.arange(0, maxlag, dtype="int")
    mi = np.zeros(len(lags))
    # loop over lags and get MI
    pb = _progressbar_start(max_value=maxlag, pbar_on=pbar_on)
    for i, lag in enumerate(lags):
        # extract lagged data
        y1 = x[:n - lag].copy()
        y2 = x[lag:].copy()
        # use np.histogram to get individual entropies
        H1, be1 = entropy1d(y1, binrule)
        H2, be2 = entropy1d(y2, binrule)
        H12, _, _ = entropy2d(y1, y2, [be1, be2])
        # use the entropies to estimate MI
        mi[i] = H1 + H2 - H12
        _progressbar_update(pb, i)
    _progressbar_finish(pb)

    return mi, lags


def entropy1d(x, binrule):
    """
    Returns the Shannon entropy according to the bin rule specified.
    """
    p, be = np.histogram(x, bins=binrule, density=True)
    r = be[1:] - be[:-1]
    P = p * r
    H = -(P * np.log2(P)).sum()

    return H, be


def entropy2d(x, y, bin_edges):
    """
    Returns the Shannon entropy according to the bin rule specified.
    """
    p, bex, bey = np.histogram2d(x, y, bins=bin_edges, normed=True)
    r = np.outer(bex[1:] - bex[:-1], bey[1:] - bey[:-1])
    P = p * r
    H = np.zeros(P.shape)
    i = ~np.isinf(np.log2(P))
    H[i] = -(P[i] * np.log2(P[i]))
    H = H.sum()

    return H, bex, bey


def first_minimum(y):
    """
    Returns the first minimum of given data series y.
    """
    try:
        fmin = np.where(np.diff(np.sign(np.diff(y))) == 2.)[0][0] + 2
    except IndexError:
        fmin = np.nan
    return fmin


def acf(x, maxlag):
    """
    Returns the acuto-correlation function up to maximum lag.
    """
    # normalize data
    n = len(x)
    a = (x - x.mean()) / (x.std() * n)
    b = (x - x.mean()) / x.std()

    # get acf
    cor = np.correlate(a, b, mode="full")
    acf = cor[n:n+maxlag+1]
    lags = np.arange(maxlag + 1)

    return acf, lags


def fnn(x, tau, maxdim, r=0.10, pbar_on=True):
    """
    Returns the number of false nearest neighbours up to max dimension.
    """
    # initialize params
    sd = x.std()
    r = r * (x.max() - x.min())
    e = sd / r
    fnn = np.zeros(maxdim)
    dims = np.arange(1, maxdim + 1, dtype="int")

    # ensure that (m-1) tau is not greater than N = length(x)
    N = len(x)
    K = (maxdim + 1 - 1) * tau
    if K >= N:
        m_c = N / tau
        i = np.where(dims >= m_c)
        fnn[i] = np.nan
        j = np.where(dims < m_c)
        dims = dims[j]

    # get first values of distances for m = 1
    d_m, k_m = mindist(x, 1, tau)

    # loop over dimensions and get FNN values
    pb = _progressbar_start(max_value=maxdim + 1, pbar_on=pbar_on)
    for m in dims:
        # get minimum distances for one dimension higher
        d_m1, k_m1 = mindist(x, m + 1, tau)
        # remove those indices in the m-dimensional calculations which cannot
        # occur in the m+1-dimensional arrays as the m+1-dimensional arrays are
        # smaller
        cond1 = k_m[1] > k_m1[0][-1]
        cond2 = k_m[0] > k_m1[0][-1]
        j = np.where(~(cond1 + cond2))[0]
        k_m_ = (k_m[0][j], k_m[1][j])
        d_k_m, d_k_m1 = d_m[k_m_], d_m1[k_m_]
        n_m1 = d_k_m.shape[0]
        # calculate quantities in Eq. 3.8 of Kantz, Schreiber (2004) 2nd Ed.
        j = d_k_m > 0.
        y = np.zeros(n_m1, dtype="float")
        y[j] = (d_k_m1[j] / d_k_m[j] > e)   # should be r instead of e = sd / r
        w = (e > d_k_m)
        num = float((y * w).sum())
        den = float(w.sum())
        # assign FNN value depending on whether denominator is zero
        if den != 0.:
            fnn[m - 1] = num / den
        else:
            fnn[m - 1] = np.nan
        # assign higher dimensional values to current one before next iteration
        d_m, k_m = d_m1, k_m1
        _progressbar_update(pb, m)
    _progressbar_finish(pb)

    return fnn, dims


def mindist(x, m, tau):
    """
    Returns the minimum distances for each point in given embedding.
    """
    z = embed(x, m, tau)
    # d = squareform(pdist(z))
    n = len(z)
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.max(np.abs(z[i] - z), axis=1)

    np.fill_diagonal(d, 99999999.)
    k = (np.arange(len(d)), np.argmin(d, axis=1))

    return d, k


def embed(x, m, tau):
    """
    Embeds a scalar time series in m dimensions with time delay tau.
    """
    n = len(x)
    k = n - (m - 1) * tau
    z = np.zeros((k, m), dtype="float")
    for i in range(k):
        z[i] = [x[i + j * tau] for j in range(m)]

    return z


def first_zero(y):
    """
    Returns the index of the first value which is zero in y.
    """
    try:
        fzero = np.where(y == 0.)[0][0]
    except IndexError:
        fzero = 0
    return fzero


def rp(x, m, tau, e, norm="euclidean", threshold_by="distance", normed=True):
    """Returns the recurrence plot of given time series."""
    if normed:
        x = normalize(x)
    z = embed(x, m, tau)
    D = squareform(pdist(z, metric=norm))
    R = np.zeros(D.shape, dtype="int")
    if threshold_by == "distance":
        i = np.where(D <= e)
        R[i] = 1
    elif threshold_by == "fan":
        nk = np.ceil(e * R.shape[0]).astype("int")
        i = (np.arange(R.shape[0]), np.argsort(D, axis=0)[:nk])
        R[i] = 1
    elif threshold_by == "frr":
        e = np.percentile(D, e * 100.)
        i = np.where(D <= e)
        R[i] = 1

    return R


def rn(x, m, tau, e, norm="euclidean", threshold_by="distance", normed=True):
    """Returns the recurrence network adjacency matrix of given time series."""
    z = embed(x, m, tau)
    D = squareform(pdist(z, metric=norm))
    np.fill_diagonal(D, np.inf)
    A = np.zeros(D.shape, dtype="int")
    if threshold_by == "distance":
        i = np.where(D <= e)
        A[i] = 1
    elif threshold_by == "fan":
        nk = np.ceil(e * A.shape[0]).astype("int")
        i = (np.arange(A.shape[0]), np.argsort(D, axis=0)[:nk])
        A[i] = 1
    elif threshold_by == "frr":
        e = np.percentile(D, e * 100.)
        i = np.where(D <= e)
        A[i] = 1

    return A


def normalize(x):
    """
    Returns the Z-score series for x.
    """
    return (x - x.mean()) / x.std()


def surrogates(x, ns, method, params=None, verbose=False):
    """
    Returns m random surrogates using the specified method.
    """
    nx = len(x)
    xs = np.zeros((ns, nx))
    if method == "iaaft":               # iAAFT
        # as per the steps given in Lancaster et al., Phys. Rep (2018)
        fft, ifft = np.fft.fft, np.fft.ifft
        TOL = 1E-6
        MSE_0 = 100
        MSE_K = 1000
        MAX_ITER = 10000
        ii = np.arange(nx)
        x_amp = np.abs(fft(x))
        x_srt = np.sort(x)

        pb = _progressbar_start(max_value=ns, pbar_on=verbose)
        for k in range(ns):
            # 1) Generate random shuffle of the data
            count = 0
            ri = np.random.permutation(ii)
            r_prev = x[ri]
            MSE_prev = MSE_0
            # while not np.all(rank_prev == rank_curr) and (count < MAX_ITER):
            while (np.abs(MSE_K - MSE_prev) > TOL) * (count < MAX_ITER):
                MSE_prev = MSE_K
                # 2) FFT current iteration yk, and then invert it but while
                # replacing the amplitudes with the original amplitudes but
                # keeping the angles from the FFT-ed version of the random
                phi_r_prev = np.angle(fft(r_prev))
                r = ifft(x_amp * np.exp(phi_r_prev * 1j), nx)
                # 3) rescale zk to the original distribution of x
                # rank_prev = rank_curr
                ind = np.argsort(r)
                r[ind] = x_srt
                MSE_K = (np.abs(x_amp - np.abs(fft(r)))).mean()
                r_prev = r
                # repeat until rank(z_k+1) = rank(z_k)
                count += 1
            if count >= MAX_ITER:
                print("maximum number of iterations reached!")
            xs[k] = np.real(r)
            _progressbar_update(pb, k)
        _progressbar_finish(pb)
    elif method == "twins":              # twin surrogates
        # 1. Estimate RP according to given parameters
        R = rp(x, m=params["m"], tau=params["tau"], e=params["eps"],
               norm=params["norm"], threshold_by=params["thr_by"])
        # import matplotlib.pyplot as pl
        # pl.imshow(R, origin="lower", cmap=pl.cm.gray_r, interpolation="none")
        # pl.show()

        # 2. Get embedded vectors
        xe = embed(x, params["m"], params["tau"])
        ne = len(xe)
        assert ne == len(R), "Something is wrong!"

        # 2. Identify twins
        _printmsg("identify twins ...", verbose)
        is_twin = []
        twins = []
        TOL = np.floor((params["tol"] * float(nx)) / 100.).astype("int")
        pb = _progressbar_start(max_value=ne, pbar_on=verbose)
        R_ = R.T
        for i in range(ne):
            diff = R_ ==  R_[i]
            j = np.sum(diff, axis=1) >= (ne - TOL)
            j = np.where(j)[0].tolist()
            j.remove(i)
            if len(j) > 0:
                is_twin.append(i)
                twins.append(j)
            _progressbar_update(pb, i)
        _progressbar_finish(pb)

        # 3. Generate surrogates
        _printmsg("generate surrogates ...", verbose)
        all_idx = range(ne)
        start_idx = np.random.choice(np.arange(ne), size=ns)
        xs[:, 0] = xe[start_idx, 0]
        pb = _progressbar_start(max_value=ns, pbar_on=verbose)
        for i in range(ns):
            j = 1
            k = start_idx[i]
            while j < nx:
                if k not in is_twin:
                    k += 1
                else:
                    twins_k = twins[is_twin.index(k)]
                    others = list(set(all_idx).difference(set(twins_k)))
                    l = np.random.choice(others)
                    k = np.random.choice(np.r_[l, twins_k])
                if k >= ne:
                    k = np.random.choice(np.arange(ne), size=1)
                xs[i, j] = xe[k, 0]
                j += 1
            _progressbar_update(pb, i)
        _progressbar_finish(pb)

    elif method == "shuffle":               # simple random shuffling
        k = np.arange(nx)
        for i in range(ns):
            j = np.random.permutation(k)
            xs[i] = x[j]


    return xs


