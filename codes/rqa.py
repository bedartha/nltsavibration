"""Contains methods for Recurrence Quantification Analysis (RQA).

Bedartha Goswami
Created: August 7, 2014
Last modified: Wed Feb 20, 2019  10:03pm
"""

import numpy as np
from progressbar import ProgressBar
from itertools import chain


def det(R, lmin=None, hist=None, verb=True):
    """returns DETERMINISM for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, lmin, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating DET...")
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    try:
        idx = l >= lmin
        num = l[idx] * Pl[idx]
    except IndexError:
        DET = np.nan
    else:
        den = l * Pl
        DET = num.sum() / den.sum()
    return DET


def entr(R, lmin=None, hist=None, verb=True):
    """returns ENTROPY for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, lmin, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating ENTR...")
    pl = nlines.astype('float') / float(len(ll))
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx1 = l >= lmin
    pl = pl[idx1]
    idx = pl > 0.
    ENTR = (-pl[idx] * np.log(pl[idx])).sum()
    return ENTR


def diagonal_lines_hist(R, lmin, verb=True):
    """returns the histogram P(l) of diagonal lines of length l."""
    if verb:
        print("diagonal lines histogram...")
        pbar = ProgressBar(maxval=len(R)-1).start()
    line_lengths = []
    for i in range(1, len(R) - lmin):
        d = np.diag(R, k=i)
        ll = _count_num_lines(d)
        line_lengths.append(ll)
        if verb: pbar.update(i)
    if verb: pbar.finish()
    line_lengths = np.array(list(chain.from_iterable(line_lengths)))
    bins = np.arange(0.5, line_lengths.max() + 0.1, 1.)
    num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths


def lam(R, vmin=None, hist=None, verb=True):
    """returns LAMINARITY for given recurrence matrix R."""
    if not vmin:
        vmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = vertical_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating LAM...")
    Pv = nlines.astype('float')
    v = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = v >= vmin
    num = v[idx] * Pv[idx]
    den = v * Pv
    LAM = num.sum() / den.sum()
    return LAM


def tt(R, vmin=None, hist=None, verb=True):
    """returns TRAPPING TIME for given recurrence matrix R."""
    if not vmin:
        vmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = vertical_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating TT...")
    Pv = nlines.astype('float')
    v = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = v >= vmin
    num = v[idx] * Pv[idx]
    den = Pv[idx]
    LAM = num.sum() / den.sum()
    return LAM


def vertical_lines_hist(R, verb=True):
    """returns the histogram P(v) of vertical lines of length v."""
    if verb:
        print("vertical lines histogram...")
        pbar = ProgressBar(maxval=len(R)-1).start()
    line_lengths = []
    for i in range(0, len(R)):
        ll = _count_num_lines(R[:, i])
        line_lengths.append(ll)
        if verb: pbar.update(i)
    if verb: pbar.finish()
    line_lengths = np.array(list(chain.from_iterable(line_lengths)))
    bins = np.arange(0.5, line_lengths.max() + 0.1, 1.)
    num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths


def _count_num_lines(arr):
    """returns a list of line lengths contained in given array."""
    diff = np.diff(arr)
    diff_neg = np.r_[diff, -1]
    diff_pos = np.r_[1, diff]
    start = np.where(arr * diff_pos)[0]
    end = np.where(arr * diff_neg)[0]
    line_lens = end - start + 1
    return line_lens


def cpr(Rx, Ry):
    """
    Returns the correlation of probabilities of recurrence (CPR).
    """
    assert Rx.shape == Ry.shape, "RPs are of different sizes!"
    N = Rx.shape[0]
    qx, qy = [np.zeros(N) for i in range(2)]
    for tau in range(N):
        qx[tau] = np.diag(Rx, k=tau).mean()
        qy[tau] = np.diag(Ry, k=tau).mean()

    # obtain indices after taking into account decorrelation time
    e = np.exp(1.)
    try:
        ix = np.where(qx < 1. / e)[0][0]
        iy = np.where(qy < 1. / e)[0][0]
        i = max(ix, iy)
    except IndexError:
        i = N

    # final estimate
    if i < N:
        # normalised data series to mean zero and standard deviation one after
        # removing entries before decorrelation time
        qx_ = qx[i:]
        qx_ = (qx_ - np.nanmean(qx_)) / np.nanstd(qx_)
        qy_ = qy[i:]
        qy_ = (qy_ - np.nanmean(qy_)) / np.nanstd(qy_)

        # estimate CPR as the dot product of normalised series
        C = (qx_ * qy_).mean()
    else:
        C = np.nan

    return C


def rmd(Rx, Ry):
    """Returns Recurrence-based Measure of Dependence (RMD)"""
    assert Rx.shape == Ry.shape, "RPs are of different sizes!"
    px_i = Rx.mean(axis=1)
    py_i = Ry.mean(axis=1)
    Rxy = Rx * Ry
    pxy_i = Rxy.mean(axis=1)

    k1 = (px_i * py_i) != 0.
    k2 = pxy_i !=0.
    k = k1 + k2
    R = (pxy_i[k] * np.log(pxy_i[k] / (px_i * py_i)[k])).sum()

    return R
