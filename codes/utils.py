"""
A suite of utility functions that assist in carrying out the analysis
=====================================================================

!! Please import as a module and use it as such -- not as a script !!

"""
# Created: Thu May 26, 2016  03:50PM
# Last modified: Wed Jan 02, 2019  03:12pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>

from progressbar import ProgressBar, Bar, Percentage, ETA
import matplotlib.pyplot as pl
import numpy as np
import datetime as dt
import os
import zipfile
from subprocess import call
from scipy.interpolate import interp1d
from scipy.special import ndtri
from scipy.stats import norm, percentileofscore


def boxfilter(q, filter_width=365, estimate="median"):
    """
    Runs a box filter on the time series to get intra-seasonal averages.

    """
    n = len(q)
    filtered = np.zeros(n)
    error = np.zeros((2, n))
    k = filter_width / 2
    filtered[0] = q[0]
    filtered[-1] = q[-1]
    for i in range(1, n):
        if i < k:
            arr = q[:i]
        elif (i >= k) and (i < (n - k)):
            arr = q[i-k:i+k+1]
        elif i >= (n - k):
            arr = q[i:]
        j = np.where(~np.isnan(arr))[0]
        if len(j) < k:
            filtered[i] = np.nan
            error[0, i] = np.nan
            error[1, i] = np.nan
        else:
            if estimate == "mean":
                mu = arr[j].mean()
                sd = arr[j].std()
                filtered[i] = mu
                error[0, i] = mu - sd / 2.
                error[1, i] = mu + sd + 2.
            elif estimate == "median":
                filtered[i] = np.median(arr[j])
                error[0, i] = np.percentile(arr[j], 25.)
                error[1, i] = np.percentile(arr[j], 75.)
    return filtered, error


def _histogram(arr, logspaced=True):
    """
    Returns the histogram counts acc. to Sturges rule for linear/log bins.
    """
    arr = _remove_nans(arr)
    be = _bin_edges(arr, logspaced=logspaced)
    prob, be_ = np.histogram(arr, bins=be, density=True)
    # assert np.all(be == be_), "Bins returned aren't same as those given!"
    if not np.all(be == be_):
        print("Bins returned aren't same as those given!")
    midpts = 0.5 * (be[:-1] + be[1:])
    return prob, midpts, be


def _remove_nans(arr):
    """
    Removes NaNs from given array.
    """
    return arr[~np.isnan(arr)]


def _bin_edges(arr, logspaced=False):
    """
    Returns bin edges for given array acc. to Sturges Formula.
    """
    n = len(arr)
    nbins = np.ceil(np.log2(n)).astype("int") + 1
    if not logspaced:
        be = np.linspace(arr.min(), arr.max(), nbins)
    else:
        be = np.logspace(np.log10(arr.min()), np.log10(arr.max()), nbins + 1)
    return be


def _convert_to_datetime(arr_list):
    """
    Converts array of integers to Python DateTime objects based on start date.

    Start date for entire analysis: 1 January 1900
    """
    d0 = dt.datetime(1900, 1, 1)
    n = len(arr_list)
    out = []
    for i in range(n):
        ti = arr_list[i]
        no = len(ti)
        tmp = []
        for j in range(no):  # observations in a station loop
            dj = d0 + dt.timedelta(ti[j])
            tmp.append(dj)
        out.append(np.array(tmp))
    return out


def _read_dbf_in_zip(fname, tmpdir="./", clean_up=True):
    """
    Returns NumPy array for the DBF file in the given ZIP archive.
    """
    ## a. unzip the relevant ZIP file
    zf = zipfile.ZipFile(fname)
    fn = [i for i in zf.namelist() if i[-3:] == "dbf"][0]
    fo = tmpdir
    zf.extract(fn, path=fo)
    ## b. use "in2csv" to read the DBF contents to STDOUT and pipe it to file
    f2 = fo + fn[:-3] + "csv"
    F2 = open(f2, "w")
    retcode = call(["in2csv", "%s" % (fo + fn)], stdout=F2)
    F2.close()
    ## c. read in the generated CSV file using NumPy
    D2 = np.genfromtxt(f2, delimiter=",", dtype="O", names=True)
    ## clean up
    if clean_up:
        os.remove(fo + fn)
        os.remove(f2)
    return D2


def _savedat(path, filename, verbose=False, save=False, **kwargs):
    """
    Saves the data in kwargs in filename under given path.
    """
    if save:
        _printmsg("Saving output...", verbose)
        f = path + filename
        locals().update(kwargs)
        var_list = ["%s=%s, " % (key, key) for key in kwargs]
        var_list = "".join(var_list)
        save_cmd = "np.savez('%s', %s)" % (f, var_list)
        exec(save_cmd)
        _printmsg("Saved to: %s.npz" % f, verbose)
    return None


def _savefig(path, filename, verbose=False, save=False):
    """
    Saves the data in kwargs in filename under given path.
    """
    if save:
        _printmsg("Saving figure...", verbose)
        f = path + filename
        pl.savefig(f)
        _printmsg("Saved to: %s" % f, verbose)
    else:
        pl.show()
    return None


def _savegraph(path, filename, G, verbose=False, save=False):
    """
    Saves igraph Graph object in filename under given path.
    """
    if save:
        _printmsg("Saving igraph Graph...", verbose)
        f = path + filename
        G.write_pickle(f)
        _printmsg("Saved to: %s" % f, verbose)
    return None


def _printmsg(msg, verbose):
    """
    Prints given message according to specified varbosity level.
    """
    if verbose:
        print(msg)
    return None


def _progressbar_start(max_value, pbar_on, pbar_title="Progress: "):
    """
    Starts a progress bar as per given maximum value.
    """
    prog_bar = None
    widgets = [pbar_title,
               Percentage(),
               ' ',
               Bar(marker=u'\u25fc', left='[', right=']', fill=u'\u00b7'),
               ' ',
               ETA(format="ETA:"),
               ' ',
               ]
    if pbar_on:
        prog_bar = ProgressBar(maxval=max_value,
                               widgets=widgets,
                               term_width=80)
        prog_bar.start()
    return prog_bar


def _progressbar_update(prog_bar, i):
    """
    Updates current progress bar with integer i.
    """
    if prog_bar:
        prog_bar.update(i)
    return None


def _progressbar_finish(prog_bar):
    """
    Terminates a given progress bar.
    """
    if prog_bar:
        prog_bar.finish()
    return None


def normalize(arr, method=None):
    """
    Normalizes given data according to specified method.
    """
    assert method, "Please provide the method for normalization!"
    _normfunc = eval("_normalize_%s" % method)

    notnan = ~np.isnan(arr)
    normed = _normfunc(arr[notnan])

    out = np.zeros(len(arr))
    out[notnan] = normed
    out[~notnan] = np.nan

    out[np.isinf(out)] = np.nan

    return out


def _normalize_minmax(arr):
    """
    Returns the min-max _normalized data for given array.
    """
    return (arr - arr.min()) / (arr.max() - arr.min())


def _normalize_gaussian(arr):
    """
    Returns the quantile normalization of the given array.
    """
    return ndtri(_cdf(arr))


def _normalize_percentile(arr):
    """
    Returns the percentile series for given array.
    """
    return _cdf(arr)

def _normalize_zscore(arr):
    """
    Returns the Z-scores of entries in given array.
    """
    return (arr - arr.mean()) / arr.std()


def _cdf(arr):
    """
    Returns the empirical CDF as a scipy.interpolate.interp1d object.
    """
    pdf, be = np.histogram(arr, bins="fd", density=True)
    rsw = (be[1:] - be[:-1])
    cdf = np.cumsum(pdf * rsw)
    bc = 0.5 * (be[1:] + be[:-1])
    f_cdf = interp1d(bc, cdf,
                     bounds_error=False,
                     # fill_value=(cdf[0], cdf[-1])
                     fill_value=(0., 1.)
                     )
    return f_cdf(arr)

def _quantile_normalization(arr, mode="mean"):
    """
    Normalizes the data to a Gaussian distribution using quantiles.
    """
    n = len(arr)
    perc = percentileofscore
    arr_ = arr.copy()[~np.isnan(arr)]
    out = np.zeros(n)
    for i in range(n):
        if not np.isnan(arr[i]):
            out[i] = norm.ppf(perc(arr_, arr[i], mode) / 100.)
        else:
            out[i] = np.nan
    return out


