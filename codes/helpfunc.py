import scipy as sp
from scipy.fftpack import fft, fftfreq
from scipy import sparse
from progressbar import ProgressBar
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import os, sys

def nearest_power_of_two(N):
	"""Computes the nearest power of 2 for a given integers N."""
	i = range(31) # for index > 30, scipy gives negative results
	poweroftwo = sp.power(2, i)
	idx = sp.argmin(abs(N - poweroftwo))
	return poweroftwo[idx]

def get_spectrum(signal, sampling_freq, verbose=1):
	"""Returns power spectrum & frequencies given a signal and sample frequency.

	NOTE:
	T = N/fs 	# acquisition time
	df = 1/T 	# = fs/N := frequency resolution
	fmax = fs/2 # maximum resolvable frequency (:= Nyquist ferquency)
	"""
	dt = 1./sampling_freq
	N = float(signal.size)
	fft_len = nearest_power_of_two(N)
	FFT = fft(signal, n=N)#fft_len
	freq = fftfreq(int(N), dt)#fft_len
	if verbose != 0:
		print "Maximum resolvable frequency is ", sampling_freq/2, "/yr"
		print "Frequency resolution of the spectra is ", sampling_freq/N, "/yr"
	return abs(FFT), freq


def rpfan(x, m=1, t=1, e=0.1):
    """ Returns array 'RP' with metric := Fixed Amount of Neighbours (FAN)
        
        x := time series
        m := dimension of embedding     (default=1)
        t := time delay of embedding    (default=1)
        e := recurrence threshold       (default=0.1)
        """
    # embed x with dimension 'm' and delay 't'
    n = x.shape[0]
    step = (m-1)*t
    count = step+1
    y = sp.zeros((n-count+1,m))
    for i in range(n-count+1):
        tt = i+step
        y[i,:] = x[-(n-tt):-(n-tt)-step-1:-t]
    # get distance matrix
    n = y.shape[0]
    dist = sp.zeros((n,n))
    if m > 1:
        for i in range(n-1):
            dist[i+1:n,i] = sp.sqrt(sp.sum(sp.square(sp.tile(y[i,:], 
                                                             (n-i-1, 1)) - 
                                                     y[i+1:n,:]), axis=1
                                           ))
    elif m == 1:
        for i in range(n-1):
            dist[i+1:n,i] = abs(sp.tile(y[i], (n-i-1,1)) - 
                                y[i+1:n]).squeeze()
    dist = dist + dist.T
    # compute fraction nearest neighbours
    lim = int(sp.floor(e*n))
    # compute RP
    RP = sp.zeros((n,n), dtype=sp.int8)
    for i in range(n):
        ii = sorted(range(n), key=lambda k: dist[i][k])
        RP[ii[:lim+1],i] = 1
    return RP

def rpeuc(x, m=1, t=1, e=0.1, normed=True):
	"""Returns the recurrence matrix based on Euclidean metric.
	"""
	x = x.squeeze()
	if normed: x = (x - x.mean()) / x.std()
	# embed x with dimension 'm' and delay 't'
	n = x.shape[0]
	step = (m-1)*t
	count = step+1
	y = sp.zeros((n-count+1,m))
	for i in range(n-count+1):
		tt = i+step
		y[i,:] = x[-(n-tt):-(n-tt)-step-1:-t]
	# get distance matrix
	n = y.shape[0]
	dist = sp.zeros((n,n))
	if m > 1:
		for i in range(n-1):
		    dist[i+1:n,i] = sp.sqrt(sp.sum(sp.square(sp.tile(y[i,:], 
								     (n-i-1, 1)) -                                                      y[i+1:n,:]), axis=1
						   ))            
	elif m == 1:
		for i in range(n-1):                   
		    dist[i+1:n,i] = abs(sp.tile(y[i], (n-i-1,1)) -
					y[i+1:n]).squeeze()
	dist = dist + dist.T
	RP = sp.zeros((n,n), dtype=sp.int8)
	RP[(dist <= e) & (dist > 0)] = 1
	RP = RP + sp.eye(n, dtype=sp.int8)
	return RP

def rneuc(x, m=1, t=1, e=0.1, normed=True):
	"""Recurrence network adjacency matrix with Euclidean metric.  
	"""
	RP = rpeuc(x, m, t, e, normed=normed)
	n = x.shape[0]
	step = (m-1)*t
	count = step+1
	n = n-count+1
	A = RP - sp.eye(n, dtype=sp.int8)
	return A

def twinsurr(X, m=1, t=1, e=0.1, nSurr=100, RP=None):
	""" Returns Twin Surrogates (TS) based on RP with FAN metric.

        X := time series
        m := dimension of embedding     (default = 1)
        t := time delay of embedding    (default = 1)
        e := recurrence threshold       (default = 0.1)
        nSurr := number of Surrogates   (default = 100)
        RP := recurrence Plot of X

        Output:
        S := matrix where each column is a TS
        nTwins := number of twins found
        twinMat := matrix of twin pairs twinMat[i, j] = 1 => twins
	"""
	if RP is None:
		RP = rpfan(X, m, t, e)
	nX = len(RP)
	print 'Searching for twins...'
	twinMat = sparse.lil_matrix((nX, nX), dtype=sp.int8)
	pbar = ProgressBar(maxval=nX).start()
	for j in range(nX):
		i = sp.tile(RP[:, j], (nX, 1)).T
		i = sp.all(RP == i, axis=0) * any(RP[:, j])
		twinMat[i, j] = 1
		pbar.update(j + 1)
	pbar.finish()
	nTwins = sum(sp.any((twinMat - sparse.eye(nX, nX)).toarray(), axis=0))
	if nTwins == 0:
		print 'Warning: No twins detected!'
		print 'Surrogates are same as original time series!'
	S = sp.empty((nX, nSurr))
	print 'Creating surrogates...'
	pbar = ProgressBar(maxval=nSurr).start()
	for i in range(nSurr):
		k, l = 0, sp.ceil(nX * sp.rand()) - 1
                k, l = int(k), int(l)
		S[k, i] = X[l]
		while k < nX - 1:
			twins = sp.where(twinMat[:, l].toarray().squeeze())[0]
			if len(twins) > 1:
                                idx = int(sp.ceil(len(twins) * sp.rand()) - 1)
				l = twins[idx]
			l += 1
			if l > nX - 1:
				while l > nX - 1:
					l = sp.ceil(nX * sp.rand()) - 1
                        l = int(l)
			S[k + 1, i] = X[l]
			k += 1
		pbar.update(i + 1)
	pbar.finish()
	return S, nTwins, twinMat.toarray()

def its(func_list, N=1, res=10000):
	T = len(func_list)
	sample = sp.empty([T, N])
	r_int = sp.random.random_integers
	for i in range(T):
		func_inv = interp1d(func_list[i].y, func_list[i].x)
		sample[i, 0] = sp.nan
		while any(sp.isnan(sample[i, :])):
			r = r_int(0, res, size=N) / float(res)
			while min(r) < min(func_inv.x) or max(r) > max(func_inv.x):
				r = r_int(0, res, size=N) / float(res)
			sample[i, :] = func_inv(r)
	return sample


def resample(name='NAO1', fs=2., **kwargs):
	# load the data
	datpath = '../output/'
	datname = 'nao_indices_2013-10-05_minmax.npz'
	data = sp.load(datpath + datname)
	NAO = data[name].tolist()
	# interpolate the index to an evenly sampled time grid
	fs = 4. # := sampling frq = 4 => 4 samples/yr => inter-sample time = 0.25 yr
	f_nao1 = interp1d(NAO['time'].squeeze(), NAO['index'].squeeze())
	t_nao1_new = sp.arange(NAO['time'].min(), NAO['time'].max() + 0.1, 1/fs)
	y_nao1_new = f_nao1(t_nao1_new)
	return t_nao1_new, y_nao1_new

def autocorr(x):
    result = sp.correlate(x, x, mode='full')
    return result[result.size/2:]

def taurec(RP):
	N = len(RP)
	p = sp.empty(N)
	pbar = ProgressBar(maxval=N).start()
	for tau in range(N):
		recnum = 0
		for i in range(N - tau):
			recnum += RP[i, i + tau]
		p[tau] = recnum / float(N - tau)
		pbar.update(tau + 1)
	pbar.finish()
	autocorr_time = sp.where(p < 1./sp.exp(1))[0][0]
	return p, range(N), autocorr_time

def clayton(u1, u2, theta, cumulative=False):
	if cumulative:
		p = (u1 ** (-theta) + u2 ** (-theta) - 1) ** (-1 / theta)
	else:
		x = (theta + 1) * (u1 * u2) ** (-(theta + 1))
		y = (u1 ** (-theta) + u2 ** (-theta) - 1) ** -((2 * theta + 1) / theta)
		p =  x * y
	return p

def joint(x1, x2, X1, X2, theta, cumulative=False):
	u1, u2 = X1.cdf(x1), X2.cdf(x2)
	if cumulative:
		copula = clayton(u1, u2, theta, cumulative=True)
		joint = copula
	else:		
		copula = clayton(u1, u2, theta)
		joint = copula * X1.pdf(x1) * X2.pdf(x2)
	return joint

def shapiro(s1, s2, corr):
	s1_ascend, s2_ascend = sp.sort(s1), sp.sort(s2)
	if len(s1.shape) == 1:
		rhomax = pearsonr(s1_ascend, s2_ascend)[0]
		sortupto = sp.floor((corr / rhomax) * len(s1))
		out1 = sp.hstack((sp.sort(s1[:sortupto]), s1[sortupto:]))
		out2 = sp.hstack((sp.sort(s2[:sortupto]), s2[sortupto:]))
	else:
		N = len(s1)
		print 'generating correlated samples...'
		pbar = ProgressBar(maxval=N).start()
		for i in range(N):
			rhomax = pearsonr(s1_ascend[i], s2_ascend[i])[0]
			sortupto = sp.floor((corr / rhomax) * s1.shape[1])
			out1 = sp.hstack((sp.sort(s1[:, :sortupto]), s1[:, sortupto:]))
			out2 = sp.hstack((sp.sort(s2[:, :sortupto]), s2[:, sortupto:]))
			pbar.update(i + 1)
		pbar.finish()
	return out1, out2

def hist2d(s1, s2, res=(100,100)):
	h2d = sp.histogram2d
	if len(s1.shape) == 1:
		h, xe, ye = h2d(s1, s2, normed=True, bins=res)
		xm, ym = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
		int_wd = sp.diff(xe)
	else:
		N = len(s1)
		h = sp.empty((res[0], res[1], N))
		xm, ym = sp.empty((N, res[0])), sp.empty((N, res[1]))
		int_wd = sp.empty((N, res[0]))
		print 'generating 2-D histogram...'
		pbar = ProgressBar(maxval=N).start()
		for i in range(N):
			h_, xe, ye = h2d(s1[i], s2[i], normed=True, bins=res)
			h[:, :, i], xm[i, :], ym[i, :] = h_, mid_pt(xe), mid_pt(ye)
			int_wd[i, :] = sp.diff(xe)
			pbar.update(i + 1)
		pbar.finish()
	return h, xm, ym, int_wd


def mid_pt(array):
	return 0.5 * (array[:-1] + array[1:])


def add_unique_postfix(fn):
    """ensures if name 'fn' in unique or gives new one."""
    if not os.path.exists(fn):
        return fn
    path, name = os.path.split(fn)
    name, ext = os.path.splitext(name)
    make_fn = lambda i: os.path.join(path, '%s(%d)%s' % (name, i, ext))
    for i in xrange(2, sys.maxint):
        uni_fn = make_fn(i)
        if not os.path.exists(uni_fn):
            return uni_fn


def iaaft(X, Ns, preserve="power"):
    """Returns Ns iAAFT surrogates of the time series X"""
    Nx = len(X)
    Xs = np.zeros((Ns, Nx))
    ri = np.zeros((Ns, Nx))
    jj = np.arange(Nx)).astype("int")
    for k in range(Ns):
        ri[k] = np.random.permutation(jj)
    print ri



    return Xs
