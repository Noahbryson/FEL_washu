import scipy.signal as sig
import numpy as np
def highpass(data:np.ndarray or list, fs:int, Wn:float, order:int = 4):
    sos = sig.butter(N=order, Wn=Wn, btype='High', output='sos', fs=fs)
    return sig.sosfiltfilt(sos, data)
def lowpass(data:np.ndarray or list, fs:int, Wn:float, order:int = 4):
    sos = sig.butter(N=order, Wn=Wn, btype='Low', output='sos', fs=fs)
    return sig.sosfiltfilt(sos, data)
def notch(data:np.ndarray or list, fs:int, Wn:float, Q:int, order:int = 4):
    b,a = sig.iirnotch(Wn, Q, fs)
    for _ in range(order):
        data = sig.filtfilt(b,a,data)
    return data

def getDeltaBand(data:np.ndarray, fs, order = 4):
        """Surface EEG 0.4-4Hz"""
        data = highpass(data,fs=fs,Wn=0.4,order=order)
        data = lowpass(data,fs=fs,Wn=4,order=order)
        return data
    
def getThetaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 4-8Hz"""
    data = highpass(data,fs=fs,Wn=4,order=order)
    data = lowpass(data,fs=fs,Wn=8,order=order)
    return data

def getAlphaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 8-13Hz"""
    data = highpass(data,fs=fs,Wn=8,order=order)
    data = lowpass(data,fs=fs,Wn=13,order=order)
    return data
    
def getBetaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 14-30Hz"""
    data = highpass(data,fs=fs,Wn=13,order=order)
    data = lowpass(data,fs=fs,Wn=30,order=order)
    return data
    
def getGammaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 30-55 Hz"""
    data = highpass(data,fs=fs,Wn=30,order=order)
    data = lowpass(data,fs=fs,Wn=55,order=order)
    return data

def EMavg_filter(data, smoothing):
    movingAvg = np.empty(data.shape)
    movingAvg[0:4] = data[0:4]
    for i in range(4,len(movingAvg)):
        movingAvg[i] = (smoothing*data[i]+(1-smoothing)*np.mean(movingAvg[0:i-1]))
    return movingAvg

def windowed_EMavg(data,windowLen,smoothing):
    samples = len(data)
    chunkSize = int(samples/windowLen)
    chunks = np.array_split(data,chunkSize)
    output = []
    for i,dat in enumerate(chunks):
        output.append(EMavg_filter(dat,smoothing))
    return flattenList(output)
    
def flattenList(dat:list):
    out = []
    for ele in dat:
        out.extend(ele)
    return out

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    ----------
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    ----------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    ----------"""
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')