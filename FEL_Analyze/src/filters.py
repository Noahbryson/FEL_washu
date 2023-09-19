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