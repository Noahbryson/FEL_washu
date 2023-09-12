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