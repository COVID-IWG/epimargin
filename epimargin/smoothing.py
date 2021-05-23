from typing import Optional, Sequence

import numpy as np
from scipy.signal import convolve, filtfilt, iirnotch
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

# supported kernels for convolution smoothing
kernels = { 
    "hanning"  : np.hanning,
    "hamming"  : np.hamming,
    "bartlett" : np.bartlett,
    "blackman" : np.blackman,
    "uniform"  : np.ones
}

def notched_smoothing(window: int = 7):
    """ Removes weekly and twice-weekly periodicity before convolving a time-reversed padded signal with a uniform moving average window"""
    fs, f0, Q = 1, 1/7, 1
    b1, a1 = iirnotch(f0, Q, fs)
    b2, a2 = iirnotch(2*f0, 2*Q, fs)
    # Frequency response
    b = convolve(b1, b2)
    a = convolve(a1, a2)
    kernel = np.ones(window)/window
    def smooth(data: Sequence[float]):
        notched = filtfilt(b, a, data)
        return convolve(np.concatenate([notched, notched[:-window-1:-1]]), kernel, mode="same")[:-window]
    return smooth

def notch_filter():
    """ implements a notch filter with notches at 1/week and 2/week; assuming input signal sampling rate is 1/week"""
    fs, f0, Q = 1, 1/7, 1
    b1, a1 = iirnotch(f0, Q, fs)
    b2, a2 = iirnotch(2*f0, 2*Q, fs)
    # Frequency response
    b = convolve(b1, b2)
    a = convolve(a1, a2)
    def filter_(data: Sequence[float]):
        return filtfilt(b, a, data)
    return filter_

def convolution(key: str = "hamming", window: int = 7):
    """ entry point for all convolution operations """
    kernel = kernels[key](window)
    def smooth(data: Sequence[float]):
        # pad the data with time reversal windows of signal at ends since all kernels here are apodizing 
        padded = np.r_[data[window-1:0:-1], data, data[-2:-window-1:-1]]
        return np.convolve(kernel/kernel.sum(), padded, mode="valid")[:-window+1]
    return smooth

def box_filter_local(window: int = 5, local_smoothing: Optional[int] = 3):
    """ implement a box filter smoother with additional LOWESS-like smoothing for data points at the end of the timeseries"""
    def smooth(data: Sequence[float]):
        smoothed = np.convolve(data, np.ones(window)/window, mode='same')
        if local_smoothing and len(data) > (local_smoothing + 1):
            for i in range(local_smoothing-1, 0, -1):
                smoothed[-i] = np.mean(data[-i-local_smoothing+1: -i+1 if i > 1 else None])
        return smoothed
    return smooth 

def lowess(**kwargs):
    """ wrapper over statsmodels lowess implementation to return a callable """
    return lambda data: sm_lowess(data, list(range(len(data))), **kwargs)
