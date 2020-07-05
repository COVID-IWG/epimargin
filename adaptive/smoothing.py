import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from typing import Optional, Sequence

kernels = { 
    "hanning"  : np.hanning,
    "hamming"  : np.hamming,
    "bartlett" : np.bartlett,
    "blackman" : np.blackman,
    "uniform"  : np.ones
}

def convolution(key: str = "hamming",  window: int = 7):
    kernel = kernels[key](window)
    def smooth(data: Sequence[float]):
        # pad the data with time reversal windows of signal at ends since all kernels here are apodizing 
        padded = np.r_[data[window-1:0:-1], data, data[-2:-window-1:-1]]
        return np.convolve(kernel/kernel.sum(), padded, mode="valid")
    return smooth

def box_filter_local(window: int = 5, local_smoothing: Optional[int] = 3):
    def smooth(data: Sequence[float]):
        box = np.ones(window)/window
        smoothed = np.convolve(data, box, mode='same')
        if local_smoothing and len(data) > (local_smoothing + 1):
            for i in range(local_smoothing-1, 0, -1):
                smoothed[-i] = np.mean(data[-i-local_smoothing+1: -i+1 if i > 1 else None])
        return smoothed
    return smooth 

def lowess(**kwargs):
    def smooth(data: Sequence[float]):
        sm_lowess(data, list(range(len(data))), **kwargs)
