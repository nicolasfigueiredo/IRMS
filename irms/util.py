import numpy as np
import math


def find_nearest(array, value):
    # Find the element of 'array' closest to 'value'
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return (idx-1)
    else:
        return idx


# TIRADO DO LIBROSA
def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`
    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
    endpoint=True)