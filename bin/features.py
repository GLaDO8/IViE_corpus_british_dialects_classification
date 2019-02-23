import numpy as np
import pandas as pd
import IPython.display as ipd
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from zipfile import ZipFile
import matplotlib.pyplot as plt
from scipy.stats import skew

#parameters
framelength = 0.025
framestride = 0.015
nfft = 512
num_fbanks = 40
n_cep_coeff = 12
N = 2

#data import
def audio_import(nclass, naudio):
    rate, data = wav.read('Read_Up/IDR' + str(nclass) + '/' + str(naudio) + '.wav')
    filename = 'Data/IDR' + str(nclass) + '/' + str(naudio) + '.csv'
    return rate, data, filename

#generate frames
def frames_gen(rate, data, framelength, framestride):
    frmlen, frmstrd, signallen = int(round(rate*framelength)), int(round(rate*framestride)), len(data)
    paddinglen = frmstrd - (signallen - frmlen) % frmstrd #making number of frames even
    paddedsig = np.concatenate((data, np.zeros(paddinglen)), axis = 0)
    paddedsiglen = len(paddedsig)
    nframes = int(np.floor((paddedsiglen - frmlen)/frmstrd) + 1)
    indices = np.tile(np.arange(0, frmlen), (nframes, 1)) + np.tile((np.arange(0, nframes*frmstrd, frmstrd)), (frmlen, 1)).T
    frames = paddedsig[indices]
    return frames, frmlen

#apply hamming window to each frame
def hamming_window(frames, frmlen):
    frames *= np.hamming(frmlen)
    return frames

#convert each windowed frame into a power spectrum
def periodogram_gen(frames, nfft):
    frame_fft = np.absolute(np.fft.rfft(frames, n = nfft, axis = 1))
    frame_periodogram = np.square(frame_fft)/nfft
    return frame_periodogram

#helper functions
def freq_to_mel(freq):
    return 2595*np.log10(1+freq/700)
def mel_to_freq(mel):
    return 700*(np.power(10, mel/2595) - 1)

# making mel-scale filterbank
def filter_bank_gen(rate, num_fbanks, nfft):
    #for x filter banks, we need x+2 mel points
    low_mel_lim = 0
    up_mel_lim = freq_to_mel(rate/2)
    mel_range = np.linspace(low_mel_lim, up_mel_lim, num_fbanks + 2)
    freq_range = mel_to_freq(mel_range)
    bins = np.floor((nfft + 1) * freq_range/rate)
    fbank = np.zeros((num_fbanks, int(np.floor(nfft/2 + 1))))
    for m in range(1, num_fbanks + 1):
        lower = int(bins[m - 1]) # lower
        peak = int(bins[m]) # peak
        upper = int(bins[m + 1]) # upper
        for k in range(lower, peak):
            fbank[m - 1, k] = (k - bins[m - 1])/(bins[m] - bins[m - 1])
        for k in range(peak, upper):
            fbank[m - 1, k] = (bins[m + 1] - k)/(bins[m + 1] - bins[m])
    return fbank

# filtered frames
def filtered_frame_gen(frame_periodogram, fbank):
    #multiply each frame with all filterbanks and add up for coefficients.
    filter_banks = np.dot(frame_periodogram, fbank.T)
    #for numerical stability
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) #if condition is true, return eps, else return original val
    filter_banks = 20*np.log10(filter_banks)
    return filter_banks

#make mfcc coefficients
def mfcc_gen(filter_banks, n_cep_coeff):
    mfcc = dct(filter_banks, type = 2, axis = 1, norm = 'ortho')[:, 1:(n_cep_coeff + 1)]
    return mfcc  

#make delta and delta-delta coefficients
def ctpn(n_cep_coeff, coeff_type, t, n):
    if((t+n) > n_cep_coeff-1):
        return coeff_type[:,n_cep_coeff-1]
    elif(0 <= (t+n) <= n_cep_coeff-1):
        return coeff_type[:, t+n]

def ctmn(n_cep_coeff, coeff_type, t, n):
    if((t-n) < 0):
        return coeff_type[:,0]
    elif(0 <= (t-n) <= n_cep_coeff-1):
        return coeff_type[:, t-n]   

#delta and delta-delta coefficients generator
def deltacoeff(t, coeff_type):
    dt = 0
    for n in range(1,N):
        dt+= n*(ctpn(n_cep_coeff, coeff_type, t, n) - ctmn(n_cep_coeff, coeff_type, t, n))/2*np.square(n)
    return dt

def deltacoeff_gen(coeff_type, n_cep_coeff):
    deltacoef = np.zeros(coeff_type.shape)
    for t in range(0, n_cep_coeff):
        dt = deltacoeff(t, coeff_type)
        deltacoef[:, t] = dt
    return deltacoef

def deltadeltacoeff_gen(deltacoef, n_cep_coeff):
    deltadeltacoef = np.zeros(deltacoef.shape)
    for t in range(0, n_cep_coeff):
        ddt = deltacoeff(t, deltacoef)
        deltadeltacoef[:, t] = ddt
    return deltadeltacoef