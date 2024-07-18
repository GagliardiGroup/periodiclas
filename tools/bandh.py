import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_h(civecs_lassi,energies_lassi,plot=False,prnt=True):
    H = np.zeros(civecs_lassi.shape)
    for i in range(civecs_lassi.shape[0]):
        H += np.outer(civecs_lassi[:,i],civecs_lassi[:,i]) * energies_lassi[i]
    if prnt:
        print(np.round(np.diag(H),2))
    if plot:
        sns.heatmap(H - np.diag(np.diag(H)))
    return H
    
def make_hdct(civecs_lassi,energies_lassi,las_charges,plot=False):
    H = make_h(civecs_lassi,energies_lassi,plot=plot)
    hdct = {}
    for charge in [0,-1,1]:
        charges = np.array(las_charges).sum(axis=1)
        state_idxs = np.where(charges == charge)[0]
        hdct[charge] = H[np.ix_(state_idxs,state_idxs)]
    return hdct

def copy_bz(energies,k):
    energies = np.hstack([energies,energies[::-1]])
    k = np.hstack([k,k+0.5])
    return energies,k

def calc_disp(civecs):
    avgk = []
    for i in range(civecs.shape[1]):
        fft = np.fft.rfft(civecs[:,i]) #fft
        fft = fft / np.linalg.norm(fft) #normalize
        ps = np.abs(fft*np.conj(fft)) #power spectrum
        kdover2pi = np.arange(len(fft))/civecs.shape[0]
        avg = sum(ps * kdover2pi)
        avgk += [avg]
    return np.array(avgk)

def calc_homo(hdct,reflect=True,ev=True):
    energies,civecs = np.linalg.eigh(hdct[1])
    k = calc_disp(civecs)
    energies = hdct[0][0,0] - energies
    if reflect:
        energies,k = copy_bz(energies,k)
    if ev:
        hartree_to_ev = 27.2114
        energies *= hartree_to_ev
    return {"energies":energies,"k":k}

def calc_lumo(hdct,reflect=True,ev=True):
    energies,civecs = np.linalg.eigh(hdct[-1])
    k = calc_disp(civecs)
    energies = energies - hdct[0][0,0]
    if reflect:
        energies,k = copy_bz(energies,k)
    if ev:
        hartree_to_ev = 27.2114
        energies *= hartree_to_ev
    return {"energies":energies,"k":k}

def make_bands(hdct,reflect=True,ev=True,plot=True):
    homo = calc_homo(hdct,reflect=reflect,ev=ev)
    lumo = calc_lumo(hdct,reflect=reflect,ev=ev)
    
    df = pd.DataFrame()
    df.loc["las","homo"] = np.max(homo["energies"])
    df.loc["las","lumo"] = np.min(lumo["energies"])
    
    if plot:
        plt.scatter(homo["k"],homo["energies"],label="LASSI N-1")
        plt.scatter(lumo["k"],lumo["energies"],label="LASSI N+1")