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
    
def make_hdct(civecs_lassi,energies_lassi,las_charges,plot=False,prnt=True):
    H = make_h(civecs_lassi,energies_lassi,plot=plot,prnt=prnt)
    hdct = {}
    for charge in [0,-1,1]:
        charges = np.array(las_charges).sum(axis=1)
        state_idxs = np.where(charges == charge)[0]
        hdct[charge] = H[np.ix_(state_idxs,state_idxs)]
    return hdct

def copy_bz(energies,k):
    energies = np.hstack([energies,energies])
    k = np.hstack([k,0.5 - (k-0.5)]) #reflection around 0.5
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

def calc_civecs_charges(civecs_lassi,las_charges):
    charges = las_charges.sum(axis=1)
    charges = (civecs_lassi**2 * charges[:,None]).sum(axis=0)
    return np.round(charges,0)

def calc_band(civecs=None,energies=None,las_charges=None,hdct=None,band_charge=1,reflect=True,ev=True):
    #band_charge = 1 --> HOMO
    #band_charge = -1 --> LUMO
    assert(band_charge in [1,-1])
    if (civecs is None) and (energies is None):
        #Diagonalize Hamiltonian
        #(Hamiltonian ill-defined for MC-PDFT)
        energies,civecs = np.linalg.eigh(hdct[band_charge])
        k = calc_disp(civecs)
        if band_charge == 1:
            energies = hdct[0][0,0] - energies
        else:
            energies = energies - hdct[0][0,0]
    else:
        civecs_charges = calc_civecs_charges(civecs,las_charges)
        charges = las_charges.sum(axis=1)
        idx = np.where(charges == band_charge)[0]
        cols = np.where(civecs_charges == band_charge)[0]
        gs_col = np.where(civecs_charges == 0)[0][0]
        civecs = civecs[np.ix_(idx,cols)]
        k = calc_disp(civecs)
        if band_charge == 1:
            energies = energies[gs_col] - energies[cols]
        else:
            energies = energies[cols] - energies[gs_col]
    if reflect:
        energies,k = copy_bz(energies,k)
    if ev:
        hartree_to_ev = 27.2114
        energies *= hartree_to_ev
    return {"energies":energies,"k":k}

def make_bands(civecs=None,energies=None,las_charges=None,hdct=None,reflect=True,ev=True,
               label="LASSI",plot=True):
    #Can take either civecs, energies, and charges OR the rootspace Hamiltonians
    energies = np.array(energies)
    homo = calc_band(civecs=civecs,energies=energies,las_charges=las_charges,hdct=hdct,
                     band_charge=1,reflect=reflect,ev=ev)
    lumo = calc_band(civecs=civecs,energies=energies,las_charges=las_charges,hdct=hdct,
                     band_charge=-1,reflect=reflect,ev=ev)
    df = pd.DataFrame()
    df.loc[label,"IP"] = -np.max(homo["energies"])
    df.loc[label,"EA"] = -np.min(lumo["energies"])
    df.loc[label,"GAP"] = np.min(lumo["energies"]) - np.max(homo["energies"])
    df = df.T
    
    if plot:
        plt.scatter(homo["k"],homo["energies"],label=f"{label} N-1")
        plt.scatter(lumo["k"],lumo["energies"],label=f"{label} N+1")
        plt.xlabel("k$d$/2$\pi$")
        if ev:
            plt.ylabel("Energy (eV)")
        else:
            plt.ylabel("Energy (Ha)")

    return df