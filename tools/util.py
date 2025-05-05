import numpy as np
from pyscf import gto, scf, lib, mcscf
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from . import bandh
import pickle

def dump_pkl(obj,fn):
    with open(fn,"wb+") as file:
        pickle.dump(obj,file)
        
def load_pkl(fn):
    with open(fn,"rb") as file:
        return pickle.load(file)

def las_charges(las):
    las_charges = [[fcisolver.charge for fcisolver in las.fciboxes[i].fcisolvers] for i in range(len(las.fciboxes))]
    las_charges = np.array(las_charges).T
    return las_charges

class LASdata:
    def __init__(self,data=None,pkl_fn=None,pct_pdft=0.75):
        assert(pct_pdft in [1,0.5,0.75,0]) #0.75 --> tPBE0
        if data is None:
            data = load_pkl(pkl_fn)
        if "energies_lassi" in data.keys():
            self.energies_lassi = data["energies_lassi"]
        else:
            self.energies_lassi = data["energies"]
        if "energies_lassipdft" in data.keys():
            self.energies_lassipdft = np.array(data["energies_lassipdft"])
        else:
            hpdft = 0
        self.civecs = data["civecs"]
        self.charges = data["charges"]
        self.mo_coeff = data["mf_coeff"]
        self.data = data
        self.pct_pdft = pct_pdft
        self.hartree_to_ev = 27.2114
        #Hamiltonian
        self.hdct = bandh.make_hdct(self.civecs,self.energies_lassi,self.charges,prnt=False)
        self.diag_then_perturb = True
        if "civecs_pdft" in data.keys():
            self.civecs_pdft = data["civecs_pdft"]
            self.diag_then_perturb = False
            self.hdct_pdft = bandh.make_hdct(self.civecs_pdft,self.energies_lassipdft,self.charges,prnt=False)
        else:
            self.civecs_pdft = None

    def make_band(self,q,sympoint=0.5):
        ecas,k = bandh.calc_band(hdct=self.hdct,band_charge=q,sympoint=sympoint).values()
        if self.diag_then_perturb:
            if self.pct_pdft != 0:
                epdft,k = bandh.calc_band(self.civecs,self.energies_lassipdft,
                                          self.charges,band_charge=q,sympoint=sympoint).values()
                e = self.pct_pdft*epdft + (1-self.pct_pdft)*ecas
            else:
                e = ecas
        else:
            hdct = self.hdct
            hdct_pdft = self.hdct_pdft
            hdct_hpdft = {}
            for k in hdct.keys():
                hdct_hpdft[k] = self.pct_pdft*hdct_pdft[k] + (1-self.pct_pdft)*hdct[k]
            e,k = bandh.calc_band(hdct=hdct_hpdft,band_charge=q,sympoint=sympoint).values()
        return e,k
    
    def get_homo(self,sympoint=0.5):
        return self.make_band(1,sympoint=sympoint)
            
    def get_lumo(self,sympoint=0.5):
        return self.make_band(-1,sympoint=sympoint)
        
    def plot_h(self,pdft=False,nodiag=False,minval=-10):
        if pdft:
            if self.civecs_pdft is not None:
                H = bandh.make_h(self.civecs_pdft,self.energies_lassipdft,plot=False)
            else:
                H = bandh.make_h(self.civecs,self.energies_lassipdft,plot=False)
        else:
            H = bandh.make_h(self.civecs,self.energies_lassi,plot=False)
        if nodiag:
            sns.heatmap(H - np.diag(np.diag(H)),cbar_kws={'label': "$H_{ij}$"})
        else:
            for i in range(H.shape[0]):
                H[i][np.where(H[i] != 0)] = np.log10(np.abs(H[i][np.where(H[i] != 0)]))
                H[i][np.where(H[i] < minval)] = minval #minimum shown value
                H[i][np.where(H[i] == 0)] = minval #arbitrary
            sns.heatmap(H,cbar_kws={'label': "$\log_{10}(|H_{ij}|)$"})
        
    def ip(self):
        e,k = self.get_homo()
        return -np.max(e)

    def ea(self):
        e,k = self.get_lumo()
        return -np.min(e)

    def hwidth(self):
        e,k = self.get_homo()
        return np.max(e) - np.min(e)

    def lwidth(self):
        e,k = self.get_lumo()
        return np.max(e) - np.min(e)

    def finite_dif_2div(self,e,k,prnt=True):
        e = e[np.argsort(k)]
        k = np.sort(k)
        idx2 = len(k)//2
        idx1 = idx2-2
        idx3 = idx2+1
        if prnt:
            print(np.round(k[[idx1,idx2,idx3]],4))
        num = e[idx3] + e[idx1] - 2*e[idx2]
        denom = (k[idx3] - k[idx2])**2
        num *= 1/self.hartree_to_ev
        return num/denom

    def lmass(self,prnt=False):
        e,k = self.get_lumo()
        div = self.finite_dif_2div(e,k,prnt=prnt)
        return 1/div

    def hmass(self,prnt=False):
        e,k = self.get_homo()
        div = self.finite_dif_2div(e,k,prnt=prnt)
        return 1/div

    def make_bands(self,plot=True,sympoint=0.5):
        homo_e, homo_k = self.get_homo(sympoint=sympoint)
        lumo_e, lumo_k = self.get_lumo(sympoint=sympoint)
        if self.pct_pdft == 1:
            label = "LASSI-tPBE"
        elif self.pct_pdft == 0.75:
            label = "LASSI-tPBE0"
        elif self.pct_pdft == 0:
            label = "LASSI"
        else:
            label = f"LASSI-{self.pct_pdft}tPBE"
        
        df = pd.DataFrame()
        df.loc[label,"IP"] = -np.max(homo_e)
        df.loc[label,"EA"] = -np.min(lumo_e)
        df.loc[label,"GAP"] = np.min(lumo_e) - np.max(homo_e)
        df = df.T

        if plot:
            plt.scatter(homo_k,homo_e,label=f"{label} N-1")
            plt.scatter(lumo_k,lumo_e,label=f"{label} N+1")
            plt.xlabel("k$d$/2$\pi$")
            plt.ylabel("Energy (eV)")
        
        return np.round(df,2)

class DMRGdata:
    def __init__(self,csv_fn,pct_pdft=0):
        assert(pct_pdft in [1,0.5,0.75,0]) #0.75 --> tPBE0
        df = pd.read_csv(csv_fn,index_col=0)
        self.df = df.copy()
        if "e_mcscf" in df.columns.tolist():
            ecas = df["e_mcscf"]
        else:
            ecas = df["e"]
        if pct_pdft > 0:
            epdft = df["e_mcpdft"]
            energies = pct_pdft*epdft + (1-pct_pdft)*ecas
        else:
            energies = ecas
        hartree_to_ev = 27.2114
        energies *= hartree_to_ev
        self.homo = energies[0] - energies[1]
        self.lumo = energies[-1] - energies[0]
        print(df["dw"])

    def ip(self):
        return -self.homo

    def ea(self):
        return -self.lumo

class PeriodicData: #Periodic
    def __init__(self,csv_fn):
        self.df = pd.read_csv(csv_fn,index_col=0)
        self.mo_occ = self.df.loc["nocc"]
        self.df = self.df.drop("nocc")
        self.hartree_to_ev = 27.2114

    def get_homo(self):
        homo_idx = np.where(self.mo_occ == 2)[0][-1]
        k = np.array(self.df.index).astype(float)
        energies = self.df.iloc[:,homo_idx].values.copy()
        energies *= self.hartree_to_ev
        return energies,k

    def get_lumo(self):
        lumo_idx = np.where(self.mo_occ == 0)[0][0]
        k = np.array(self.df.index).astype(float)
        energies = self.df.iloc[:,lumo_idx].values.copy()
        energies *= self.hartree_to_ev
        return energies,k

    def ip(self):
        e,k = self.get_homo()
        return -np.max(e)

    def ea(self):
        e,k = self.get_lumo()
        return -np.min(e)

    def hwidth(self):
        e,k = self.get_homo()
        return np.max(e) - np.min(e)

    def lwidth(self):
        e,k = self.get_lumo()
        return np.max(e) - np.min(e)

    def finite_dif_2div(self,e,k,prnt=True):
        e = e[np.argsort(k)]
        k = np.sort(k)
        idx2 = len(k)//2
        idx1 = idx2-1
        idx3 = idx2+1
        if prnt:
            print(np.round(k[[idx1,idx2,idx3]],4))
        num = e[idx3] + e[idx1] - 2*e[idx2]
        denom = (k[idx3] - k[idx2])**2
        num *= 1/self.hartree_to_ev
        return num/denom

    def lmass(self,prnt=False):
        e,k = self.get_lumo()
        div = self.finite_dif_2div(e,k,prnt=prnt)
        return 1/div

    def hmass(self,prnt=False):
        e,k = self.get_homo()
        div = self.finite_dif_2div(e,k,prnt=prnt)
        return 1/div

def plot_charges(charges,labels):
    df = pd.DataFrame()
    df["Value"] = charges
    names = []
    for i,l in enumerate(labels):
        name = l[2:]
        charge = np.round(charges[i],2)
        name = f"{name}\n(${charge}$)"
        names += [name]
    df["Name"] = names
    colors = []
    for c in charges:
        if c > 0:
            colors += ["blue"]
        else:
            colors += ["red"]
    df["Value"] = np.abs(df["Value"])
    
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={"projection": "polar"})
    
    upperLimit = 1
    lowerLimit = 0
    
    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (1 - lowerLimit) / 1
    heights = slope * df.Value + lowerLimit
    
    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df.index)
    
    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index)+1))
    angles = [element * width for element in indexes]
    
    # Draw bars
    bars = ax.bar(
        color=colors,
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        edgecolor="white")
    
    # ax.set_xticks(ANGLES)
    ax.set_xticklabels([""]*8);
    ax.set_ylim(0,1)
    ax.grid(axis="x")
    # ax.spines['polar'].set_visible(False)
    
    ax.vlines(angles, 0, 1, color="grey", ls=(0, (4, 4)), zorder=11)
    ax.set_rlabel_position(10) 
    
    # little space between the bar and the label
    labelPadding = 0
    
    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, df["Name"]):
    
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)
    
        # Flip some labels upside down
        alignment = ""
        if angle == 2*np.pi:
            # print("hi")
            alignment = "center"
        elif angle == np.pi:
            alignment = "center"
        elif angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"
        rotation=0

        # Finally add the labels
        # print(angle,label)
        fs = 13
        if angle in [np.pi, 2*np.pi]:
            ax.text(
                x=angle, 
                y=1.25,
                s=label, 
                ha=alignment, 
                va='center',
                fontsize=fs,
                rotation=rotation, 
                rotation_mode="anchor")
        else:
            ax.text(
                x=angle, 
                y=1.15,
                s=label, 
                ha=alignment, 
                va='center',
                fontsize=fs,
                rotation=rotation, 
                rotation_mode="anchor")


