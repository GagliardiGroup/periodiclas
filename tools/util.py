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
    def __init__(self,data=None,pkl_fn=None,pdft=False):
        if data is None:
            data = load_pkl(pkl_fn)
        if "energies_lassi" in data.keys():
            self.energies_lassi = data["energies_lassi"]
        else:
            self.energies_lassi = data["energies"]
        if pdft:
            self.energies_lassipdft = np.array(data["energies_lassipdft"])
        self.civecs = data["civecs"]
        self.charges = data["charges"]
        self.data = data
        self.pdft = pdft
        #Hamiltonian
        self.hdct = bandh.make_hdct(self.civecs,self.energies_lassi,self.charges,prnt=False)

    def get_homo(self):
        if not self.pdft:
            e,k = bandh.calc_band(hdct=self.hdct,band_charge=1).values()
        else:
            e,k = bandh.calc_band(self.civecs,self.energies_lassipdft,self.charges,band_charge=1).values()
        return e,k

    def get_lumo(self):
        if not self.pdft:
            e,k = bandh.calc_band(hdct=self.hdct,band_charge=-1).values()
        else:
            e,k = bandh.calc_band(self.civecs,self.energies_lassipdft,self.charges,band_charge=-1).values()
        return e,k

    def make_h(self,plot=False):
        return bandh.make_h(self.civecs,self.energies_lassi,plot=plot)

    def ip(self):
        e,k = self.get_homo()
        return -np.max(e)

    def ea(self):
        e,k = self.get_lumo()
        return -np.min(e)

    def make_bands(self,plot=True):
        homo_e, homo_k = self.get_homo()
        lumo_e, lumo_k = self.get_lumo()
        label = "LASSI"
        if self.pdft:
            label = "LASSI-PDFT"
        
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
        
        return df

class DMRGdata:
    def __init__(self,csv_fn,pdft=True):
        df = pd.read_csv(csv_fn,index_col=0)
        self.df = df.copy()
        if pdft:
            energies = df["e_mcpdft"]
        else:
            if "e_mcscf" in df.columns.tolist():
                energies = df["e_mcscf"]
            else:
                energies = df["e"]
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
        df = self.df.copy()
        homo_idx = np.where(self.mo_occ == 2)[0][-1]
        k = np.array(self.df.index).astype(float)
        energies = self.df.iloc[:,homo_idx].values
        energies *= self.hartree_to_ev
        return energies,k

    def get_lumo(self):
        df = self.df.copy()
        lumo_idx = np.where(self.mo_occ == 0)[0][0]
        k = np.array(self.df.index).astype(float)
        energies = self.df.iloc[:,lumo_idx].values
        energies *= self.hartree_to_ev
        return energies,k

    def ip(self):
        e,k = self.get_homo()
        return -np.max(e)

    def ea(self):
        e,k = self.get_lumo()
        return -np.min(e)

def plot_charges(charges,labels):
    df = pd.DataFrame()
    df["Value"] = charges
    names = []
    for i,l in enumerate(labels):
        name = l[2:]
        charge = np.round(charges[i],2)
        name = f"{name}\n({charge})"
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
        if angle in [np.pi, 2*np.pi]:
            ax.text(
                x=angle, 
                y=1.2,
                s=label, 
                ha=alignment, 
                va='center', 
                rotation=rotation, 
                rotation_mode="anchor")
        else:
            ax.text(
                x=angle, 
                y=1.1,
                s=label, 
                ha=alignment, 
                va='center', 
                rotation=rotation, 
                rotation_mode="anchor")

