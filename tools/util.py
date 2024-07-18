import numpy as np
from pyscf import gto, scf, lib, mcscf
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from . import bandh

def las_charges(las):
    las_charges = [[fcisolver.charge for fcisolver in las.fciboxes[i].fcisolvers] for i in range(len(las.fciboxes))]
    las_charges = np.array(las_charges).T
    return las_charges

class LASdata:
    def __init__(self,data,pkl_fn=None):
        energies = data["energies"]
        civecs = data["civecs"]
        charges = data["charges"]
        self.data = data
        self.hdct = bandh.make_hdct(civecs,energies,charges)

    def get_homo(self):
        e,k = bandh.calc_homo(self.hdct).values()
        return e,k

    def get_lumo(self):
        e,k = bandh.calc_lumo(self.hdct).values()
        return e,k

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
        ax.text(
            x=angle, 
            y=1.1,
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 

#For handling DMRG data:
# class DMRGdata:
#     def __init__(self,csv_fn):
#         df = pd.read_csv(csv_fn,index_col=0)
#         self.df = df.copy()
#         hartree_to_ev = 27.2114
#         df["e"] = df["e"]*hartree_to_ev
#         self.df = df
#         # assert((df["dw"] < 1e-10).all())
#         self.homo = df.loc[0,"e"] - df.loc[1,"e"]
#         self.lumo = df.loc[-1,"e"] - df.loc[0,"e"]
#         print(df["dw"])

