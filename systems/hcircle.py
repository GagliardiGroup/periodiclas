import numpy as np
import pandas as pd
from pyscf import gto, scf, lib, mcscf
import math
import os
from pyscf.mcscf import avas
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import lassi

import sys
sys.path.append('..')
from tools import rotsym, sign_control

class HCircle:
    def __init__(self,dist,num_h,num_h_per_frag,fn="output.log"):
        self.dist = dist
        self.num_h = num_h
        self.num_h_per_frag = num_h_per_frag
        self.nfrags = num_h // num_h_per_frag
        self.fn = fn
        assert(self.num_h % self.num_h_per_frag == 0)
        assert(self.nfrags%2 == 0)

    def get_mol(self,basis="sto-3g",plot=False):
        rnum = self.dist**2
        rdenom = 2*(1-np.cos(2*np.pi/self.num_h))
        radius = np.sqrt(rnum/rdenom)
        
        def polygon(sides, radius=1, rotation=0, translation=None):
            one_segment = np.pi * 2 / sides
        
            points = [
                (math.sin(one_segment * i + rotation) * radius,
                 math.cos(one_segment * i + rotation) * radius)
                for i in range(sides)]
        
            if translation:
                points = [[sum(pair) for pair in zip(point, translation)]
                          for point in points]
        
            return points

        points = polygon(self.num_h,radius)
        df = pd.DataFrame(points,columns=["x","y"])
        df["el"] = "H"

        if plot:
            plt.scatter(df["x"],df["y"])
            plt.gca().set_aspect('equal')
        
        mol = gto.Mole()
        atms = []
        for i,row in df.iterrows():
            x,y,el = row["x"],row["y"],row["el"]
            atms += [(el,(x,0,y))]
        mol.atom = atms
        mol.basis = "sto-3g"
        mol.output = self.fn
        mol.verbose = lib.logger.INFO
        mol.symmetry = False
        mol.build()
        return mol

    def make_and_run_hf(self):
        mol = self.get_mol()
        mf = scf.ROHF(mol)
        mf.kernel()
        self.mf_coeff = mf.mo_coeff
        self.mf_occ = mf.mo_occ
        self.mf_ene = mf.mo_energy
        return mf

    def make_las_init_guess(self):
        mf = self.make_and_run_hf()
        mol = mf.mol
        nfrags = self.nfrags
        nao_per_frag = mol.nao // nfrags
        nelec_per_frag = mol.nelectron // nfrags
        natoms_per_frag = len(mol._atom)//nfrags
        
        ref_orbs = [nao_per_frag]*(nfrags)
        ref_elec = [nelec_per_frag]*(nfrags)
        las = LASSCF(mf, ref_orbs, ref_elec)

        frag_atoms = [[natoms_per_frag*i+j for j in range(natoms_per_frag)] for i in range(nfrags)]
        las.mo_coeff = las.localize_init_guess(frag_atoms, mf.mo_coeff)
        las.mo_coeff = sign_control.fix_mos(las)
        return las

    def make_las_state_average(self):
        las = self.make_las_init_guess()
        nfrags = self.nfrags
        
        #Lists of N Lists: Define N root spaces for LASSI
        las_charges = []
        las_spins = [] #2s
        las_smults = [] #2s+1
        las_wfnsyms = []
        
        #Neutral rootspace as reference
        las_charges += [[0]*nfrags]
        las_spins += [[0]*nfrags]
        las_smults += [[las_spins[0][0]+1]*nfrags]
        
        #Specify rootspaces for homo:
        for i in range(nfrags):
            idxarr = np.eye(nfrags)[:,i].astype(int)
        
            #Charges -- 1, 0
            las_charges += [list(idxarr)]
        
            #2*Spins -- 1/2, 0 --> 1,0
            spins = idxarr
            las_spins += [list(spins)]
        
            #Smults
            las_smults += [list(spins + 1)]
        
        #Specify rootspaces for lumo:
        for i in range(nfrags):
            idxarr = np.eye(nfrags)[:,i].astype(int)
        
            #Charges -- -1, 0
            las_charges += [list(-idxarr)]
        
            #2*Spins -- 1/2, 0 --> 1,0
            spins = idxarr
            las_spins += [list(spins)]
        
            #Smults
            las_smults += [list(spins + 1)]
        
        nrootspaces = len(las_charges)
        las_weights = np.ones(nrootspaces)/nrootspaces
        las = las.state_average(las_weights,las_charges,las_spins,las_smults)
        las.max_cycle_macro = 100 
        return las
    
