import numpy as np
import pandas as pd
import math
import os
from pyscf import gto, scf, lib, mcscf
from pyscf.mcscf import avas
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import lassi
from .hcircle import HCircle

import sys
sys.path.append('..')
from tools import rotsym, sign_control

class NPJunc(HCircle):
    def __init__(self,dist,nfrags=8,n_per_frag=1,fn="output.log"):
        self.dist = dist
        self.nfrags = nfrags
        self.n_per_frag = n_per_frag
        self.fn = fn
        
    def get_mol(self,basis="sto-3g",plot=False):
        atms = [
        ["C", (-0.57367671, 0, 0.34338119)],
        ["H", (-0.59785279, 0,  1.41783945)],
        ["C", (0.59261205, 0, -0.34238682)],
        ["H", (0.57891746, 0, -1.41883382)],            
        ]

        mol = gto.Mole()
        mol.atom = atms
        mol.build()

        from dsk.las import rotsym
        n_geom = int(self.nfrags*self.n_per_frag)
        mol = rotsym.rot_trans(mol,n_geom,self.dist)
        mol.basis = basis
        mol.output = self.fn
        mol.verbose = lib.logger.INFO

        els = [atm[0] for atm in mol.atom]
        coords = np.vstack([atm[1] for atm in mol.atom])
        n = len(mol.atom)
        self.p_num = int(0*n/4)
        self.n_num = int(2*n/4)
        self.p_frag = int(self.p_num/4)
        self.n_frag = int(self.n_num/4)
        els[self.p_num] = "B"
        els[self.n_num] = "N"
        mol.atom = [(els[i],list(coords[i])) for i in range(len(els))]        
        mol.build()

        #We are going to excite on the n/4 fragment:
        self.ex_num = int(n/4)
        self.ex_frag = int(self.ex_num/4)
        return mol

    def make_las_init_guess(self):
        nfrags = self.nfrags
        mf = self.make_and_run_hf()
        mol = mf.mol

        nao_per_cell = mol.nao // nfrags
        nelec_per_cell = mol.nelectron // nfrags
        natoms_per_cell = len(mol._atom)//nfrags
        
        #LAS fragments -- (2,2)
        nao_per_frag = 2*self.n_per_frag
        nelec_per_frag = 2*self.n_per_frag
        atms_in_frag = []
        for i in range(self.n_per_frag):
            atms_in_frag += [int(0*(i+1)),int(2*(i+1))]
        natoms_per_frag = len(atms_in_frag)
        
        ref_orbs = [nao_per_frag]*(nfrags)
        ref_elec = [nelec_per_frag]*(nfrags)
        las = LASSCF(mf, ref_orbs, ref_elec)
        
        frag_atoms = [[int(i) for i in np.array(atms_in_frag) + j*natoms_per_cell] for j in range(nfrags)]
        ncas_avas,nelecas_avas,casorbs_avas = avas.AVAS(mf,["2py"]).kernel()
        assert(ncas_avas == nao_per_frag*nfrags)
        las.mo_coeff = las.localize_init_guess(frag_atoms, casorbs_avas)
        # las.mo_coeff = sign_control.fix_mos(las,verbose=True) #no sign control

        #This returns the state where the P electron has moved to B
        return las

    def make_las_state_average(self):
        las = self.make_las_init_guess()
        nfrags = self.nfrags
        
        #Lists of N Lists: Define N root spaces for LASSI
        las_charges = []
        las_spins = [] #2s, pos or neg
        
        #The reference is 2 2 2 2 2...
        base_charges = np.zeros(nfrags)
        base_spins = np.zeros(nfrags)
        las_charges += [base_charges]
        las_spins += [base_spins]
        spintyps_lst = [[1,-1],[-1,1]]
        
        #Make charge separated states:
        #Same as last time, but all now:
        for posfrag in range(nfrags):
            for negfrag in range(nfrags):
                if posfrag == negfrag:
                    continue
                for spintyps in spintyps_lst:
                    charges = base_charges.copy()
                    spins = base_spins.copy()
                    pos_spin, neg_spin = spintyps
                    charges[posfrag] = 1
                    charges[negfrag] = -1
                    spins[posfrag] = pos_spin
                    spins[negfrag] = neg_spin
                    las_charges += [charges]
                    las_spins += [spins]

        self.las_charges = np.vstack(las_charges)
        self.las_spins = np.vstack(las_spins)

        # return las_charges,las_spins
        las_charges = [list(arr.astype(int)) for arr in las_charges]
        las_smults = [list(np.abs(arr.astype(int)) + 1) for arr in las_spins]
        las_spins = [list(arr.astype(int)) for arr in las_spins]
        
        nrootspaces = len(las_charges)
        las_weights = np.ones(nrootspaces)/nrootspaces
        # return las_weights,las_charges,las_spins,las_smults
        las = las.state_average(las_weights,las_charges,las_spins,las_smults)
        las.max_cycle_macro = 100

        #Add excited state on n/4 fragment:
        lroots = np.ones([nfrags,nrootspaces])
        lroots[self.ex_frag,0] += 1 #Add the excited state here
        self.lroots = lroots.astype(int)

        return las