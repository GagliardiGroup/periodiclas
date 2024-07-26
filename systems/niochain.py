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

class NiOChain(HCircle):
    def __init__(self,nfrags,dist=2,nio_dist=2.09,fn="output.log"):
        self.nfrags = nfrags
        self.dist = dist
        self.nio_dist = nio_dist
        self.fn = fn

    def get_mol(self,basis="minao4s",plot=False):
        #To run this you will need to put minao4s.py in your gto/basis dir
        mol = gto.Mole()
        mol.atom = atms = [
        ['Ni', (0, 0, self.nio_dist/2)],
        ['O', (0, 0, -self.nio_dist/2)],
        ]
        
        mol.build()
        mol.basis = basis
        mol.output = self.fn
        mol.verbose = lib.logger.INFO
        mol.build()
        
        atms = mol.atom
        els = [atm[0] for atm in atms]
        coords = [atm[1] for atm in atms]
        coords = np.array(coords)

        ogcoords = coords.copy()
        ogels = els.copy()
        dvec = np.array([1,0,0])
        for i in range(1,self.nfrags):
            coords2 = ogcoords.copy()
            coords2 += dvec * i * self.dist
            coords = np.vstack([coords,coords2])
            els += ogels
        coords = coords - coords.mean(axis=0)
        atms2 = []
        for i in range(len(els)):
            atms2 += [[els[i],tuple(coords[i])]]
        mol.atom = atms2
        mol.build()

        return mol

    def make_las_init_guess(self):
        nfrags = self.nfrags
        mf = self.make_and_run_hf()
        mol = mf.mol

        nao_per_cell = mol.nao // nfrags
        nelec_per_cell = mol.nelectron // nfrags
        natoms_per_cell = len(mol._atom)//nfrags
        
        #LAS fragments -- (2,2)
        nao_per_frag = 9
        nelec_per_frag = 16
        atms_in_frag = [0,1] # change this?
        natoms_per_frag = len(atms_in_frag)
        
        ref_orbs = [nao_per_frag]*(nfrags)
        ref_elec = [nelec_per_frag]*(nfrags)
        las = LASSCF(mf, ref_orbs, ref_elec)
        
        frag_atoms = [[int(i) for i in np.array(atms_in_frag) + j*natoms_per_cell] for j in range(nfrags)]
        las.mo_coeff = las.localize_init_guess(frag_atoms, mf.mo_coeff)
        return las

    def make_las_state_average(self):
        las = self.make_las_init_guess()
        nfrags = self.nfrags
        
        #Lists of N Lists: Define N root spaces for LASSI
        las_charges = []
        las_spins = [] #2s, pos or neg
        
        #The reference is 2 2 2 2 2
        base_charges = np.zeros(nfrags)
        base_spins = np.zeros(nfrags)
        las_charges += [base_charges]
        las_spins += [base_spins]
        spintyps_lst = [[1,-1],[-1,1]]
        
        #Make electron states:
        posfrag = self.nfrags//2 - 1
        negfrag = posfrag + 1
        for spintyp in spintyps_lst:
            pos_spin, neg_spin = spintyp
            charges = base_charges.copy()
            spins = base_spins.copy()
            charges[negfrag] = -1
            charges[posfrag] = 1
            spins[negfrag] = neg_spin
            spins[posfrag] = pos_spin
            las_charges += [charges]
            las_spins += [spins]

        self.las_charges = np.vstack(las_charges)
        self.las_spins = np.vstack(las_spins)

        # return las_charges,las_spins
        las_charges = [list(arr.astype(int)) for arr in self.las_charges]
        las_smults = [list(np.abs(arr.astype(int)) + 1) for arr in self.las_spins]
        las_spins = [list(arr.astype(int)) for arr in self.las_spins]
        
        nrootspaces = len(las_charges)
        las_weights = np.ones(nrootspaces)/nrootspaces
        # return las_weights,las_charges,las_spins,las_smults
        las = las.state_average(las_weights,las_charges,las_spins,las_smults)
        las.max_cycle_macro = 50
        
        return las