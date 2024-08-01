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

class EthChain_CT(HCircle):
    def __init__(self,nfrags,charge=1,dist=2.5,c_dist=1.4,fn="output.log",basis="sto3g"):
        #charge = 1 --> hole transport
        #charge = -1 --> electron transport
        self.nfrags = nfrags
        self.dist = dist
        self.basis = basis
        self.c_dist = c_dist
        self.fn = fn
        self.charge = charge

    def get_mol(self,basis="sto-3g",plot=False):
        mol = gto.Mole()
        mol.atom = atms = [
        ['C', (0, 0,  self.c_dist/2)],
        ['C', (0, 0, -self.c_dist/2)],
        ['H', (0, 9.27056314e-01,  1.20411032e+00)],
        ['H', (0, -9.26212677e-01,  1.20585179e+00)],
        ['H', (0, 9.26243081e-01, -1.20544427e+00)],
        ['H', (0, -9.26975035e-01, -1.20466940e+00)],
        ]
        
        mol.build()
        mol.basis = self.basis
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
        nao_per_frag = 2
        nelec_per_frag = 2
        atms_in_frag = [0,1]
        natoms_per_frag = len(atms_in_frag)
        
        ref_orbs = [nao_per_frag]*(nfrags)
        ref_elec = [nelec_per_frag]*(nfrags)
        las = LASSCF(mf, ref_orbs, ref_elec)
        
        frag_atoms = [[int(i) for i in np.array(atms_in_frag) + j*natoms_per_cell] for j in range(nfrags)]
        ncas_avas,nelecas_avas,casorbs_avas = avas.AVAS(mf,["C 2px"]).kernel()
        assert(ncas_avas == nao_per_frag*nfrags)
        las.mo_coeff = las.localize_init_guess(frag_atoms, casorbs_avas)
        las.mo_coeff = sign_control.fix_mos(las,verbose=False)
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
        for i in range(nfrags):
            charges = base_charges.copy()
            spins = base_spins.copy()
            charges[i] = self.charge
            spins[i] = 1
            las_charges += [charges]
            las_spins += [spins]

        self.las_charges = np.vstack(las_charges)
        self.las_spins = np.vstack(las_spins)

        las_charges = [list(arr.astype(int)) for arr in self.las_charges]
        las_smults = [list(np.abs(arr.astype(int)) + 1) for arr in self.las_spins]
        las_spins = [list(arr.astype(int)) for arr in self.las_spins]
        
        nrootspaces = len(las_charges)
        las_weights = np.ones(nrootspaces)/nrootspaces
        las = las.state_average(las_weights,las_charges,las_spins,las_smults)
        las.max_cycle_macro = 100
        
        return las

class EthChain_ET(EthChain_CT):
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

        self.las_charges = np.vstack(las_charges)
        self.las_spins = np.vstack(las_spins)

        las_charges = [list(arr.astype(int)) for arr in self.las_charges]
        las_smults = [list(np.abs(arr.astype(int)) + 1) for arr in self.las_spins]
        las_spins = [list(arr.astype(int)) for arr in self.las_spins]
        
        nrootspaces = len(las_charges)
        las_weights = np.ones(nrootspaces)/nrootspaces
        las = las.state_average(las_weights,las_charges,las_spins,las_smults)
        las.max_cycle_macro = 100

        lroots = np.ones([nfrags,nrootspaces])*2 #excited states
        self.lroots = lroots.astype(int)
        
        return las