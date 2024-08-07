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

class PACircle(HCircle):
    def __init__(self,dist,ncells,n_per_frag=1,fn="output.log",basis="3-21g",density_fit=False):
        self.dist = dist
        self.ncells = ncells
        self.basis = basis
        self.n_per_frag = n_per_frag
        assert(ncells % n_per_frag == 0)
        self.nfrags = self.ncells // self.n_per_frag
        self.fn = fn
        self.density_fit = density_fit

    def get_mol(self,plot=False):
        atms = [
            ["C", (-0.5892731037811102, 0, 0.3262391909203111)],
            ["H", (-0.5866101957855856, 0, 1.4126530286778238)],
            ["C", (0.5916281105038108, 0, -0.3261693897255898)],
            ["H", (0.5889652025082863, 0, -1.4125832274831025)],
        ]
        
        mol = gto.Mole()
        mol.atom = atms
        mol.build()

        mol = rotsym.rot_trans(mol,self.ncells,self.dist)
        mol.basis = self.basis
        mol.output = self.fn
        mol.verbose = lib.logger.INFO
        mol.build()
        return mol

    def make_las_init_guess(self):
        nfrags = self.nfrags
        mf = self.make_and_run_hf()
        mol = mf.mol
        
        nao_per_cell = mol.nao // self.ncells
        nelec_per_cell = mol.nelectron // self.ncells
        natoms_per_cell = len(mol._atom)// self.ncells
        
        #LAS fragments -- (2,2)
        nao_per_frag = 2*self.n_per_frag
        nelec_per_frag = 2*self.n_per_frag
        atms_in_frag = []
        for i in range(self.n_per_frag):
            atms_in_frag += [np.array([0,2]) + natoms_per_cell*i]
        atms_in_frag = np.hstack(atms_in_frag)
        natoms_per_frag = len(atms_in_frag)
        
        ref_orbs = [nao_per_frag]*(nfrags)
        ref_elec = [nelec_per_frag]*(nfrags)
        # print(ref_orbs,ref_elec)
        las = LASSCF(mf, ref_orbs, ref_elec)

        # print(atms_in_frag,natoms_per_frag)
        frag_atoms = [[int(i) for i in np.array(atms_in_frag) + j*natoms_per_cell*self.n_per_frag] for j in range(nfrags)]
        ncas_avas,nelecas_avas,casorbs_avas = avas.AVAS(mf,["2py"]).kernel()
        assert(ncas_avas == nao_per_frag*nfrags)
        # print(frag_atoms)
        las.mo_coeff = las.localize_init_guess(frag_atoms, casorbs_avas)
        las.mo_coeff = sign_control.fix_mos(las,verbose=False)
        return las