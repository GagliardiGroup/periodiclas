import numpy as np
import pandas as pd
from pyscf import gto, scf, lib, mcscf
import math
import os
from .hcircle import HCircle

class PACircle(HCircle):
    def __init__(self,dist,nfrags,n_per_frag=2,fn="output.log"):
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
        mol.build()
        return mol

    def make_las_init_guess(self):
        from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
        from mrh.my_pyscf.lassi import lassi
        from dsk.las import sign_control
        from pyscf.mcscf import avas

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
            atms_in_frag += [np.array([0,2]) + 4*i]
        atms_in_frag = np.hstack(atms_in_frag)
        natoms_per_frag = len(atms_in_frag)
        
        ref_orbs = [nao_per_frag]*(nfrags)
        ref_elec = [nelec_per_frag]*(nfrags)
        las = LASSCF(mf, ref_orbs, ref_elec)
        
        frag_atoms = [[int(i) for i in np.array(atms_in_frag) + j*natoms_per_cell] for j in range(nfrags)]
        ncas_avas,nelecas_avas,casorbs_avas = avas.AVAS(mf,["2py"]).kernel()
        assert(ncas_avas == nao_per_frag*nfrags)
        las.mo_coeff = las.localize_init_guess(frag_atoms, casorbs_avas)
        las.mo_coeff = sign_control.fix_mos(las,verbose=True)
        return las