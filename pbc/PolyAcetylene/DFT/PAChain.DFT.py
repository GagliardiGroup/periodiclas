import sys
import os
import pickle
import numpy as np
import pandas as pd
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from pyscf import mcscf, dmrgscf, lib

def get_xyz(nU=1, d= 2.47, nR=1):
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0),
    ("H", -0.5866101958,  1.4126530287,  0.0),
    ("C",  0.5916281105, -0.3261693897,  0.0),
    ("H",  0.5889652025, -1.4125832275,  0.0)]

    nU *=nR

    translated_coords = []
    for t in range(int(nU)):
        shift = t * d
        translated_coords.extend([(elem, x + shift, y, z)
            for elem, x, y, z in coords])
    return translated_coords


def getCell(nC, nU=1, d=2.47, XC='PBE', maxMem=500000, basis='321G',pseudo=None):
    """
    Build the Cell object
    """
    nR = nC/2
    cell = pgto.Cell()
    cell.atom = get_xyz(nU, d, nR=nR)
    cell.a = np.diag([2.47*nR*nU, 17.5, 17.5])
    cell.basis = basis
    cell.pseudo = pseudo
    cell.dimension = 1
    cell.precision=1e-12
    cell.verbose = lib.logger.INFO
    cell.max_memory = maxMem
    cell.output = f"PAChain.{XC}.{nC}.log"
    cell.build()
    return cell

def get_gdf(cell, filename, kpts=[1,1,1], restart=True):
    """
    Calculate the 2e Integrals
    Using the Gaussian Density Fitting.
    """
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell, kpts=kpts)
        gdf._cderi_to_save = filename
        gdf._j_only = False
        gdf.build()
    return filename

def runDFT(cell, nC, nkpts, XC='PBE'):
    """
    Mean-Field Calculation
    """

    if not XC == 'HF':
        from pyscf.pbc import dft
        kmf = dft.KRKS(cell, xc=XC).density_fit()
    else:
        kmf = scf.KRHF(cell).density_fit()

    kmf.max_cycle=50
    kmf.kpts = cell.make_kpts([nkpts, 1,1])
    kmf.chkfile = f'PAChain.{XC}.{nC}.chk'
    kmf.with_df._cderi = get_gdf(cell, kmf.chkfile.rstrip('.chk')+'.h5', kpts = kmf.kpts)
    kmf.exxdiv = None
    kmf.conv_tol = 1e-12
    kmf.kernel()

    if not kmf.converged:
        kmf.newton().run()

    assert kmf, "mean-field didn't converge"
    return kmf


def get_bands(cell, kmf, nkpts, R=2.47):
    '''
    For the converged kmf, get the band structure at kpts
    '''
    kpts = kmf.kpts
    energies, mos = kmf.get_bands(kpts)
    energies = np.vstack(energies)
    kptsnorm = np.arange(nkpts)/nkpts 

    kptsnorm = np.hstack([kptsnorm, np.array(1)]) #append gamma at end

    df = pd.DataFrame()
    for i in range(energies.shape[1]):
        e_band = energies[:,i]
        e_band = np.hstack([e_band,np.array(e_band[0])])
        for k, e in zip(kptsnorm,e_band):
            df.loc[k,i] = e
        df.loc["nocc",i] = kmf.mo_occ[0][i]
    return df

class PeriodicDataAnalysis:
    '''
    Some functions to analyze the periodic DFT data
    '''
    def __init__(self, nC, XC, df):
        self.df = df
        self.nC = nC
        self.XC = XC
        self.mo_occ = self.df.loc["nocc"]
        self.df = self.df.drop("nocc")
        self.hartree_to_ev = 27.21139

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

    def savepickel(self):
        """
        Save the LAS Band Structure Data
        """
        nC = self.nC
        XC = self.XC
        homo_e, homo_k = self.get_homo()
        lumo_e, lumo_k = self.get_lumo()
        data = {'nC':nC,
                'XC':XC,
                'homo_e':homo_e,
                'homo_k':homo_k,
                'lumo_e':lumo_e,
                'lumo_k':lumo_k, 
                'bandgap': -lumo_e.min() + homo_e.max()
                }

        with open(f"hchain.{XC}.{nC}.pkl", "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    nC = 8
    for XC in ['PBE0', ]: 
        d = 2.47
        nkpts = 16
        cell =  getCell(nC, nU=1, d=d, XC=XC, maxMem=950000, basis='3-21G')
        kmf = runDFT(cell, nC, nkpts, XC=XC)
        data = get_bands(cell, kmf, nkpts, R=d)
        
        # Save the data
        PeriodicDataAnalysis(nC, XC, data).savepickel()
