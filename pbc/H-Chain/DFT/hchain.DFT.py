import sys
import os
import pickle
import numpy as np
import pandas as pd
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from pyscf import mcscf, dmrgscf, lib


def getCell(nH, XC='PBE', basis='STO-6G',maxMem=200000, R=1.4):
    """
    Build the Cell object
    """
    cell = pgto.Cell()
    cell.basis=basis
    cell.a = np.diag([nH*R, 17.479, 17.479])
    cell.atom = [['H', np.array([i*R, 0, 0]) ] for i in range(nH)]
    cell.verbose = lib.logger.INFO
    cell.output = f"HChain.{XC}.{nH}.log"
    cell.max_memory = maxMem
    cell.precision = 1e-12
    cell.build()
    return cell

def runDFT(cell, nH, nkpts, XC='PBE'):
    """
    Mean-Field Calculation
    """
    if not XC == 'HF':
        kmf = scf.KRKS(cell).density_fit()
        kmf.xc = XC
    else:
        kmf = scf.KRHF(cell).density_fit()

    kmf.max_cycle=50
    kmf.kpts = cell.make_kpts([nkpts,1, 1])
    kmf.chkfile = f'Hchain.{XC}.{nH}.chk'
    kmf.conv_tol = 1e-12
    kmf.exxdiv = None
    kmf.kernel()

    if not kmf.converged:
        kmf.newton().run()

    assert kmf, "mean-field didn't converge"
    return kmf


def get_bands(cell, kmf, nH, nkpts, R=1.4):
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
    def __init__(self, nH, XC, df):
        self.df = df
        self.nH = nH
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
        nH = self.nH
        XC = self.XC
        homo_e, homo_k = self.get_homo()
        lumo_e, lumo_k = self.get_lumo()
        data = {'nH':nH,
                'XC':XC,
                'homo_e':homo_e,
                'homo_k':homo_k,
                'lumo_e':lumo_e,
                'lumo_k':lumo_k, 
                'bandgap': -lumo_e.min() + homo_e.max()
                }

        with open(f"hchain.{XC}.{nH}.pkl", "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    for nH in [2, 4, ]:
        for XC in ['HF', 'PBE', 'PBE0', 'SCAN']:
            R = 1.4 # Distance between the H-atoms
            nkpts = int(32/nH)
            cell =  getCell(nH, XC, maxMem=350000, R=R)
            kmf = runDFT(cell, nH, nkpts, XC=XC)
            data = get_bands(cell, kmf, nH, nkpts, R=R)
            
            # Save the data
            PeriodicDataAnalysis(nH, XC, data).savepickel()
            
