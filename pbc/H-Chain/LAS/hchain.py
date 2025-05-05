import os
import sys
import pickle
import numpy as np
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCFNoSymm as LASSCF
from mrh.my_pyscf.lassi import lassi
from periodiclas.tools import sign_control, util

def getCell(nH, nHFrag, maxMem=200000, R=1.4):
    """
    Build the Cell object
    """
    cell = pgto.Cell()
    cell.basis='STO-6G'
    cell.a = np.diag([17.5, 17.5, nH * R])
    cell.atom = [['H', np.array([0, 0, i*R]) ] for i in range(nH)]
    cell.verbose = lib.logger.INFO
    cell.output = f"HChain.{nHFrag}.{nH}.log"
    cell.max_memory = maxMem
    cell.precision = 1e-12
    cell.build()
    return cell

def get_gdf(filename, restart=True):
    """
    Calculate the 2e Integrals
    Using the Gaussian Density Fitting.
    """
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell)
        gdf._cderi_to_save = filename
        gdf.build()
    return filename

def runSCF(cell, nH):
    """
    Mean-Field Calculation
    """
    kmf = scf.ROHF(cell).density_fit()
    kmf.max_cycle=1000
    kmf.chkfile = f'Hchain.{nH}.chk'
    kmf.with_df._cderi = get_gdf(kmf.chkfile.rstrip('.chk')+'.h5')
    kmf.exxdiv = None
    kmf.conv_tol = 1e-12
    kmf.kernel()

    if not kmf.converged:
        kmf.newton().run()

    assert kmf, "mean-field didn't converge"
    return kmf

def genModelSpace(nfrags):
    """
    Model Space Creation for LAS Band Structure
    """
    identity = np.eye(nfrags, dtype=int)
    las_charges = [[0] * nfrags] + identity.tolist() + (-identity).tolist()
    las_spins = [[0] * nfrags] + identity.tolist() + identity.tolist()
    las_smults = [[1] * nfrags] + (identity + 1).tolist() + (identity + 1).tolist()
                    
    nrootspaces = len(las_charges)
    las_weights = np.full(nrootspaces, 1/nrootspaces)
                                
    return las_weights, las_charges, las_spins, las_smults

# Running LASSCF
def runLASSCF(nHfrag, nfrags, kmf):
    """
    Optimize the Individual Fragment LAS
    """

    # Active Space
    norb  = tuple([nHfrag,]*nfrags)
    nele  = tuple([nHfrag,]*nfrags)
    nspin = tuple([1,]*nfrags)
    
    # Fragmentation
    frags =tuple([list(range(nH))[i:i + nHfrag] \
            for i in range(0,nH, nHfrag)])
    
    las = LASSCF(kmf, norb, nele, spin_sub=nspin)
    las.mo_coeff = las.localize_init_guess(frags,kmf.mo_coeff)
    las.mo_coeff = sign_control.fix_mos(las)
    las_weights,las_charges,las_spins,las_smults=genModelSpace(nfrags)
    las = las.state_average(las_weights,las_charges,las_spins,las_smults)
    las.lasci_()
    return las

def processlas(las):
    """
    Sign-Fixing for the LAS CI Vectos
    """
    las.ci = sign_control.fix_sign(las)
    las.dump_spaces()
    return las

def runLASSI(las):
    """
    LAS State Interaction
    """
    lsi = lassi.LASSI(las)
    energies_lassi, civecs_lassi = lsi.kernel()
    return lsi, energies_lassi

def savepickel(mf, lsi, R=1.4,  pdftenergy=0, nHfrag=2, nH=2):
    """
    Save the LAS Band Structure Data
    """

    civec = lsi.get_sivec_vacuum_shuffle(state=0)
    nfrags = int(nH/nHfrag)
    charges = util.las_charges(lsi._las)

    data = {"energies_lassi":lsi.e_roots,
            "civecs":civec,
            "charges":charges,
            "nfrags":nfrags,
            "dist":R,
            "mf_coeff":mf.mo_coeff,
            "mf_occ":mf.mo_occ,
            "mf_ene":mf.mo_energy}

    with open(f"hchain.{nHfrag}.{nH}.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":

    for nH in [8, 16, 32, 64]:
        for i in [2, 4, 8]:
            nHfrag = min(i, nH)
            nfrags = int(nH/nHfrag)
            R = 1.4 # Distance between the H-atoms

            cell =  getCell(nH, nHfrag, maxMem=200000, R=R)
            kmf = runSCF(cell, nH)

            las = runLASSCF(nHfrag, nfrags, kmf)
            las = processlas(las)
            lsi, energies_lassi = runLASSI(las)
            
            # Save data
            savepickel(kmf, lsi, R=R, nHfrag=nHfrag, nH=nH)
    
