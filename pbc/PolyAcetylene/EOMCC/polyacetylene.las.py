import os
import sys
import pickle
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import lassi
from mrh.my_pyscf import mcpdft

def get_xyz(nU=1, d= 2.47):
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0000000000),
    ("H", -0.5866101958,  1.4126530287,  0.0000000000),
    ("C",  0.5916281105, -0.3261693897,  0.0000000000),
    ("H",  0.5889652025, -1.4125832275,  0.0000000000)]

    translated_coords = []
    for t in range(nU):
        shift = t * d
        translated_coords.extend([(elem, x + shift, y, z) 
            for elem, x, y, z in coords])
    return translated_coords


def getCell(nC, nkpts=1, d=2.47, maxMem=500000, basis='321G', pseudo=None):
    """
    Build the Cell object
    """
    cell = pgto.Cell()
    cell.atom = get_xyz(nU=1, d=d)
    cell.a = np.diag([2.47, 17.5, 17.5])
    cell.basis = basis
    cell.pseudo = pseudo
    cell.precision=1e-12
    cell.verbose = lib.logger.INFO
    cell.max_memory = maxMem
    cell.output = f"PAChain.{2}.{nkpts}.log"
    cell.build()
    return cell

def runSCF(cell, nC, nkpts):
    """
    Mean-Field Calculation
    """
    nmp = [nkpts, 1, 1]
    kpts = cell.make_kpts(nmp)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.max_cycle=1000
    kmf.chkfile = f'PAchain.{2}.{nkpts}.chk'
    kmf.exxdiv = None
    kmf.conv_tol = 1e-12
    kmf.kernel()

    if not kmf.converged:
        kmf.newton().run()

    assert kmf, "mean-field didn't converge"
    return kmf


def runCCSDBands(cell, kmf, nkpts):
    from pyscf.pbc import gto, cc
    from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
    kpts = cell.make_kpts([nkpts, 1, 1])
    mycc = cc.KRCCSD(kmf)
    ekrcc = mycc.kernel()[0]
    eomip = mycc.ipccsd(nroots=1)[0]
    eomea = mycc.eaccsd(nroots=1)[0]
    return ekrcc, eomip, eomea, kmf.kpts

def savepickel(mf, lsi, pdftenergy=0, nCfrag=2, nC=2, R=2.47):
    """
    Save the LAS Band Structure Data
    """

    civec = lsi.get_sivec_vacuum_shuffle(state=0)
    nfrags = int(nC/nCfrag)
    charges = util.las_charges(lsi._las)

    data = {"energies_lassi":lsi.e_roots,
            "energies_lassipdft":pdftenergy,
            "civecs":civec,
            "charges":charges,
            "nfrags":nfrags,
            "dist":R,
            "mf_coeff":mf.mo_coeff,
            "mf_occ":mf.mo_occ,
            "mf_ene":mf.mo_energy}

if __name__ == "__main__":

    nC = int(sys.argv[1]) # No of CH-atoms
    nCfrag = min(2, nC)
    nfrags = int(nC/nCfrag)
    assert nC%2==0
    d = 2.47

    for nkpts in [1, 2, 4, 8, 16]:

        cell =  getCell(nC, nkpts=nkpts, d=d, maxMem=900000, basis='321G')

        kmf = runSCF(cell, nC, nkpts)

        e, eip, ea, k = runCCSDBands(cell, kmf, nkpts)

        data = {
        'eip': eip,
        'ea': ea,
        'k': k}

        with open(f"PAChain.{2}.{nkpts}.pkl", "wb") as f:
            pickle.dump(data, f)

        print("Results:")
        print("Ionization Energy: {:.2f}".format((max(eip)[0])))
        print("ElectAtt Energy: {:.2f}".format((min(ea)[0])))
        print("Band Gap: {:.2f}".format((max(eip) + min(ea))[0]))
