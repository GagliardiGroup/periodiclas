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
from periodiclas.tools import sign_control, util

def get_xyz(nU=1, d= 2.47):
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0),
    ("H", -0.5866101958,  1.4126530287,  0.0),
    ("C",  0.5916281105, -0.3261693897,  0.0),
    ("H",  0.5889652025, -1.4125832275,  0.0)]
    
    nU *=4
    
    translated_coords = []
    for t in range(nU):
        shift = t * d
        translated_coords.extend([(elem, x + shift, y, z) 
            for elem, x, y, z in coords])
    return translated_coords


def getCell(nC, nU=1, d=2.47, maxMem=500000, basis='321G', pseudo=None):
    """
    Build the Cell object
    """
    cell = pgto.Cell()
    cell.atom = get_xyz(nU, d)
    cell.a = np.diag([2.47*4*nU, 17.5, 17.5])
    cell.basis = basis
    cell.pseudo = pseudo
    cell.precision=1e-12
    cell.verbose = lib.logger.INFO
    cell.max_memory = maxMem
    cell.output = f"PAChain.{8}.{nC}.log"
    cell.build()
    return cell

def initguess(mol, mf, ao_label: list, activespacesize:int):
    '''
    Based on the ao_label find the orb which has
    highest character of that ao
    '''
    from pyscf.lo import orth
    baslst = mol.search_ao_label(ao_label)
    assert len(baslst) >=activespacesize
    orbindex=[]

    mo_coeff = mf.mo_coeff

    nkpts, nao = 1, mf.mo_coeff.shape[1]
    s_sc = mf.get_ovlp()
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin',pre_orth_ao=None, s=s_sc)
    C = reduce(np.dot,(orth_coeff.T.conj(), s_sc, mf.mo_coeff))
    for orb in baslst:
        ao = C[orb]
        A = np.argsort(ao*ao.conj())[-activespacesize:][::-1]
        for element in A:
            if element not in orbindex:
                orbindex.append(element)
                break
    orbind = [x+1 for x in orbindex]
    return sorted(orbind[:activespacesize])

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


def runSCF(cell, nC):
    """
    Mean-Field Calculation
    """
    kmf = scf.ROHF(cell).density_fit()
    kmf.max_cycle=1000
    kmf.chkfile = f'PAchain.{nC}.chk'
    kmf.with_df._cderi = get_gdf(kmf.chkfile.rstrip('.chk')+'.h5') #, restart=False) 
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


def runLASSCF(nCfrag, nfrags, cell, kmf):
    """
    Optimize the Individual Fragment LAS
    """
    nC = nCfrag*nfrags

    # Active Space
    norb  = tuple([nCfrag,]*nfrags)
    nele  = tuple([nCfrag,]*nfrags)
    nspin = tuple([1,]*nfrags)
    
    # Fragmentation
    frags = [[x*2 for x in range(nC)][i:i + nCfrag] for i in range(0, len([x*2 for x in range(nC)]), nCfrag)]

    las = LASSCF(kmf, norb, nele, spin_sub=nspin)
    orblst = initguess(cell, kmf, ['C 2pz', 'C 3pz'], 2*nC)[:nC]
    sortedmo = las.sort_mo(orblst, kmf.mo_coeff)
    las.mo_coeff = las.localize_init_guess(frags,sortedmo)
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

def runLASSIPDFT(lsi, states=[0]):
    mc = mcpdft.LASSI(lsi, 'tPBE', states=states)
    energies = mc.kernel()[0]
    return energies

def getBANDGAP(nele, lsi, energies_lassi):
    stateidx = [
        np.where((np.asarray(lsi.rootsym)[:, :2].sum(axis=1) == nele))[0],
        np.where((np.asarray(lsi.rootsym)[:, :2].sum(axis=1) == nele - 1))[0],
        np.where((np.asarray(lsi.rootsym)[:, :2].sum(axis=1) == nele + 1))[0]]
    stateidx = [list(x) for x in stateidx]
    ip = 27.21139*(min(energies_lassi[stateidx[1]]) - min(energies_lassi[stateidx[0]]))
    ea = 27.21139*(min(energies_lassi[stateidx[0]]) - min(energies_lassi[stateidx[2]]))
    return ip, ea

def getNatorbOcc(nele, lsi):
    from mrh.my_pyscf.lassi.lassi import root_make_rdm12s
    stateidx = [
        np.where((np.asarray(lsi.rootsym)[:, :2].sum(axis=1) == nele))[0],
        np.where((np.asarray(lsi.rootsym)[:, :2].sum(axis=1) == nele - 1))[0],
        np.where((np.asarray(lsi.rootsym)[:, :2].sum(axis=1) == nele + 1))[0]]
    stateidx = [int(list(x)[0]) for x in stateidx]
    for state in stateidx:
        natorb_casdm1 = root_make_rdm12s (lsi, lsi.ci, lsi.si, state=state, opt=lsi.opt)[0].sum (0)
        mo_occ = lsi._las.canonicalize (natorb_casdm1=natorb_casdm1)[2]
        mo_occ = [x for x in mo_occ if 0 < x < 2]
        print("State-",state, mo_occ)

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

    with open(f"PAChain.{nCfrag}.{nC}.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":

    nC = int(sys.argv[1]) # No of CH-atoms
    nCfrag = min(8, nC)
    nfrags = int(nC/nCfrag)
    assert nC%2==0
    d = 2.47

    cell =  getCell(nC, nU=nfrags, d=d, maxMem=950000, basis='321G')

    kmf = runSCF(cell, nC)

    las = runLASSCF(nCfrag, nfrags, cell, kmf)
    las = processlas(las)

    lsi,energies_lassi = runLASSI(las)
    IP_LASSI, EA_LASSI = getBANDGAP(nC, lsi, energies_lassi)

    energies_mcpdft = runLASSIPDFT(lsi, states=[x for x in range(len(energies_lassi))])
    IP_PDFT, EA_PDFT = getBANDGAP(nC, lsi,np.asarray(energies_mcpdft))
    
    print("Results: LASSI ", )
    print("Ionization Energy: ", IP_LASSI)
    print("ElectAtt   Energy: ", EA_LASSI)
    print("Band Gap: ", IP_LASSI-EA_LASSI)

    print("Results: PDFT", )
    print("Ionization Energy: ", IP_PDFT)
    print("ElectAtt   Energy: ", EA_PDFT)
    print("Band Gap: ", IP_PDFT-EA_PDFT)
    
    savepickel(kmf, lsi, pdftenergy=energies_mcpdft, nCfrag=nCfrag, nC=nC, R=2.47)

