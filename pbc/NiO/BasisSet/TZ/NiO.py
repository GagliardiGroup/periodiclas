import os, sys
import shutil
import numpy as np
import scipy.linalg as la
from pyscf.pbc import scf, gto, df, dft, tools
from pyscf.tools import molden
from pyscf import mcpdft, lib, mcscf
from pyscf.mcscf import avas,project_init_guess
from pyscf import mcscf, dmrgscf
from functools import reduce
from pyscf import __config__
from pyscf.pbc.tools import k2gamma
from mrh.my_pyscf import mcpdft
from pyscf.tools import molden
pre_orth_method = getattr(__config__, 'pbc_scf_analyze_pre_orth_method', 'ANO')

def getCell(nU=1, spin=0, maxMem=200000,basis='def2SVP', pseudo=None):
    cell = gto.Cell()
    cell.fromfile(f"NiO_{nU}.POSCAR")
    cell.basis = basis
    cell.pseudo = pseudo
    cell.output=f'NiO_{nU}.log_'
    cell.verbose=lib.logger.INFO
    cell.precision=1e-12
    cell.exp_to_discard=0.1 # To remove the linear depandancies
    cell.spin=0
    cell.max_memory = maxMem
    cell.build()
    return cell

# Compute the integrals
def get_gdf(filename, restart=True, kpts=[[0,0,0]]):
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = filename
        gdf.build()
    return filename

def initguess(mol, mf, ao_label: list, activespacesize:int):
    '''
    Based on the ao_label find the orb which has
    highest character of that ao
    '''
    from pyscf.lo import orth
    baslst = mol.search_ao_label(ao_label)
    assert len(baslst) >=activespacesize
    orbindex=[]
    if len(mf.mo_coeff) > 1:
        mo_coeff = mf.mo_coeff[0]
    else:
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

def runSCF(cell,nU=1, Restricted=True, Unrestricted=False):
    '''
    Run the Hartree Fock
    '''
    if Restricted:
        kmf = scf.ROHF(cell).density_fit()
    elif Unrestricted:
        kmf = scf.UHF(cell).density_fit()
    else:
        raise NotImplementedError

    kmf.max_cycle=100
    kmf.chkfile = f'NiO_{nU}.chk'
    kmf.init_guess='chk'
    kmf.with_df._cderi = get_gdf(kmf.chkfile.rstrip('.chk')+'.h5', restart=False)
    kmf.exxdiv = None
    kmf.conv_tol = 1e-10
    kmf.kernel()

    if not kmf.converged:
        kmf.newton().run()

    assert kmf, "mean-field didn't converge"
    return kmf

def getfrags(nfrags: int):
    '''
    If NiO is in POSCAR Format where first N-atoms are Ni and
    next N-atoms are O, then get the frag atom no
    '''
    assert nfrags>=1, "You should know what you are doing"
    frags = []
    offset = nfrags * 2
    for i in range(nfrags):
        frag = [i * 2, i * 2 + offset, i * 2 + 1, i * 2 + offset + 1]
        frags.append(frag)
    return frags

# Running LASSCF
def runLASSCF(nfrags, cell, kmf):
    ncas = 10 # Only 3d orbitals
    nelec = 16
    norb  = tuple([ncas,]*nfrags)
    nele  = tuple([nelec,]*nfrags)
    nspin = tuple([1,]*nfrags)

    orblst = initguess(cell, kmf, ao_label=['Ni 3d', ], activespacesize=sum(norb))

    frags = getfrags(nfrags)
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    las = LASSCF(kmf, norb, nele, spin_sub=nspin)
    sortedmo = las.sort_mo(orblst, kmf.mo_coeff)
    mo0 = las.localize_init_guess(frags, sortedmo)
    molden.from_mo(cell, f'NiO.{nfrags}.las.molden', mo0[:, las.ncore:las.ncore+(ncas*nfrags)])
    las_weights,las_charges,las_spins,las_smults=genModelSpace(nfrags)
    las = las.state_average(las_weights,las_charges,las_spins,las_smults)
    elasci = las.lasci_(mo0)
    las.mo_coeff = mo0
    return las

# Model Space for the band gap
def genModelSpace(nfrags):
    las_charges = []
    las_spins = [] #2s
    las_smults = [] #2s+1

    las_charges += [[0]*nfrags]
    las_spins += [[0]*nfrags]
    las_smults += [[las_spins[0][0]+1]*nfrags]

    for i in range(nfrags):
        idxarr = np.eye(nfrags)[:,i].astype(int)
        las_charges += [list(idxarr)]
        spins = idxarr
        las_spins += [list(spins)]
        las_smults += [list(spins + 1)]

    for i in range(nfrags):
        idxarr = np.eye(nfrags)[:,i].astype(int)
        las_charges += [list(-idxarr)]
        spins = idxarr
        las_spins += [list(spins)]
        las_smults += [list(spins + 1)]

    nrootspaces = len(las_charges)
    las_weights = np.ones(nrootspaces)/nrootspaces
    return las_weights,las_charges,las_spins,las_smults

def processlas(las):
    from periodiclas.tools import sign_control, util
    las.ci = sign_control.fix_sign(las)
    las.dump_spaces()
    return las

def runLASSSI(las):
    from mrh.my_pyscf.lassi import lassi
    lsi = lassi.LASSI(las)
    energies_lassi, civecs_lassi = lsi.kernel()
    return lsi, energies_lassi

def runLASSIPDFT(lsi, states=[0]):
    from mrh.my_pyscf import mcpdft
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

if __name__ == "__main__":
    nU = int(sys.argv[1])
    nelec = 16
    cell =  getCell(nU=nU, maxMem=750000, basis='gth-tzvp-molopt', pseudo='gth-pade')
    kmf = runSCF(cell, nU=nU, Restricted=True, Unrestricted=False)
    molden.from_mo(cell, f'NiO_{nU}.molden', kmf.mo_coeff)
    las = runLASSCF(nU, cell, kmf)
    las = processlas(las)
    lsi,energies_lassi = runLASSSI(las)
    IP_LASSI, EA_LASSI = getBANDGAP(nU*nelec, lsi,energies_lassi)
    energies_tPBE = runLASSIPDFT(lsi, states=[x for x in range(len(energies_lassi))])
    energies_tPBE0 = [0.25*x+0.75*y for x, y in zip(energies_lassi, energies_tPBE)]
    IP_PDFT, EA_PDFT = getBANDGAP(nU*nelec, lsi,np.asarray(energies_tPBE))
    IP2_PDFT, EA2_PDFT = getBANDGAP(nU*nelec, lsi,np.asarray(energies_tPBE0))

    print("Results: LASSI ", )
    print("Ionization Energy: ", IP_LASSI)
    print("ElectAtt   Energy: ", EA_LASSI)
    print("Band Gap: ", IP_LASSI-EA_LASSI)

    print("Results: tPBE", )
    print("Ionization Energy: ", IP_PDFT)
    print("ElectAtt   Energy: ", EA_PDFT)
    print("Band Gap: ", IP_PDFT-EA_PDFT)

    print("Results: tPBE0", )
    print("Ionization Energy: ", IP2_PDFT)
    print("ElectAtt   Energy: ", EA2_PDFT)
    print("Band Gap: ", IP2_PDFT-EA2_PDFT)

