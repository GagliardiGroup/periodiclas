import sys
import os
import shutil
import pickle
import numpy as np
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from pyscf import mcscf, dmrgscf, lib

nodes=1
if nodes > 1:
      dmrgscf.settings.MPIPREFIX = f'mpirun -n {nodes} --bind-to bobe'

# Get the cell
def getCell(nH, maxMem=200000, R=1.4):
    cell = pgto.Cell()
    cell.basis='STO-6G'
    cell.a = np.diag([17.5, 17.5, nH * R])
    cell.atom = [['H', np.array([0, 0, i*R]) ] for i in range(nH)]
    cell.verbose=4
    cell.precision=1e-12
    cell.output=f'Hchain.{nH}.out'
    cell.max_memory=maxMem
    cell.build()
    return cell

# DMRG Discarded weight extracter
def getDW(mc, nH):
    dmrgfile = mc.fcisolver.runtimeDir + '/dmrg.out'
    with open(dmrgfile, "r") as f:
        dws = np.array([line.strip().split("|")[-1].split("=")[-1] \
                for line in f if "DW =" in line]).astype(float)
    # Also, copy the dmrg.out to current path
    dmrgout = os.path.join(os.getcwd(), f'dmrg.{nH}.out')
    shutil.copy(dmrgfile, dmrgout)
    return dws

# Compute the integrals
def get_gdf(filename, restart=True):
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell)
        gdf._cderi_to_save = filename
        gdf.build()
    return filename

# Running the mean-field calculation
def runSCF(cell, nH):
    kmf = scf.ROHF(cell).density_fit()
    kmf.max_cycle=1000
    kmf.chkfile = f'Hchain.{nH}.chk'
    kmf.with_df._cderi = get_gdf(kmf.chkfile.rstrip('.chk')+'.h5')
    kmf.exxdiv=None
    kmf.conv_tol=1e-12
    kmf.kernel()

    if not kmf.converged:
        kmf.newton().run()

    assert kmf, "mean-field didn't converge"
    return kmf

# For LAS Guess Orbitals
def genInitGuess(nH, kmf):
    nHfrag = int(nH/2) # TB consistent with the molecular calculation.
    norb = tuple([nHfrag,]*2)
    nele = tuple([nHfrag,]*2)
    nspin = tuple([1,]*2)
    frags =tuple([list(range(nH))[i:i + nHfrag] \
            for i in range(0,nH, nHfrag)])
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    las = LASSCF(kmf, norb, nele, spin_sub=nspin)
    mo0 = las.localize_init_guess(frags,kmf.mo_coeff)
    return mo0

# DMRGsolver
def get_dmrgsolver(mol, spin=None, Mvalue=500):
    solver = dmrgscf.DMRGCI(mol, maxM=Mvalue)
    solver.memory = int(mol.max_memory/1000)
    solver.nroots = 1
    solver.scratchDirectory = lib.param.TMPDIR
    solver.runtimeDir = lib.param.TMPDIR
    solver.threads = lib.num_threads()
    solver._restart = False
    if spin is not None:
        solver.spin = spin
    return solver

def runDMRG(nH, nelec, cell, kmf, mo_coeff=None, Mvalue=500):
    if mo_coeff is None:
        mo0 = genInitGuess(nH, kmf)
    else:
        mo0 = mo_coeff
    mc = mcscf.CASCI(kmf, nH, nelec, ncore=0)
    solver = get_dmrgsolver(cell, spin=nelec%2, Mvalue=Mvalue)
    mc.fcisolver = solver
    Energy  = mc.kernel(mo0)[0]
    DW = getDW(mc, nelec)
    print("DW", DW)
    return Energy, DW[-1]

if __name__ == "__main__":

    nH = int(sys.argv[1]) # No of H-atoms
    assert nH%2==0
    R = 1.4 # Distance between the H-atoms

    cell =  getCell(nH, maxMem=500000, R=R)
    kmf = runSCF(cell, nH)
    from pyscf import lo, mcscf

    lmo_occ = lo.PM(cell, kmf.mo_coeff[:, :int(nH/2)]).kernel()
    lmo_vir = lo.PM(cell, kmf.mo_coeff[:, int(nH/2):]).kernel()

    lmo = np.hstack((lmo_occ, lmo_vir)) 

    Data={}

    for nelec in [nH+1]:
        with lib.temporary_env(kmf):
            E, DW = runDMRG(nH, nelec, cell, kmf, mo_coeff=lmo, Mvalue=3000)
            Data[nelec] = [E, DW]
    
    with open(f"hchain.DMRG.{nH}.pkl", "wb") as f:
        pickle.dump(Data, f)

