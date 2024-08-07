import numpy as np
from pyscf import gto, scf, lib, mcscf
import math
import time
import os
from pyscf import dmrgscf, lib
from .hcircle import HCircle
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
dmrgscf.settings.BLOCKSCRATCHDIR = os.path.join(lib.param.TMPDIR, str(os.getpid()))
dmrgscf.settings.MPIPREFIX = "" 

class HDMRG(HCircle):
    def __init__(self,num_h,dist=2.5,mval=500,pdft=True,fn=None):
        self.data_name = f"hcircle{num_h}_d{int(dist*10)}_m{mval}"
        if not fn:
            super().__init__(dist,num_h,2,fn=f"{self.data_name}.log")
        else:
            super().__init__(dist,num_h,2,fn=fn)
        self.mval = mval
        self.pdft = pdft
        self.rundir = f"{self.data_name}/"
        self.density_fit = False

    def make_casci(self,charge):
        mol = self.get_mol()
        mol.charge = charge
        mol.spin = mol.nelectron%2
        mol.build()

        las = self.make_las_init_guess()
        ncas = las.ncas
        nelecas = sum(las.nelecas) - mol.charge
        twos = mol.spin
        if self.pdft:
            from pyscf import mcpdft
            mc = mcpdft.CASCI(mol,"tPBE",ncas,nelecas)
        else:
            mc = mcscf.CASCI(mol,ncas,nelecas)
        mc.mo_coeff = las.mo_coeff
        return mc

    def make_dmrg(self,charge):
        scrdir = f"/scratch/midway3/king1305/{self.data_name}_charge{charge}"
        if os.path.isdir(scrdir):
            os.system(f"rm -r {scrdir}")
        mc = self.make_casci(charge)
        mc.fcisolver = dmrgscf.DMRGCI(mc.mol,maxM=self.mval)
        mc.fcisolver.threads = lib.num_threads()
        mc.fcisolver.spin = mc.mol.spin # setting to default spin here
        mc.fcisolver.memory = int(mc.mol.max_memory / 1000)
        mc.fcisolver.runtimeDir = f"{self.rundir}charge{charge}/"
        mc.fcisolver.scratchDirectory = scrdir
        mc.fcisolver.block_extra_keyword = ["onepdm"]
        mc.verbose = 4
        mc.canonicalization = False
        return mc

    def dryrun(self,charge):
        mc = self.make_dmrg(charge)
        dmrgscf.dryrun(mc)
        cwd = os.getcwd()
        print(f"Running DMRG (charge {charge})...")
        os.chdir(mc.fcisolver.runtimeDir)
        os.system("block2main dmrg.conf > dmrg.out")
        os.chdir(cwd)
        return self.get_energy(mc)

    def get_energy(self,mc):
        fn = mc.fcisolver.runtimeDir
        fn = f"{fn}/dmrg.out"
        with open(fn,"r") as f:
            lines = f.readlines()
        dws = []
        for line in lines:
            if "DMRG Energy =" in line:
                e = line.strip().split("=")[-1]
            if "DW =" in line:
                dws += [line.strip().split("|")[-1].split("=")[-1]]
        e = float(e)
        dws = np.array(dws).astype(float)
        return e, dws[-1]

    def run_dmrg(self,charge):
        mc = self.make_dmrg(charge)
        mc.kernel()
        e, dw = self.get_energy(mc)
        results = {}
        results["dw"] = dw
        if not self.pdft:
            results["e_mcscf"] = e
        else:
            results["e_mcscf"] = mc.e_mcscf
            results["e_mcpdft"] = mc.e_tot
        return results