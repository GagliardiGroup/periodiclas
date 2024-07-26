import numpy as np
from pyscf import gto, scf, lib, mcscf
import math

def polygon(sides,dist):
    rnum = dist**2
    rdenom = 2*(1-np.cos(2*np.pi/sides))
    radius = np.sqrt(rnum/rdenom)
    
    one_segment = np.pi * 2 / sides

    points = [
        (math.sin(one_segment * i) * radius,
         math.cos(one_segment * i) * radius)
        for i in range(sides)]

    return points

def rot(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def trans_rot(coords,point,theta):
    coords = np.array(coords.copy())
    avg = np.mean(coords,axis=0)
    coords = coords - avg
    yval = coords[:,1]
    coords = np.array(coords)[:,[0,2]]
    rotmat = rot(theta)
    coords = np.matmul(rotmat,coords.T).T
    coords += np.array(point).T
    coords = np.vstack([coords[:,0],yval,coords[:,1]]).T
    return coords

def rot_trans(mol,ncells,dist):
    lib.param.BOHR
    els = [atm[0] for atm in mol._atom]
    xyzs = [np.array(atm[1])*lib.param.BOHR for atm in mol._atom]
    points = polygon(ncells,dist)
    newatms = []
    for i in range(ncells):
        coords = trans_rot(xyzs,points[i],-i*2*np.pi/ncells)
        for j,el in enumerate(els): 
            newatms += [[el,list(coords[j])]]
    mol.atom = newatms
    mol.build()
    return mol

