import numpy as np
import matplotlib.pyplot as plt

def fix_mos(las,verbose=False):
    mo = las.mo_coeff
    ref_orbs = las.ncas_sub
    s = las._scf.get_ovlp()
    
    mos = mo[:,las.ncore:las.ncore+las.ncas]
    nfrags = len(ref_orbs)

    #Make transmos to get overlaps correct
    ao_offset = las.mol.nao // nfrags
    mo_offset = ref_orbs[0]
    for i in range(nfrags):
        if i == 0:
            transmos = mos[:,:mo_offset]
            continue
        idx = (np.arange(mos.shape[0])+ao_offset*i)%mos.shape[0]
        to_add = mos[:,mo_offset*i:mo_offset*(i+1)][idx,:]
        transmos = np.hstack([transmos,to_add])
       
    #Fix signs
    mo_offset = ref_orbs[0]

    for i in range(mo_offset):
        idx = [mo_offset*j+i for j in range(nfrags)]
        transmo = transmos[:,idx]
        ovlp = np.linalg.multi_dot([transmo.T,s,transmo])[0]
        
        assert(np.allclose(np.abs(ovlp),np.ones(len(ovlp)),atol=1e-1))
        
        if verbose:
            print(f"orbital {i}:",ovlp)
        sign = ovlp < 0
        sign = np.ones(transmo.shape[1]) - 2*sign
        sign = sign.reshape(1,transmo.shape[-1])
        mos[:,idx] = mos[:,idx]*sign
    mo[:,las.ncore:las.ncore+las.ncas] = mos
    
    check = np.linalg.multi_dot([mo.T,s,mo]) - np.eye(mo.shape[1])
    assert(np.allclose(check,0))
    return mo

def fix_sign(las):
    nfrags = len(las.ncas_sub)
    newci = [[]]*nfrags
    for frag_idx in range(nfrags):
        for state_idx in range(len(las.ci[0])):                 
            to_add = las.ci[frag_idx][state_idx].copy()
            if len(to_add.shape) == 3: #excited states
                for civec in to_add:
                    idx = np.argmax(np.abs(civec.ravel()))
                    sign = -np.sign(civec.ravel()[idx])
                    civec *= sign
            else:
                idx = np.argmax(np.abs(to_add.ravel()))
                sign = -np.sign(to_add.ravel()[idx])
                to_add *= sign
            newci[frag_idx] = newci[frag_idx] + [to_add]
    return newci

# def fix_sign(las):
#     nfrags = len(las.ncas_sub)
#     newci = [[]]*nfrags
#     for frag_idx in range(nfrags):
#         for state_idx in range(len(las.ci[0])):
#             to_add = las.ci[frag_idx][state_idx]
#             idx = np.argmax(np.abs(to_add.ravel()))
#             sign = -np.sign(to_add.ravel()[idx])
#             newci[frag_idx] = newci[frag_idx] + [to_add * sign]
#     return newci

#Look at CI vectors:
def transci(las,las_charges,charge,plot=False):
    ci = las.ci.copy()
    nfrags = len(las.ncas_sub)
    charges = np.array(las_charges).sum(axis=1)
    state_idxs = np.where(charges == charge)[0]
    mat = []
    cuts = []
    for i,state_idx in enumerate(state_idxs):
        lst = []
        for frag_idx in range(nfrags):
            lst += [ci[frag_idx][state_idx].ravel()]
            if i == 0:
                cuts += [len(ci[frag_idx][state_idx].ravel())]
        mat += [np.hstack(lst)]
    mat = np.vstack(mat)
    print(np.cumsum(cuts))

    #translate assuming charge starts on first fragment
    transmat = mat.copy()
    offset = ci[0][state_idxs[0]+1].ravel().shape[0]
    for i in range(mat.shape[0]):
        idxs = (np.arange(mat.shape[1]) + offset*i)%mat.shape[1]
        transmat[i,:] = transmat[i,idxs]

    if plot:
        sns.heatmap(transmat)
        for c in np.cumsum(cuts):
            plt.axvline(c,color="red",linestyle="--",linewidth=2)

    return transmat
