import numpy as np
from pyscf.scf import hf
from pyscf import gto, ao2mo

"""
Reconstruction of SCF method
        mf.get_init_guess   Completed
        mf.get_hcore        Completed
        mf.get_ovlp         Completed
        mf.get_veff         Completed
        mf.get_fock         Completed
        mf.get_grad         Completed
        mf.eig              Completed
        mf.get_occ          Completed
        mf.make_rdm1        Completed
        mf.energy_tot       
        mf.dump_chk         
"""


def gen_emb_orb(dm:np.ndarray, na, cal:np.ndarray, nocc):
    """
    DMET 

    Ref: PhD Thesis of Boxiao Zheng

    Attributes:
        dm: density matrix of RHF
        na: number of FRAGMENT orbitals or INDICES
        cal: transformation coefficient from ATOMIC to LOCALIZED
        nocc: number of total occupied number

    Return: Fragment/Bath/Unentangled orbitals in ATOMIC base
    """

    if type(na) in [list, np.ndarray]:
        mask = np.ones(dm.shape[0], np.bool_)
        mask[na] = False

    if type(na)==int: 
        daa = dm[:na,:na]
        dab = dm[:na,na:]
        dbb = dm[na:,na:,]
        calcuta = cal[:,:na]
        calcutb = cal[:,na:]
        lenna = na
    else: 
        daa = dm[na][:,na]
        dab = dm[na][:,mask]
        dbb = dm[mask][:,mask]
        calcuta = cal[:,na]     # C_{\mu j}, j \in N_A
        calcutb = cal[:,mask]   # C_{\mu j}, j \in N_B
        lenna = len(na)

    up, sp, vp = np.linalg.svd(daa)
    p = np.einsum("ij,j->ij", up, np.sqrt(sp))
    q = np.dot(np.linalg.inv(p), dab).conj().T
    uq, sq, vq = np.linalg.svd(np.dot(q, q.conj().T))
    pnorm = np.sqrt(np.einsum("ik,ik->k", p, p.conj()))
    q = np.einsum("ij,j->ij", uq[:,:lenna], np.sqrt(sq)[:lenna])
    qnorm = np.sqrt(np.einsum("jk,jk->k", q, q.conj()))
    eed = dbb - np.dot(q, q.conj().T)
    uu, su, vu = np.linalg.svd(eed)
    uocc = uu[:,np.where(abs(su)>1E-12)[0]]   # choose occupied unentangled orbitals

    # caf: ATOMIC -> FRAGMENT / IMPURITY coefficient
    # cab: ATOMIC -> BATH coefficient 
    # cau: ATOMIC -> UNENTANGLED / CORE coefficient
    caf = np.einsum("nj,k,jk->nk", calcuta, 1/pnorm, p)
    cab = np.einsum("nj,k,jk->nk", calcutb, 1/qnorm, q)
    cau = np.dot(calcutb, uocc)

    occfb = np.hstack((sp, sq[:lenna]))

    return (caf, cab, cau, occfb)

def make_Hring(l=1):
    r = l/(2 * np.sin(np.pi / 10))
    atmlst = []
    for i in range(10):
        atmlst.append(['H', (r*np.cos(np.pi/10*i), r*np.sin(np.pi/10*i), 0)])
    mol = gto.Mole()
    mol.atom = atmlst
    mol.basis = 'sto-6g'
    mol.build()

    return mol


class DMET_SCF(hf.SCF): # SCF in the (fragment + environment) subspace

    def __init__(self, mf:hf.RHF, eo_coeff, mu_glob=0):

        mol = mf.mol
        self._scf = mf

        hf.SCF.__init__(self, mol)
        caf, cab, cau, occfb = eo_coeff
        self.na = caf.shape[1]
        self.cal = lo.orth_ao(self._scf)
        self.fb_coeff = np.hstack((caf, cab))
        self.un_coeff = cau
        self.dm = np.diag(occfb)    # as initial guess of
        print(np.allclose(np.sum(occfb), self.na*2))
        # unentangled dm in AO
        self.un_dm = 2. * np.dot(self.un_coeff, self.un_coeff.conj().T)
        self.mu_glob = mu_glob
        
        self.nocc = cau.shape[1] + self.na

        # densitry matrix projected to frag+bath bases
        self.init_guess = "project" 

        # eri[ijkl] = (ij|kl) , in frag and bath orbital
        erifb = ao2mo.kernel(mol.intor('int2e', aosym='s8'), self.fb_coeff, compact=False)
        self.erifb = erifb.reshape((self.na*2, self.na*2, self.na*2, self.na*2))

    def get_occ(self, mo_energy, mo_coeff=None):
        occidx = np.argsort(mo_energy)[:self.na]
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[occidx] = 2

        return mo_occ
    
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        
        hao = hf.get_hcore(mol)
        vj, vk = hf.get_jk(mol, self.un_dm)
        hall = hao + vj - .5 * vk
        h = np.einsum("mj,mn,nk->jk", self.fb_coeff.conj(), hall, self.fb_coeff)

        # minus global chemical potential
        for i in range(self.na):
            h[i,i] -= self.mu_glob
        return h

    def get_ovlp(self, mol=None):   # unitary matrix
        fbdim = self.fb_coeff.shape[1]
        return np.eye(fbdim)

    # with new 'get_jk' function, then 'get_veff' and 'get_fock' are used as before
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        # density matrix in the frag+bath bases
        return hf.dot_eri_dm(self.erifb, dm, hermi, with_j, with_k)

    def get_init_guess(self, mol=None, key='project'):
        if key == "project": return self.init_guess_by_project(mol)
        else:
            raise NotImplementedError("Other initial guess methods are not available!")

    def init_guess_by_project(self, mol=None):
        return self.dm

    def eig(self, h, s):
        # eigenvalue solver since orbitals are orthornomal.
        e, c = np.linalg.eig(h)
        idx = np.argmax(abs(c.real), axis=0)
        c[:,c[idx,np.arange(len(e))].real<0] *= -1
        return e, c

    # temporately defined as energy_elec
    def energy_tot(self, dm=None, h1e=None, vhf=None):
        return self.energy_elec(dm, h1e, vhf)[0]

    def fci(self):
        # Full-CI Solver for DMET
        from pyscf.fci.direct_spin0 import FCISolver
        norb = 2*self.na
        nelec = 2*self.na
        h1e = reduce(np.dot, (self.mo_coeff.conj().T, self.get_hcore(), self.mo_coeff))
        eri = ao2mo.kernel(self.erifb, self.mo_coeff, compact=False)
        eri = eri.reshape(norb,norb,norb,norb)
        cis = FCISolver(self.mol)
        e1, c1 = cis.kernel(h1e, eri, norb, nelec)
        print("c1 = \n", c1)
        print("e1 = ", e1)
        dm = cis.make_rdm1(c1, norb, nelec)
        print("FCIdm = \n", dm)
        return e1, dm


if __name__ == "__main__":
    # Test of H10 system
    from functools import reduce
    from pyscf import gto, scf, lo, ao2mo
    import numpy as np
    
    mol = make_Hring()
    print(mol.nao)
    print(mol.ao_loc)
    m = scf.RHF(mol)
    m.kernel()
    print(m.e_tot)
    cam = m.mo_coeff
    cal = lo.orth_ao(m)
    clm = np.dot(np.linalg.inv(cal), cam)
    occ = m.mo_occ
    # Density matrix from C^{L->M} N C^{L->M\dagger}
    dm = np.einsum("ij,j,jk->ik", clm, occ, clm.conj().T)
    nocc = mol.nelec[0]
    nfrag = [0]   # indices of chosen localized orbitals

    # molecular coefficient of F/B/U orbitals
    c = np.hstack(gen_emb_orb(dm, nfrag, cal, nocc)[:3])
    s = m.get_ovlp()
    csc = reduce(np.dot, (c.conj().T, s, c))
    print("Orthonormal: ", np.allclose(csc, np.eye(nocc+len(nfrag))))

    dmetscf = DMET_SCF(m, gen_emb_orb(dm, nfrag, cal, nocc), mu_glob=0)
    dmetscf.kernel()
    print(dmetscf.mo_energy)
    print(dmetscf.mo_occ)
    print(dmetscf.make_rdm1())
    e1, dm = dmetscf.fci()
    print(e1 * 10. + m.energy_nuc())