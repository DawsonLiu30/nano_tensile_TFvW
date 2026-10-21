"""Cheap total axial stress/energy consistency test, no SCF or optimization.

Finite strain scales cell, fractional ionic coordinates and density with
inverse volume at fixed fractional-grid values (electron count conserved).
Includes TF, vW, XC, Hartree, local PP and ion-ion energy, with component
diagnostics and two finite-difference steps. Report only, no production fix.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, copy, json, hashlib, time
from pathlib import Path
import numpy as np
from ase.io import read
from ase.units import Bohr, Hartree
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.ions import Ions
from dftpy.interface import ConfigParser
from stress_candidate import install

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
GPA = Hartree / Bohr**3 * 160.21766208

def evaluate(atoms, values, config, strain=0.0):
    a = atoms.copy(); cell = np.array(a.cell); cell[2] *= 1+strain
    a.set_cell(cell, scale_atoms=True)
    grid = DirectGrid(lattice=np.asarray(a.cell)/Bohr, nr=values.shape)
    rho = DirectField(grid=grid, griddata_3d=values/(1+strain))
    cfg, others = ConfigParser(copy.deepcopy(config), ions=Ions.from_ase(a), rhoini=rho, grid=grid)
    ev = others['E_v_Evaluator']; rr = others['rho']
    en = ev.get_energy_potential(rr, calcType={'E'}, split=True)
    stress = ev.get_stress(rr, split=True) if strain == 0 else None
    return {k:float(v.energy) for k,v in en.items()}, stress, float(rr.integral()), grid.volume

def main():
    p=argparse.ArgumentParser();p.add_argument('--case-dir', required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); base=Path(a.case_dir); out=Path(a.output)
    atoms=read(base/'final.extxyz'); z=np.load(base/'final_density.npz')
    values=z['density_electrons_bohr3'];cfg=json.loads((base/'config.json').read_text())
    cfg['PATH']['pppath']=str((base/'inputs').resolve())
    assert np.max(abs(z['lattice_bohr']-np.asarray(atoms.cell)/Bohr))<1e-6
    install(); t=time.time(); es, ss, n, v=evaluate(atoms,values,cfg)
    rows=[]
    for h in [1e-4,5e-5]:
        ep,_,np_,_=evaluate(atoms,values,cfg,h); em,_,nm,_=evaluate(atoms,values,cfg,-h)
        for k in es:
            fd=(ep[k]-em[k])/(2*h*v)*GPA
            ana=float(ss[k][2,2])*GPA
            rows.append({'step':h,'component':k,'analytic_GPa':ana,'finite_difference_GPa':fd,'abs_error_GPa':abs(fd-ana)})
        assert abs(n-np_)<1e-7 and abs(n-nm)<1e-7
    total=[r for r in rows if r['component']=='TOTAL']
    # 0.001 GPa is 50 times tighter than the production axial acceptance.
    # Two-step stability guards cancellation/truncation; this is a test criterion.
    tol=.001; stability=.0005
    accepted=bool(all(r['abs_error_GPa']<=tol for r in total) and abs(total[0]['finite_difference_GPa']-total[1]['finite_difference_GPa'])<=stability)
    d={'scope':'Fixed fractional saved density, conserved electrons, affine ions; total axial discrete derivative, not self-consistent validation',
       'case':base.name,'accepted':accepted,'tolerance_GPa':tol,'fd_step_stability_GPa':stability,'electrons':n,'grid':list(values.shape),
       'elapsed_s':time.time()-t,'energy_eV':es['TOTAL']*Hartree,'rows':rows,
       'source_hashes':{x:sha(base/x) for x in ['final.extxyz','final_density.npz','config.json','inputs/al.lda.recpot']},
       'candidate_sha256':sha(Path(__file__).with_name('stress_candidate.py'))}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'accepted':accepted,'total':total,'elapsed_s':d['elapsed_s']}))
    return 0 if accepted else 2
if __name__=='__main__':raise SystemExit(main())
