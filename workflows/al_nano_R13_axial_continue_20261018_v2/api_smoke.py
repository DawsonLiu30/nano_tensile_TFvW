"""No-DFT API/derivative tests; these do not validate an Al calculation."""
from pathlib import Path
import json, hashlib
import numpy as np
import ase
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from ase.optimize import BFGSLineSearch
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.utils import field2distrib

class Analytic(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=['energy'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        r=self.atoms.positions; cell=self.atoms.cell.array; v=self.atoms.get_volume()
        q=np.log(cell[2,2]/5.)
        self.results={'energy':.5*np.sum(r*r)+10*q*q,'forces':-r,
                      'stress':(r.T@r+np.diag([0,0,20*q]))/v}

def run():
    a=Atoms('Al2',positions=[[.03,0,0],[-.02,0,0]],cell=[6,7,5.03],pbc=True)
    a.calc=Analytic();f=FrechetCellFilter(a,mask=[0,0,1,0,0,0],exp_cell_factor=2.,scalar_pressure=0.,constant_volume=False)
    opt=BFGSLineSearch(f,maxstep=.005,logfile=None)
    interface=opt.optimizable
    value=interface.get_value() if hasattr(interface,'get_value') else interface.get_potential_energy()
    assert interface._use_force_consistent_energy is False
    startcell=a.cell.array.copy();x=f.get_positions();analytic=f.get_forces()[-1,-1];h=1e-5
    energies=[]
    for sign in [-1,1]:
        z=x.copy();z[-1,-1]+=sign*h;f.set_positions(z);energies.append(f.get_potential_energy(force_consistent=False))
    f.set_positions(x)
    err=abs(analytic+(energies[1]-energies[0])/(2*h));assert err<1e-8
    history=[]
    opt.attach(lambda:history.append((a.get_potential_energy(),a.cell[2,2],float(np.linalg.norm(f.get_forces(),axis=1).max()))))
    accepted=bool(opt.run(fmax=1e-5,steps=100));assert accepted
    assert np.allclose(a.cell.array[:2],startcell[:2],atol=1e-12,rtol=0)
    assert abs(a.cell[2,2]-5.)<1e-5
    assert max(np.diff([x[0] for x in history]),default=0)<1e-12
    cell=np.diag([12.,13.,14.]);nr=[4,4,4];ne=228.
    g0=DirectGrid(lattice=cell,nr=nr);rho=DirectField(grid=g0);rho[:]=np.arange(1,65).reshape(4,4,4)
    rho*=ne/rho.integral();newcell=cell.copy();newcell[2,2]*=1.001
    g1=DirectGrid(lattice=newcell,nr=nr);out=DirectField(grid=g1);field2distrib(rho,out)
    before=float(out.integral());out*=ne/out.integral()
    assert abs(float(out.integral())-ne)<1e-10
    assert np.allclose(out,rho/1.001,atol=1e-15,rtol=0)
    return {'scope':'Analytic calculator and density remapping only; no DFT/Al physical acceptance.',
       'ASE_version':ase.__version__,'free_energy_missing_fallback':True,
       'filter_zz_force_derivative_error':float(err),'analytic_optimizer_converged':accepted,
       'outer_steps':opt.nsteps,'last_analytic_fmax':history[-1][2],
       'initial_cell_A':startcell.tolist(),'final_cell_A':a.cell.array.tolist(),
       'energy_nonincreasing':True,'remap_before_normalization_electrons':before,
       'remap_after_normalization_electrons':float(out.integral()),'same_nr_mapping':'same fractional indices, then volume normalization'}

if __name__=='__main__':
    report=run();out=Path(__file__).with_name('api_smoke_results.json')
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
