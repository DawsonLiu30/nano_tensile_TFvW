"""Independent real ASE adapter tests with an analytic calculator, never DFT."""
from pathlib import Path
import sys,json,tempfile,hashlib
import numpy as np
import ase
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.filters import FrechetCellFilter
import ase.optimize.bfgslinesearch as bmod
HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'r13_linesearch_recovery_20261018'
sys.path.insert(0,str(CODE))
from staged_optimizer import AxialLineSearch,POLICY,repeating_cycle

class Analytic(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=['energy'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        r=self.atoms.positions;v=self.atoms.get_volume();q=np.log(self.atoms.cell[2,2]/13.)
        self.results={'energy':.5*np.sum(r*r)+10*q*q,'forces':-r,'stress':(r.T@r+np.diag([0,0,20*q]))/v}
def setup(td):
    a=Atoms('Al2',positions=[[.03,0,0],[-.02,0,0]],cell=[20,21,13.03],pbc=True);a.calc=Analytic()
    f=FrechetCellFilter(a,mask=[0,0,1,0,0,0],exp_cell_factor=2.,scalar_pressure=0.,constant_volume=False)
    opt=AxialLineSearch(f,str(Path(td)/'relax.log'),dict(optimizer='BFGSLineSearch_axial',optimizer_policy=POLICY,maxstep_A=.005))
    return a,f,opt
tests=[]
with tempfile.TemporaryDirectory() as td:
    a,f,opt=setup(td);states=[];opt.attach(lambda:states.append((a.positions.copy(),a.cell.array.copy())))
    ok=opt.run(fmax=.001,steps=60);assert ok
    assert abs(a.cell[2,2]-13.)<.001 and np.allclose(a.cell.array[:2],np.diag([20,21,13.03])[:2],atol=1e-12,rtol=0)
    assert opt.dyn.optimizable._use_force_consistent_energy is False
    events=[json.loads(x) for x in (Path(td)/'ionic_line_search.jsonl').read_text().splitlines()]
    accepted=[x for x in events if x['kind']=='line_search_result'];assert accepted and all(x['task'].startswith('CONV') for x in accepted)
    assert len(states)==opt.nsteps+1
    tests.append(dict(name='real_adapter_analytic_convergence_no_free_energy',passed=True,steps=opt.nsteps,checkpoint_callbacks=len(states)))
original=bmod.LineSearch
try:
    class WarningSearch(original):
        def _line_search(self,func,grad,x,p,g,e,eold,**kw):
            moved=func(x+.1*p);self.task='WARNING: ROUNDING ERRORS PREVENT PROGRESS'
            return .1,moved,e,False
    bmod.LineSearch=WarningSearch
    with tempfile.TemporaryDirectory() as td:
        a,f,opt=setup(td);saved=[];opt.attach(lambda:saved.append(a.positions.copy()))
        start=a.positions.copy()
        try:opt.run(fmax=.001,steps=60)
        except RuntimeError as err:assert 'Ionic line-search rejected' in str(err)
        else:raise AssertionError('Warning alpha was silently accepted')
        assert opt.nsteps==0 and len(saved)==1 and np.array_equal(saved[0],start)
        assert not np.array_equal(a.positions,start)
        assert bmod.LineSearch is WarningSearch
        tests.append(dict(name='warning_trial_rejected_and_prior_completed_observer_state_retained',passed=True,checkpoint_callbacks=len(saved)))
finally:bmod.LineSearch=original
h=[dict(x=np.full((3,3),(k%4)*.01),energy_eV=float(k%4)*1e-5,filter_fmax_eV_A=.003) for k in range(13)]
assert repeating_cycle(h)['period']==4
h[-1]['filter_fmax_eV_A']=.0005;assert repeating_cycle(h) is None
tests.append(dict(name='cycle_stop_does_not_replace_force_acceptance',passed=True))
report={'scope':'Analytic/mocked tests only; no DFT physical validation','ASE_version':ase.__version__,'checks':tests,'passed':len(tests),'source_sha256':hashlib.sha256((CODE/'staged_optimizer.py').read_bytes()).hexdigest()}
(HERE/'adapter_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report,indent=2))
