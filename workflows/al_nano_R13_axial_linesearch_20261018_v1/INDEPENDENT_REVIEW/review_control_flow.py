"""Independent mock electronic tests; no electronic optimization is performed."""
import sys,json,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ase.units import Bohr
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
import dftpy.optimization.optimization as m

HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'r13_linesearch_recovery_20261018'
sys.path.insert(0,str(CODE))
from electronic_trace import ElectronicTrace
from recovery_base import BudgetExpired
original=m.Optimization.optimize_rho
checks=[]
def fixture(cell):
    grid=DirectGrid(lattice=cell/Bohr,nr=[2,2,2]);rho=DirectField(grid=grid)
    rho[:]=6./grid.volume;return rho
def scenario(kind):
    initial=np.diag([20.,20.,13.]);current=initial.copy();current[2,2]*=1.001
    geom={'cell_A':current.tolist(),'pbc':[True]*3}
    guess=fixture(current);calls=[]
    flags={'retry':[1,0],'no_retry_flag2':[2],'second_failure':[1,1]}.get(kind,[0])
    if kind=='stale_density':guess=fixture(initial)
    if kind=='transverse':geom['cell_A'][0][0]+=.01
    if kind=='bad_pbc':geom['pbc'][2]=False
    if kind=='bad_charge':guess*=1.01
    def native(opt,guess_rho):
        calls.append(np.asarray(guess_rho).copy());opt.converged=flags[len(calls)-1];opt.functional=SimpleNamespace(energy=-1.)
        if kind=='cell_changed_during_solve':geom['cell_A'][2][2]+=.0001
        return guess_rho.copy()
    m.Optimization.optimize_rho=native
    try:
        with tempfile.TemporaryDirectory() as td:
            t=ElectronicTrace(td,2,[2,2,2],initial,geometry_provider=lambda:geom,max_logical_calls=1)
            opt=SimpleNamespace(optimization_method='TN',optimization_options={'econv':2e-11,'ncheck':3},rho=guess)
            err=None
            try:
                with t.installed():
                    m.Optimization.optimize_rho(opt,guess_rho=guess)
                    if kind=='budget':m.Optimization.optimize_rho(opt,guess_rho=guess)
            except (ValueError,RuntimeError,BudgetExpired) as exc:err=exc
            if kind in ['valid','retry']:assert err is None
            else:assert err is not None
            if kind in ['stale_density','transverse','bad_pbc','bad_charge']:assert len(calls)==0
            if kind in ['no_retry_flag2','cell_changed_during_solve','budget']:assert len(calls)==1
            if kind in ['retry','second_failure']:assert len(calls)==2 and t.retry_count==1
            if kind=='retry':assert t.records[0]['native_converged_flag']==1 and t.records[-1]['native_converged_flag']==0
            if kind=='budget':assert isinstance(err,BudgetExpired)
            if kind=='valid':assert np.allclose(t.records[0]['calculator_cell_A'],current,atol=0,rtol=0)
            checks.append({'name':kind,'passed':True,'physical_calls':len(calls),'error':None if err is None else type(err).__name__})
    finally:m.Optimization.optimize_rho=original
for kind in ['valid','retry','second_failure','no_retry_flag2','stale_density','transverse','bad_pbc','bad_charge','cell_changed_during_solve','budget']:scenario(kind)
report={'scope':'Mock optimize_rho only; no DFT, force calculation, or physical validation.',
 'passed':len(checks),'checks':checks,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [CODE/'electronic_trace.py',CODE/'recovery_base.py']}}
(HERE/'control_flow_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report,indent=2))
