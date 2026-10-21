"""Independent control-flow and FD-gate checks. Mock densities only; no DFT."""
from pathlib import Path
import sys, json, copy, hashlib, tempfile
import numpy as np
from types import SimpleNamespace
WORK=Path(__file__).resolve().parents[2]/'r13_electronic_recovery_20261017'
sys.path.insert(0,str(WORK))
import recover_electronic as r
from electronic_trace import ElectronicTrace
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from ase.units import Bohr
import dftpy.optimization.optimization as module

checks=[]
def check(name,fn):
    fn();checks.append({'name':name,'passed':True})
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid case was accepted')

def fixtures():
    pos=np.zeros((76,3));force=np.zeros((76,3));force[10,2]=.0039
    centre=dict(accepted=True,native_converged_flag=0,grid=[140,140,96],electrons=228,ncheck=3,method='TN',econv_Ha_per_atom=1e-11,
        initial_density='warm',energy_eV=-4760.,forces_eV_A=force.tolist(),stress_GPa=np.zeros((3,3)).tolist(),cell_A=np.diag([20.,20.,13.]).tolist(),positions_A=pos.tolist(),force_atom_index=10,
        physical_attempts=[dict(accepted=True,native_converged_flag=0,initial_density='warm')])
    out={key:copy.deepcopy(centre) for key in ['TN_reference','TN_tight_center','TN_tight_HEG','FD_m1000','FD_p1000','FD_m0500','FD_p0500']}
    out['TN_reference']['econv_Ha_per_atom']=1e-9
    out['TN_tight_HEG']['initial_density']='HEG';out['TN_tight_HEG']['physical_attempts'][0]['initial_density']='HEG'
    for h,label in [(.001,'1000'),(.0005,'0500')]:
        for sign,tag in [(-1,'m'),(1,'p')]:
            q=out['FD_'+tag+label];q['positions_A'][10][2]=sign*h;q['energy_eV']=-4760.-sign*h*.0039
    return out

check('Gate accepts analytic linear-energy fixture',lambda: r.evaluate_gate(fixtures()))
def bad_fd():
    f=fixtures();f['FD_p1000']['energy_eV']+=1e-5;reject(lambda:r.evaluate_gate(f))
check('Gate rejects wrong force derivative',bad_fd)
def bad_geom():
    f=fixtures();f['FD_m0500']['positions_A'][9][0]=.001;reject(lambda:r.evaluate_gate(f))
check('Gate rejects extra displaced atom',bad_geom)
def bad_flag():
    f=fixtures();f['TN_tight_center']['native_converged_flag']=1;reject(lambda:r.evaluate_gate(f))
check('Gate rejects native failure despite accepted label',bad_flag)
def no_heg():
    f=fixtures();f['TN_tight_HEG']['initial_density']='warm';reject(lambda:r.evaluate_gate(f))
check('Gate requires independent HEG centre',no_heg)
def bad_criterion():
    f=fixtures();f['TN_tight_center']['ncheck']=1;reject(lambda:r.evaluate_gate(f))
check('Gate rejects weakened ncheck',bad_criterion)
def bad_econv():
    f=fixtures();f['TN_tight_center']['econv_Ha_per_atom']=1e-9;reject(lambda:r.evaluate_gate(f))
check('Gate rejects weakened candidate tolerance',bad_econv)
def nan_charge():
    f=fixtures();f['TN_tight_HEG']['electrons']=float('nan');reject(lambda:r.evaluate_gate(f))
check('Gate rejects nonfinite charge',nan_charge)

grid=DirectGrid(lattice=np.diag([10.,10.,10.]),nr=[2,2,2])
rho=DirectField(grid=grid,griddata_3d=np.full((2,2,2),6./grid.volume))
original=module.Optimization.optimize_rho
def retry_case(flags,method='TN',mode='warm',invalid=False,exception=False):
    calls=[]
    def native(opt,guess_rho):
        calls.append(np.array(guess_rho).copy())
        if exception:raise RuntimeError('Unrelated native failure')
        opt.converged=flags[len(calls)-1];opt.functional=SimpleNamespace(energy=-1.)
        ans=guess_rho.copy()
        if invalid:ans[0,0,0]=0
        return ans
    module.Optimization.optimize_rho=native
    with tempfile.TemporaryDirectory() as td:
        trace=ElectronicTrace(td,2,[2,2,2],grid.lattice*Bohr);trace.initial_mode=mode
        opt=SimpleNamespace(optimization_method=method,optimization_options={'econv':2e-11,'ncheck':3},rho=rho.copy())
        try:
            with trace.installed():out=module.Optimization.optimize_rho(opt,guess_rho=rho.copy())
            err=None
        except (RuntimeError,ValueError) as exc:err=exc
        finally:module.Optimization.optimize_rho=original
    return calls,trace,err
def recover_once():
    calls,t,e=retry_case([1,0]);assert e is None and len(calls)==2 and t.records[0]['native_converged_flag']==1 and t.records[-1]['native_converged_flag']==0 and t.retry_count==1
    assert abs(calls[1].mean()*grid.volume-6)<1e-12
check('Warm TN flag1 recovers once from charge-normalized HEG with original native0',recover_once)
def bound_failure():
    calls,t,e=retry_case([1,1]);assert e is not None and len(calls)==2 and t.retry_count==1
check('Second native failure stops without third attempt',bound_failure)
for name,kw in [('Flag2 never retried',dict(flags=[2])),('LBFGS failure never retried',dict(flags=[1],method='LBFGS')),('Initial HEG failure never retried',dict(flags=[1],mode='HEG')),('Unrelated exception never retried',dict(flags=[1],exception=True)),('Invalid returned density never retried',dict(flags=[1],invalid=True))]:
    def f(kw=kw):
        calls,t,e=retry_case(**kw);assert e is not None and len(calls)==1 and t.retry_count==0
    check(name,f)

def same_base():
    frozen=Path('/mnt/c/OFDFT/nano_tensile_TFvW/workflows/al_nano_smallstep_20261017_v1/recovery_base.py')
    assert hashlib.sha256(frozen.read_bytes()).digest()==hashlib.sha256((WORK/'recovery_base.py').read_bytes()).digest()
check('Scientific base runner is byte-identical to submitted smallstep source',same_base)
summary={'scope':'No DFT: mocked density retry control-flow, analytic FD-gate fixture, source identity','checks':checks,'passed':len(checks),'failed':0,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.glob('*.py')}}
(Path(__file__).resolve().parent/'review_test_results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'failed':0}))
