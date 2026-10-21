"""Bounded ASE BFGSLineSearch on the existing z-only FrechetCellFilter."""
import json,time
from pathlib import Path
import numpy as np

POLICY={'revision':'R13_axial_BFGSLineSearch_20261018_v1','optimizer':'BFGSLineSearch',
 'maxstep_generalized_A':.005,'cell_mask':[0,0,1,0,0,0],'exp_cell_factor':'n_atoms',
 'c1':.23,'c2':.46,'alpha':10.,'stpmax':50.,'max_ionic_steps':60,'max_logical_scf':120,
 'cycle_periods':[2,4],'cycle_windows':3,'cycle_position_tolerance_A':1e-4,
 'cycle_energy_tolerance_eV':1e-6,'cycle_force_relative_improvement':.05,
 'electronic_method':'TN','econv_Ha_per_atom':1e-11,'ncheck':3,
 'scope':'Short candidate trial; not yet a physically accepted optimizer workflow'}

def append(path,row):
    with Path(path).open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def repeating_cycle(history):
    for period in POLICY['cycle_periods']:
        needed=period*POLICY['cycle_windows']+1
        if len(history)<needed:continue
        recent=history[-needed:];same=[(recent[i],recent[i-period]) for i in range(period,len(recent))]
        dx=max(float(np.max(np.abs(a['x']-b['x']))) for a,b in same)
        de=max(abs(a['energy_eV']-b['energy_eV']) for a,b in same)
        force_reduction=1-recent[-1]['filter_fmax_eV_A']/max(recent[0]['filter_fmax_eV_A'],1e-30)
        if dx<=POLICY['cycle_position_tolerance_A'] and de<=POLICY['cycle_energy_tolerance_eV'] and force_reduction<POLICY['cycle_force_relative_improvement'] and recent[-1]['filter_fmax_eV_A']>.001:
            return {'period':period,'max_same_phase_coordinate_change_A':dx,'max_same_phase_energy_change_eV':de,'relative_force_improvement':force_reduction}
    return None

class AxialLineSearch:
    def __init__(self,cell_filter,logfile,ionic):
        from ase.optimize import BFGSLineSearch
        import ase.optimize.bfgslinesearch as bmod
        if ionic['optimizer']!='BFGSLineSearch_axial' or ionic.get('optimizer_policy')!=POLICY or ionic['maxstep_A']!=.005:raise ValueError('Axial linesearch policy mismatch')
        self.cell_filter=cell_filter;self.atoms=cell_filter.atoms;self.start_cell=self.atoms.cell.array.copy();self.history=[]
        self.events=Path(logfile).with_name('optimizer_events.jsonl');self.trials=Path(logfile).with_name('ionic_line_search.jsonl')
        adapter=self
        class Audited(BFGSLineSearch):
            def func(self,x):
                value=super().func(x)
                append(adapter.trials,{'kind':'energy_trial','ionic_step':self.nsteps,'energy_eV':float(value*self.alpha),'generalized_x':np.asarray(x).tolist(),'cell_A':adapter.atoms.cell.array.tolist()})
                return value
            def fprime(self,x):
                grad=super().fprime(x)
                append(adapter.trials,{'kind':'gradient_trial','ionic_step':self.nsteps,'gradient_norm':float(np.linalg.norm(grad)*self.alpha),'generalized_x':np.asarray(x).tolist(),'cell_A':adapter.atoms.cell.array.tolist()})
                return grad
            def step(self,*args,**kwargs):
                original=bmod.LineSearch
                class TracedSearch(original):
                    def _line_search(ls,*p,**kw):
                        t=time.monotonic();row={'kind':'line_search_result','ionic_step':self.nsteps}
                        try:
                            result=super()._line_search(*p,**kw)
                            row.update(alpha=result[0],scaled_final_energy=result[1],scaled_initial_energy=result[2],no_update=bool(result[3]),task=str(ls.task))
                            if str(ls.task).startswith(('WARNING','ERROR')) or result[0] is None:
                                raise RuntimeError('Ionic line-search rejected: '+str(ls.task))
                            if not str(ls.task).startswith('CONV'):
                                raise RuntimeError('Ionic line-search did not establish convergence: '+str(ls.task))
                            return result
                        except BaseException as exc:row['error']=repr(exc);raise
                        finally:row['elapsed_s']=time.monotonic()-t;append(adapter.trials,row)
                bmod.LineSearch=TracedSearch
                try:return super().step(*args,**kwargs)
                finally:bmod.LineSearch=original
        self.dyn=Audited(cell_filter,logfile=logfile,maxstep=.005,c1=.23,c2=.46,alpha=10.,stpmax=50.)
        self.dyn.attach(self._record,interval=1)
    @property
    def nsteps(self):return self.dyn.nsteps
    def attach(self,fn,interval=1,*args,**kw):
        if interval!=1:raise ValueError('Checkpoint every accepted step required')
        self.dyn.attach(fn,interval=interval,*args,**kw)
    def _record(self):
        cell=self.atoms.cell.array;mask=np.ones((3,3),dtype=bool);mask[2,2]=False
        if not np.allclose(cell[mask],self.start_cell[mask],atol=1e-8,rtol=0) or np.diag(cell).min()<10 or not np.all(self.atoms.pbc):raise ValueError('Axial cell constraint/image changed')
        x=np.asarray(self.cell_filter.get_positions()).copy();f=float(np.linalg.norm(self.cell_filter.get_forces(),axis=1).max())
        row={'ionic_step':self.nsteps,'optimizer':'BFGSLineSearch','stage':'atoms_and_axial_zz_linesearch','energy_eV':float(self.atoms.get_potential_energy()),'atomic_fmax_eV_A':float(np.linalg.norm(self.atoms.get_forces(),axis=1).max()),'filter_fmax_eV_A':f,'cell_A':cell.tolist(),'maxstep_generalized_A':.005}
        if self.history:
            from ase.geometry import find_mic
            previous=self.history[-1];displacements=find_mic(self.atoms.positions-previous['atomic_positions'],cell,self.atoms.pbc)[0]
            row['max_generalized_row_step_A']=float(np.linalg.norm(x-previous['x'],axis=1).max())
            row['max_actual_atomic_displacement_A']=float(np.linalg.norm(displacements,axis=1).max())
            row['delta_Lz_A']=float(cell[2,2]-previous['cell_A'][2][2])
        append(self.events,row);self.history.append({**row,'x':x,'atomic_positions':self.atoms.positions.copy()})
    def _stop_checks(self):
        # Attached after base checkpoint: retain this completed state before a stop.
        cycle=repeating_cycle(self.history)
        if cycle:
            append(self.events,{'ionic_step':self.nsteps,'event':'stagnation_stop','cycle':cycle})
            from recovery_base import BudgetExpired
            raise BudgetExpired('Repeated ionic cycle detected; checkpoint retained')
    def run(self,fmax,steps):
        if not 0<steps<=60 or fmax!=.001:raise ValueError('Bounded steps/force policy changed')
        self.dyn.attach(self._stop_checks,interval=1)
        return self.dyn.run(fmax=fmax,steps=steps)

def make_optimizer(cell_filter,logfile,ionic):return AxialLineSearch(cell_filter,logfile,ionic)
