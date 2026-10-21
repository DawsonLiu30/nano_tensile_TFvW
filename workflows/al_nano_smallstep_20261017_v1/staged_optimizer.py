"""Reusable small-step, fixed-cell ionic finish after variable-cell relaxation.

Historical workflow used this optimizer family, but its looser force threshold
is not reused. Current atomic/filter/axial stress gates remain unchanged.
"""
import json
from pathlib import Path
import numpy as np

POLICY = {
    'revision': 'fixed_cell_smallstep_LBFGS_finish_20261017_v1',
    'optimizer': 'LBFGS',
    'maxstep_A': 0.01,
    'cell_degrees_of_freedom_during_finish': 'none; cell fixed at coherent parent checkpoint',
    'filter_role': 'original z-only FrechetCellFilter retained for final force/stress acceptance',
    'electronic_optimizer': 'unchanged LBFGS',
    'scope': 'historical optimizer family; current stricter R13 physical endpoint still requires acceptance',
}

class FixedCellFinish:
    """ASE-compatible adapter; base runner remains responsible for acceptance."""
    def __init__(self,cell_filter,logfile,ionic):
        from ase.optimize import LBFGS
        if ionic['optimizer']!='LBFGS_fixed_cell_finish' or ionic.get('optimizer_policy')!=POLICY:
            raise ValueError('Fixed-cell finish policy mismatch')
        if ionic['maxstep_A']!=.01:
            raise ValueError('Small-step finish must use 0.01 A')
        self.cell_filter=cell_filter
        self.atoms=cell_filter.atoms
        self.start_cell=self.atoms.cell.array.copy()
        self.event_path=Path(logfile).with_name('optimizer_events.jsonl')
        self.dyn=LBFGS(self.atoms,logfile=logfile,maxstep=.01)
        self.dyn.attach(self._record,interval=1)
    @property
    def nsteps(self):return self.dyn.nsteps
    def attach(self,function,interval=1,*args,**kwargs):
        if interval!=1:raise ValueError('Checkpoint observation must be every step')
        self.dyn.attach(function,interval=interval,*args,**kwargs)
    def _record(self):
        if not np.allclose(self.atoms.cell.array,self.start_cell,atol=1e-12,rtol=0):
            raise ValueError('Fixed-cell finish changed cell')
        row={'ionic_step':self.nsteps,'optimizer':'LBFGS','stage':'fixed_cell_smallstep_finish',
             'atomic_fmax_eV_A':float(np.linalg.norm(self.atoms.get_forces(),axis=1).max()),
             'filter_fmax_eV_A':float(np.linalg.norm(self.cell_filter.get_forces(),axis=1).max()),
             'energy_eV':float(self.atoms.get_potential_energy()),'maxstep_A':.01}
        with self.event_path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def run(self,fmax,steps):
        if steps<=0:raise ValueError('Bounded positive step count required')
        # Returning atomic convergence does not bypass base.acceptance(), which
        # additionally checks the unchanged filter force and sigma_zz targets.
        return self.dyn.run(fmax=fmax,steps=steps)

def make_optimizer(cell_filter,logfile,ionic):
    return FixedCellFinish(cell_filter,logfile,ionic)
