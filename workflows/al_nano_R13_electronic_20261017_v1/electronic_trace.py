"""Process-local observation plus one explicit, bounded TN warm->HEG retry.

No package file is edited. Native convergence flags and ncheck are unchanged.
Failed physical attempts remain in JSONL; only native flag 0 may return a density.
"""
from __future__ import annotations
import contextlib, json, time, traceback
from pathlib import Path
import numpy as np
from ase.units import Bohr

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):
        return float(x) if np.isfinite(x) else str(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,bytes):return x.decode('utf8',errors='replace')
    return x

def validate_density(rho,n,grid,cell):
    arr=np.asarray(rho)
    if list(arr.shape)!=list(grid) or list(map(int,rho.grid.nr))!=list(grid):raise ValueError('Density grid changed')
    if not np.all(np.isfinite(arr)) or not np.all(arr>0):raise ValueError('Nonfinite/nonpositive density')
    ne=float(np.sum(rho.integral()))
    if not np.isfinite(ne) or abs(ne-3*n)>1e-6:raise ValueError('Electron count mismatch')
    if not np.allclose(np.asarray(rho.grid.lattice)*Bohr,cell,atol=1e-8,rtol=0):raise ValueError('Density cell changed')
    return {'electrons':ne,'density_min':float(arr.min()),'density_max':float(arr.max()),'grid':list(map(int,rho.grid.nr))}

def make_heg(rho,n):
    out=rho.copy();out[:]=3*n/float(rho.grid.volume)
    return out

class ElectronicTrace:
    def __init__(self,out,n,grid,cell,allow_retry=True):
        self.out=Path(out);self.n=n;self.grid=list(grid);self.cell=np.asarray(cell).copy()
        self.allow_retry=allow_retry;self.label='production';self.initial_mode='warm'
        self.logical_count=0;self.physical_count=0;self.failed_physical=0;self.retry_count=0
        self.current={};self.records=[]
    def append(self,name,row):
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/name).open('a',encoding='utf8') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
    def summary(self):
        return {'logical_calls':self.logical_count,'physical_attempts':self.physical_count,
                'failed_physical_attempts':self.failed_physical,'HEG_retries':self.retry_count,
                'all_physical_attempts_converged':self.failed_physical==0,
                'retry_policy':'Only TN warm native flag=1 permits one HEG attempt; no other failure is retried.'}
    @contextlib.contextmanager
    def installed(self):
        import dftpy.optimization.optimization as m
        old_opt=m.Optimization.optimize_rho;old_dir=m.Optimization.get_direction;old_ls=m.LineSearchDcsrch2
        def traced_ls(func,*args,**kw):
            row={**self.current,'kind':'line_search','alpha0':kw.get('alpha0'),'options':{k:v for k,v in kw.items() if k!='func0'},'trials':[]}
            if kw.get('func0') is not None:row['initial_energy_Ha']=float(kw['func0'][0]);row['initial_directional_derivative']=float(kw['func0'][1])
            def observed(alpha,*a,**k):
                val=func(alpha,*a,**k);row['trials'].append({'alpha':float(alpha),'energy_Ha':float(val[0]),'directional_derivative':float(val[1])});return val
            try:
                result=old_ls(observed,*args,**kw)
                row.update(alpha_out=result[0],returned_energy_Ha=result[1],returned_directional_derivative=result[2],task=result[3],line_search_evaluations=result[4],accepted_step=result[0] is not None)
                return result
            except BaseException as exc:row['error']=repr(exc);raise
            finally:self.append('line_search.jsonl',row)
        def traced_dir(opt,resA,*args,**kw):
            t=time.monotonic();row={**self.current,'kind':'direction','method':opt.optimization_method}
            try:
                residual=resA[-1];row['residual_squared_integral']=float(np.sum(np.asarray(residual)**2)*residual.grid.dV)
                result=old_dir(opt,resA,*args,**kw)
                row.update(direction_squared_integral=float(np.sum(np.asarray(result[0])**2)*result[0].grid.dV),inner_direction_iterations=int(result[1]));return result
            except BaseException as exc:row['error']=repr(exc);raise
            finally:row['elapsed_s']=time.monotonic()-t;self.append('directions.jsonl',row)
        def traced_opt(opt,*args,**kw):
            if len(args)>1 or set(kw)-{'guess_rho'} or (args and 'guess_rho' in kw):
                raise ValueError('This bounded wrapper supports only one guess_rho argument')
            self.logical_count+=1;logical=self.logical_count
            guess=kw.get('guess_rho',args[0] if args else None)
            if guess is None:guess=opt.rho
            validate_density(guess,self.n,self.grid,self.cell)
            mode=self.initial_mode
            for attempt in (1,2):
                self.physical_count+=1;self.current={'logical_call':logical,'label':self.label,'physical_attempt':attempt,'initial_density':mode,'electronic_method':opt.optimization_method}
                row={**self.current,'effective_econv_Ha_total':float(opt.optimization_options['econv']),'ncheck':opt.optimization_options['ncheck']}
                started=time.monotonic();native_flag=None;rho=None;error=None
                try:
                    # Native optimization initializes its direction history each call.
                    rho=old_opt(opt,guess_rho=guess)
                    native_flag=int(opt.converged);row['native_converged_flag']=native_flag
                    row['energy_Ha']=float(opt.functional.energy)
                    row.update(validate_density(rho,self.n,self.grid,self.cell))
                    if not np.isfinite(row['energy_Ha']):raise ValueError('Nonfinite electronic energy')
                    if native_flag!=0:raise RuntimeError('Native electronic failure flag='+str(native_flag))
                    row['accepted']=True
                    return rho
                except BaseException as exc:
                    error=exc;row.update(accepted=False,error=repr(exc),traceback=traceback.format_exc());self.failed_physical+=1
                finally:
                    row['elapsed_s']=time.monotonic()-started;self.records.append(clean(row));self.append('physical_electronic_attempts.jsonl',row)
                retry=(self.allow_retry and opt.optimization_method=='TN' and mode=='warm' and attempt==1 and native_flag==1 and isinstance(error,RuntimeError) and str(error)=='Native electronic failure flag=1')
                if not retry:raise error
                self.retry_count+=1;mode='HEG_retry';guess=make_heg(guess,self.n)
            raise RuntimeError('Bounded electronic retry exhausted')
        m.LineSearchDcsrch2=traced_ls;m.Optimization.get_direction=traced_dir;m.Optimization.optimize_rho=traced_opt
        try:yield self
        finally:
            m.LineSearchDcsrch2=old_ls;m.Optimization.get_direction=old_dir;m.Optimization.optimize_rho=old_opt

