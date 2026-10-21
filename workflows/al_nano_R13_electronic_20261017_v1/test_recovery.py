"""Synthetic rejection/flow tests, no DFT or remote operations."""
import copy,json,tempfile,unittest,contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from ase.units import Bohr
from electronic_trace import ElectronicTrace,validate_density,make_heg
from recover_electronic import evaluate_gate,SETTINGS,diagnostic
import recovery_base
import dftpy.optimization.optimization as dm

class Rho(np.ndarray):
    def __new__(cls,value=1.):
        cell=np.diag([20.,20.,13.4]);lat=cell/Bohr;volume=float(np.linalg.det(lat))
        a=np.full((2,2,2),228/volume*value).view(cls)
        a.grid=SimpleNamespace(nr=np.array([2,2,2]),lattice=lat,volume=volume,dV=volume/8)
        return a
    def __array_finalize__(self,o):self.grid=getattr(o,'grid',None)
    def integral(self):return float(np.sum(self)*self.grid.dV)

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def trace(self):return ElectronicTrace(self.root,76,[2,2,2],np.diag([20.,20.,13.4]))
    def native_case(self,flags,method='TN',mode='warm',invalid=False,exc=None):
        calls=[]
        def native(opt,guess_rho):
            calls.append(guess_rho.copy())
            if exc:raise exc
            opt.converged=flags[len(calls)-1];opt.functional=SimpleNamespace(energy=-175.-len(calls));opt.mu=-.1*len(calls)
            opt.rho=guess_rho.copy()
            if invalid:opt.rho[0,0,0]=0
            return opt.rho
        opt=SimpleNamespace(optimization_method=method,optimization_options={'econv':7.6e-10,'ncheck':3})
        trace=self.trace();trace.initial_mode=mode
        return trace,opt,calls,native
    def test_native_success_no_retry(self):
        t,o,c,n=self.native_case([0])
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():r=dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),1);self.assertEqual(o.converged,0);self.assertEqual(t.failed_physical,0)
    def test_retry_keeps_fail_and_updates_all_caller_state(self):
        t,o,c,n=self.native_case([1,0]);guess=Rho();guess[0,0,0]*=.5;guess*=228/guess.integral()
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():r=dm.Optimization.optimize_rho(o,guess_rho=guess)
        self.assertEqual(len(c),2);self.assertEqual(o.converged,0);self.assertEqual(o.functional.energy,-177.);self.assertEqual(o.mu,-.2)
        np.testing.assert_array_equal(r,o.rho);self.assertGreater(np.ptp(c[0]),0);self.assertEqual(float(np.ptp(c[1])),0)
        self.assertAlmostEqual(c[1].integral(),228);self.assertEqual([x['native_converged_flag'] for x in t.records],[1,0]);self.assertEqual(t.retry_count,1)
        self.assertFalse(t.summary()['all_physical_attempts_converged'])
    def test_two_failures_stop(self):
        t,o,c,n=self.native_case([1,1])
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaisesRegex(RuntimeError,'flag=1'):dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),2)
    def test_flag2_not_retried(self):
        t,o,c,n=self.native_case([2])
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaisesRegex(RuntimeError,'flag=2'):dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),1)
    def test_LBFGS_not_retried(self):
        t,o,c,n=self.native_case([1],method='LBFGS')
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaises(RuntimeError):dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),1)
    def test_HEG_failure_not_retried(self):
        t,o,c,n=self.native_case([1],mode='HEG')
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaises(RuntimeError):dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),1)
    def test_invalid_density_not_retried(self):
        t,o,c,n=self.native_case([0],invalid=True)
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaisesRegex(ValueError,'density'):dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),1)
    def test_deadline_not_retried(self):
        t,o,c,n=self.native_case([],exc=recovery_base.BudgetExpired('Wall-time budget reached'))
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaises(recovery_base.BudgetExpired):dm.Optimization.optimize_rho(o,guess_rho=Rho())
        self.assertEqual(len(c),1)
    def test_diagnostic_deadline_does_not_proceed_to_TN(self):
        from ase import Atoms
        def dump(p,v):Path(p).write_text(json.dumps(v))
        base=SimpleNamespace(validate_output_root=lambda p:p.resolve(),budget_guard=lambda s:contextlib.nullcontext())
        g=SimpleNamespace(base=base,dump=dump)
        args=SimpleNamespace(output_root=self.root,parent_runroot=self.root,max_runtime_s=6600)
        atoms=Atoms('Al76',positions=np.zeros((76,3)),cell=np.diag([20,20,13.4]),pbc=True)
        prep=(g,{},atoms,{}, {'GRID':{'nr':[2,2,2]}},{},{},{'revision_manifest_sha256':'test'})
        with patch('recover_electronic.field',return_value=Rho()),patch('recover_electronic.scf',side_effect=recovery_base.BudgetExpired('Wall-time budget reached')) as call:
            self.assertEqual(diagnostic(args,prep),1);self.assertEqual(call.call_count,1)
        result=json.loads((self.root/'R13_0V/diagnostics_003_electronic/diagnostic_result.json').read_text())
        self.assertFalse(result['accepted']);self.assertIn('BudgetExpired',result['error'])
    def test_unsupported_native_arguments_rejected(self):
        t,o,c,n=self.native_case([0])
        with patch.object(dm.Optimization,'optimize_rho',n),t.installed():
            with self.assertRaises(ValueError):dm.Optimization.optimize_rho(o,guess_rho=Rho(),lphi=True)
        self.assertEqual(len(c),0)
    def test_search_task_trials_are_not_modified(self):
        def ls(fun,**kw):v=fun(.2);return None,v[0],v[1],b'WARN ROUNDING ERRORS',1,v
        t=self.trace()
        with patch.object(dm,'LineSearchDcsrch2',ls),t.installed():
            result=dm.LineSearchDcsrch2(lambda x:(-1+x*x,2*x,None,None),alpha0=.1,func0=(-1.,-.1,None,None))
        self.assertIsNone(result[0]);self.assertEqual(result[3],b'WARN ROUNDING ERRORS')
        row=json.loads((self.root/'line_search.jsonl').read_text());self.assertEqual(row['task'],'WARN ROUNDING ERRORS');self.assertEqual(row['trials'][0]['alpha'],.2)
    def test_density_invalid_grid_charge_cell(self):
        with self.assertRaisesRegex(ValueError,'grid'):validate_density(Rho(),76,[3,2,2],np.diag([20,20,13.4]))
        with self.assertRaisesRegex(ValueError,'count'):validate_density(Rho(2),76,[2,2,2],np.diag([20,20,13.4]))
        with self.assertRaisesRegex(ValueError,'cell'):validate_density(Rho(),76,[2,2,2],np.diag([20,20,14]))
    def fixture(self):
        positions=np.zeros((76,3));forces=np.zeros((76,3));forces[58,2]=.003
        physical={'accepted':True,'native_converged_flag':0,'initial_density':'warm'}
        c={'accepted':True,'native_converged_flag':0,'method':'TN','econv_Ha_per_atom':1e-11,'ncheck':3,'grid':[140,140,96],'electrons':228.,'cell_A':np.diag([20,20,13.4]).tolist(),'positions_A':positions.tolist(),'forces_eV_A':forces.tolist(),'stress_GPa':np.zeros((3,3)).tolist(),'energy_eV':-100.,'force_atom_index':58,'initial_density':'warm','physical_attempts':[physical]}
        rows={k:copy.deepcopy(c) for k in ['TN_reference','TN_tight_center','TN_tight_HEG','FD_m1000','FD_p1000','FD_m0500','FD_p0500']}
        rows['TN_reference']['econv_Ha_per_atom']=1e-9;rows['TN_tight_HEG']['initial_density']='HEG';rows['TN_tight_HEG']['physical_attempts'][0]['initial_density']='HEG'
        for h,suffix in zip([.001,.0005],['1000','0500']):
            for sign,tag in [(-1,'m'),(1,'p')]:
                r=rows['FD_'+tag+suffix];r['positions_A'][58][2]=sign*h;r['energy_eV']=-100-.003*sign*h+.5*h*h
        return rows
    def test_selfconsistent_FD_gate_passes(self):self.assertTrue(evaluate_gate(self.fixture())['accepted'])
    def test_failed_native_flag_rejected(self):
        rows=self.fixture();rows['FD_m1000']['native_converged_flag']=1
        with self.assertRaises(ValueError):evaluate_gate(rows)
    def test_force_difference_rejected(self):
        rows=self.fixture();rows['TN_tight_HEG']['forces_eV_A'][5][0]=.00011
        with self.assertRaisesRegex(ValueError,'reproduction'):evaluate_gate(rows)
    def test_energy_difference_rejected(self):
        rows=self.fixture();rows['TN_tight_HEG']['energy_eV']+=.000011
        with self.assertRaisesRegex(ValueError,'reproduction'):evaluate_gate(rows)
    def test_repeat_stress_rejected(self):
        rows=self.fixture();rows['TN_tight_HEG']['stress_GPa'][2][2]=.0011
        with self.assertRaisesRegex(ValueError,'reproduction'):evaluate_gate(rows)
    def test_repeat_is_really_HEG(self):
        rows=self.fixture();rows['TN_tight_HEG']['initial_density']='warm'
        with self.assertRaisesRegex(ValueError,'HEG'):evaluate_gate(rows)
    def test_wrong_electronic_threshold_rejected(self):
        rows=self.fixture();rows['TN_tight_center']['econv_Ha_per_atom']=1e-9
        with self.assertRaisesRegex(ValueError,'threshold'):evaluate_gate(rows)
    def test_ncheck_change_rejected(self):
        rows=self.fixture();rows['FD_m0500']['ncheck']=1
        with self.assertRaisesRegex(ValueError,'protocol'):evaluate_gate(rows)
    def test_wrong_FD_axis_rejected(self):
        rows=self.fixture();rows['FD_m0500']['positions_A'][58][0]=-.0005
        with self.assertRaisesRegex(ValueError,'displacement'):evaluate_gate(rows)
    def test_bad_FD_rejected(self):
        rows=self.fixture();rows['FD_p1000']['energy_eV']+=1e-5
        with self.assertRaisesRegex(ValueError,'finite difference'):evaluate_gate(rows)
    def test_cell_change_rejected(self):
        rows=self.fixture();rows['FD_p0500']['cell_A'][2][2]+=1e-4
        with self.assertRaisesRegex(ValueError,'cell'):evaluate_gate(rows)
    def test_invalid_retry_rejected(self):
        rows=self.fixture();rows['TN_tight_center']['physical_attempts'].insert(0,{'accepted':False,'native_converged_flag':2,'initial_density':'warm'})
        with self.assertRaisesRegex(ValueError,'retry'):evaluate_gate(rows)
    def test_parent_full_acceptance_retained(self):
        p={'ionic':{'fmax_eV_A':.001,'filter_fmax_eV_A':.001},'cell':{'stress_max_abs_GPa':.05}};m={'fmax_eV_A':.0009,'filter_fmax_eV_A':.0009,'stress_GPa':np.diag([100,100,.04]).tolist()}
        self.assertTrue(recovery_base.acceptance(True,m,[{'accepted':True}],p))
        for key in ['fmax_eV_A','filter_fmax_eV_A']:
            mm=copy.deepcopy(m);mm[key]=.00101;self.assertFalse(recovery_base.acceptance(True,mm,[{'accepted':True}],p))
        self.assertFalse(recovery_base.acceptance(True,m,[{'accepted':False}],p))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'tests':r.testsRun,'passed':r.testsRun-len(r.errors)-len(r.failures),'errors':len(r.errors),'failures':len(r.failures),'scope':'Synthetic program checks, no DFT'}))
    raise SystemExit(0 if r.wasSuccessful() else 1)
