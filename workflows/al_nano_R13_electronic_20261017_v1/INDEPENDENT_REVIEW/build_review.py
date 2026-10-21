"""Read-only analysis of already-computed R13 results; no DFT or scheduler calls."""
from pathlib import Path
import json, hashlib, re

ROOT = Path('C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/FOLLOWUP_RESULTS/DFTPY_R13')
OUT = Path(__file__).resolve().parent
HIST = Path('C:/OFDFT/AL_DEFECTS_USB_HANDOFF_20260717/02_ACTIVE_DFTPY_ISERVICE_LAMMU_RAW/03_RAW_CASES/03_runs/tfvw_lam0p9_mu0p1')
PROBE = Path('C:/Users/m1331/Documents/Codex/2026-09-24/g/work/followup_results_20261017/runtime_probe.json')
def read(p): return p.read_text(encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p): return json.loads(read(p))
def rows(p): return [json.loads(x) for x in read(p).splitlines() if x.strip()]
result,config,cp = [j(ROOT / n) for n in ('result.json','config.json','checkpoint.json')]
electronic,ionic = [rows(ROOT / n) for n in ('electronic_evaluations.jsonl','ionic_evaluations.jsonl')]
log=read(ROOT/'dftpy_full_stdout.log')
last=log.rsplit('The final grid size is',1)[-1]
last_rows=[]
for line in last.splitlines():
    v=line.split()
    if len(v)==7 and v[0].isdigit():
        last_rows.append(dict(step=int(v[0]),energy_Ha_printed=float(v[1]),dE_Ha_printed=float(v[2]),residual_squared_norm_printed=float(v[3]),Ndirection=int(v[4]),Nlinesearch=int(v[5])))
tail=ionic[-20:]
sources=[ROOT/n for n in ['result.json','config.json','checkpoint.json','electronic_evaluations.jsonl','ionic_evaluations.jsonl','dftpy_full_stdout.log','inputs/dft_engine.py','inputs/recovery_runner.py','provenance.json']]
sources += [HIST/n for n in ['dftpy_pristine_input.ini','dftpy_vacancy_input.ini','run_status_job1575772_task80.json','result.json']]
sources += [PROBE]
out=dict(
    scope='Independent source and log review only; no DFT, SSH, submission or physical acceptance changes',
    case='R13_0V',job='2252172',accepted=False,
    verified=dict(electronic_calls=len(electronic),electronic_successes=sum(x['accepted'] for x in electronic),electronic_failure=electronic[-1],
        completed_ionic_states=len(ionic),completed_ionic_last_step=ionic[-1]['ionic_step'],
        force_initial_eV_A=ionic[0]['fmax_eV_A'],force_last_eV_A=ionic[-1]['fmax_eV_A'],
        force_min_eV_A=min(x['fmax_eV_A'] for x in ionic),force_target_eV_A=.001,
        min_force_step=min(ionic,key=lambda x:x['fmax_eV_A'])['ionic_step'],
        energy_change_meV=(ionic[-1]['energy_eV']-ionic[0]['energy_eV'])*1000,
        last20_energy_span_meV=(max(x['energy_eV'] for x in tail)-min(x['energy_eV'] for x in tail))*1000,
        last_stress_zz_GPa=ionic[-1]['stress_GPa'][2][2],
        wall_seconds=result['elapsed_s'],peak_rss_MiB=result['peak_rss_MiB'],
        last_electronic_printed_rows=last_rows,
        last_coherent_checkpoint_sha256=cp['checkpoint_bundle_sha256'],
        config_OPT=config['OPT'],versions=j(ROOT/'provenance.json').get('versions',j(ROOT/'provenance.json').get('packages')),
        runtime_probe=j(PROBE)),
    confirmed_mechanism=[
        'Electronic minimizer returned flag 1 (line-search failure); runner rejected downstream forces/stress and stopped.',
        'Only two electronic updates were printed in failed evaluation; ncheck=3 requires three earlier energies against latest before declaring energy convergence.',
        'Last coherent checkpoint is step 100 and its force still exceeds target. Failed evaluation 102 is not a valid physical endpoint.',
        'Ionic small-step LBFGS and electronic LBFGS are separate algorithms despite identical short name.',
        'Root-agent read-only runtime probe of the same remote environment confirms scipy.optimize._dcsrch is available despite reported SciPy 1.10.1. Importing the exact frozen engine enables the compat flag and replaces LineSearchDcsrch2 with its _line_search_scalar. Runtime evidence takes precedence over version-only inference.'
    ],
    hypotheses=[
        'Tiny electronic dE and residual are consistent with near-flat line-search difficulty, but task/alpha/derivative details were not logged, so numerical roundoff versus search-direction/step failure cannot be distinguished.',
        'Exact failing line-search task was not logged, so confirmed compatibility-layer activation does not identify the underlying reason for search failure.'
    ],
    historical_electronic_fallback=dict(method='TN',input=str(HIST/'dftpy_pristine_input.ini'),status=j(HIST/'run_status_job1575772_task80.json'),
        scope='Historical successful bulk TF=.9/vW=.1 calculation at econv=1e-6, force=.01, different grid. Evidence of executed optimizer family only, not current nano/.001/econv1e-9 validation.',
        CG_HS='Existing bulk-scan preparation script and explanatory document found; no matching accepted raw CG-HS case established in this targeted search.'),
    recommended_minimum_test=[
        'Start from coherent step100 checkpoint, preserve model/PP/grid/stress candidate and force=.001, axial stress=.05 GPa acceptance.',
        'At fixed checkpoint geometry first compare existing electronic LBFGS with TN at identical econv=1e-9 Ha/atom, ncheck=3, maxiter=1000/maxls=50; do not move ions until a clean electronic result is available.',
        'Preserve verified callable provenance and record task/alpha/direction derivative on failure, optimizer identity, energy history, residual, charge integral and finite density.',
        'Compare energy, all atomic force vectors and axial stress between successful fixed-geometry solutions. If force sensitivity is comparable to .001 eV/A, add a tighter-electronic-tolerance diagnostic rather than declare geometric convergence.',
        'Only after fixed-geometry check passes, resume the preserved small-step ionic LBFGS workflow with explicitly recorded electronic fallback; keep original failure evidence immutable.',
        'TN still uses line search and is not guaranteed to cure the event. If it fails, inspect recorded task before testing a bounded CG-HS alternative. Do not launch another blind 1000-step LBFGS-only attempt.'
    ],
    prohibited_shortcuts=['ncheck=1','looser econv','looser ionic force criterion','accept failed density using its tiny energy change','claim historical FIRE success','change functional/grid/PP to hide optimizer problem'],
    official_sources=[
        'https://raw.githubusercontent.com/Quantum-MultiScale/DFTpy/v2.1.2/src/dftpy/optimization/optimization.py',
        'https://raw.githubusercontent.com/Quantum-MultiScale/DFTpy/v2.1.2/src/dftpy/math_utils.py',
        'https://raw.githubusercontent.com/Quantum-MultiScale/DFTpy/v2.1.2/src/dftpy/interface.py',
        'https://dftpy.rutgers.edu/tutorials/config.html',
        'https://raw.githubusercontent.com/scipy/scipy/v1.10.1/scipy/optimize/_linesearch.py',
        'https://dftpy.rutgers.edu/releases.html'
    ],
    source_hashes={str(p):sha(p) for p in sources})
if out['verified']['versions'] is None:
    prov=j(ROOT/'provenance.json')
    out['verified']['versions']={k:v for k,v in prov.items() if 'version' in k or 'package' in k}
assert len(electronic)==102 and sum(x['accepted'] for x in electronic)==101
assert electronic[-1]['converged_flag']==1 and len(last_rows)==3
assert ionic[-1]['ionic_step']==100 and ionic[-1]['fmax_eV_A']>.001
assert cp['checkpoint_bundle_sha256']==sha(ROOT/'checkpoint_bundle.npz')
(OUT/'review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out['verified'],ensure_ascii=False,indent=2))
