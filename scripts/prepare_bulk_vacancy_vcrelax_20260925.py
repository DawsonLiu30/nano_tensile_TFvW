"""Reproducible common initial structures and separately specified QE/OF methods.

Writes a new campaign only; never runs calculations or alters prior campaigns.
The legacy topology builder is reused with explicitly overridden length and ID.
"""
from pathlib import Path
import argparse
import hashlib
import json
import build_bulk_vacancy_campaign_20260924 as geom

REPO = Path(__file__).resolve().parents[1]
ID = 'al_bulk_vacancy_vcrelax_20260925'
RY_EV = 13.605693122994
BOHR_A = 0.529177210903
PAW_SHA = '450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a'
LOCAL_SHA = 'c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def put(path, data):
    data = data.encode('utf-8') if isinstance(data, str) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f'Refusing to change existing input: {path}')
    if not path.exists():
        path.write_bytes(data)

def build(paw_source):
    folder = REPO / 'campaigns' / ID
    paw = Path(paw_source).read_bytes()
    local = (REPO / 'al.lda.recpot').read_bytes()
    assert sha(paw) == PAW_SHA and sha(local) == LOCAL_SHA
    geom.CAMPAIGN = ID
    geom.A_REF = 4.039848
    geom.LENGTH = 3 * geom.A_REF
    g, files = geom.build()
    g.update(purpose='Common ideal FCC starting configurations for independent zero-pressure variable-cell relaxations.',
             a_ref_meaning='Common starting lattice constant from June16 PBE/PAW input, not an asserted equilibrium a0 for either new calculation.',
             generator='scripts/prepare_bulk_vacancy_vcrelax_20260925.py')
    g['excluded_from_first_round'] = ['Half-cell-tied [110] vector (3,3,0) a/2; no isolated far-vacancy reference asserted.']
    files['geometry.json'] = (json.dumps(g, indent=2, ensure_ascii=False)+'\n').encode()
    files['pseudo/Al_PAW_PBE.UPF'] = paw
    files['pseudo/al.lda.recpot'] = local
    threshold = .0005 / (RY_EV / BOHR_A)
    protocol = {
        'calculation':'vc-relax', 'xc':'PBE', 'pseudopotential_type':'PAW',
        'cell_dofree':'all','press_kbar':0,'press_conv_thr_kbar':.5,
        'force_norm_target_eV_A':.001,
        'ecutwfc_Ry':58.7989154810,'ecutrho_Ry':470.3913238484,
        'ecutwfc_eV':58.7989154810*RY_EV,'kmesh':[3,3,3],'koffset':[0,0,0],
        'smearing':'mp','degauss_Ry':.02,'nbnd':200,
        'conv_thr_Ry':1e-8,'electron_maxstep':300,'mixing_beta':.2,
        'mixing_ndim':4,'nstep':200,'max_seconds':39600,
        'ion_dynamics':'bfgs','cell_dynamics':'bfgs','etot_conv_thr_Ry':1e-6,
        'forc_conv_thr_Ry_bohr':threshold,'force_component_tolerance_eV_A':.0005,
        'nosym':True,'noinv':True,'tstress':True,'tprnfor':True,
        'historical_source':'/home/dawson666/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616/pair_scan/pair_01_r2p8566A/pristine_vcrelax/vc-relax.in',
        'historical_source_sha256':'d22c3c847ba57ac7480ffbf6ebc10cca7ae2b3c0636a4250a9457894bc10c63c',
        'provenance_note':'Cutoffs, PP, k mesh, smearing, XC, mixing beta and cell settings retained from actual historical input. Force and energy targets are the stricter September24 campaign targets; nbnd=200, mixing_ndim=4, nstep=200 and wall limit remain explicit September24 operational/numerical choices.',
        'convergence_scope':'800 eV is a PPT tested bulk EOS cutoff and the actual historical 108-site vc setting. This does not assert complete defect k-point or cell-size convergence.',
    }
    qe={'schema_version':1,'campaign_id':ID,'protocol':protocol,
        'geometry_manifest':{'file':'geometry.json','sha256':sha(files['geometry.json'])},
        'pseudo':{'file':'pseudo/Al_PAW_PBE.UPF','sha256':PAW_SHA},
        'case_order':g['case_order'],'cases':{}}
    for cid in g['case_order']:
        c=g['cases'][cid]
        text=f"""&CONTROL
 calculation = 'vc-relax'
 prefix = 'Al_{cid}'
 outdir = './scratch'
 pseudo_dir = '../../../pseudo'
 restart_mode = 'from_scratch'
 verbosity = 'high'
 disk_io = 'medium'
 tstress = .true.
 tprnfor = .true.
 max_seconds = 39600
 nstep = 200
 etot_conv_thr = 1.0d-6
 forc_conv_thr = {threshold:.14e}
/
&SYSTEM
 ibrav = 0
 nat = {c['n_atoms']}
 ntyp = 1
 nbnd = 200
 ecutwfc = 58.7989154810
 ecutrho = 470.3913238484
 occupations = 'smearing'
 smearing = 'mp'
 degauss = 0.02
 nspin = 1
 tot_charge = 0.0
 nosym = .true.
 noinv = .true.
/
&ELECTRONS
 conv_thr = 1.0d-8
 electron_maxstep = 300
 scf_must_converge = .true.
 mixing_mode = 'plain'
 mixing_beta = 0.2
 mixing_ndim = 4
 diagonalization = 'david'
 diago_david_ndim = 2
 startingpot = 'atomic'
 startingwfc = 'atomic+random'
/
&IONS
 ion_dynamics = 'bfgs'
/
&CELL
 cell_dynamics = 'bfgs'
 press = 0.0
 press_conv_thr = 0.5
 cell_dofree = 'all'
/
ATOMIC_SPECIES
Al 26.9815385 Al_PAW_PBE.UPF

CELL_PARAMETERS angstrom
"""
        text+='\n'.join(' '.join(f'{v:.16f}' for v in row) for row in c['cell_A'])
        text+='\n\nATOMIC_POSITIONS crystal\n'
        text+='\n'.join('Al '+' '.join(f'{v:.16f}' for v in row) for row in c['fractional'])
        text+='\n\nK_POINTS automatic\n3 3 3 0 0 0\n'
        rel=f'qe/inputs/{cid}/pw.in'
        files[rel]=text.encode()
        qe['cases'][cid]={'input':rel,'sha256':sha(files[rel]),'nat':c['n_atoms'],'q':c['q'],
            'structure':c['structure_files']['poscar'],'structure_sha256':c['structure_sha256']['poscar'],
            'extxyz':c['structure_files']['extxyz'],'extxyz_sha256':c['structure_sha256']['extxyz'],
            'removed_parent_ids':c['removed_parent_ids']}
    of={'schema_version':1,'campaign_id':ID,
        'geometry':{'file':'geometry.json','sha256':sha(files['geometry.json'])},
        'xc':'LDA','pseudo':{'file':'pseudo/al.lda.recpot','sha256':LOCAL_SHA},
        'grid':[78,78,78],'lambda_tf':.9,'mu_vw':.1,
        'electronic':{'method':'LBFGS','econv_Ha_per_atom':1e-9,'ncheck':3,'maxiter':1000,'maxls':50,'maxfun':100,'densityini':'heg','reuse':True},
        'ionic':{'optimizer':'LBFGS','fmax_eV_A':.001,'filter_fmax_eV_A':.001,'max_steps':1000,'maxstep_A':.15},
        'cell':{'filter':'FrechetCellFilter','scalar_pressure_GPa':0,'hydrostatic_strain':False,'constant_volume':False,'stress_max_abs_GPa':.05,'exp_cell_factor':'n_atoms'},
        'runtime':{'max_walltime_s':6600},
        'numerical_provenance':'September15 LDA/local lambda=.9,mu=.1 grid test: 60 to 78 changes Eb(1NN) by -0.1705 meV and Eb(2NN) by -0.0759 meV; September16 EOS 78 vs96 gives da0=4.97e-6 A and dB0=0.000627 GPa. Fixed point count retained through variable-cell trajectory; actual spacing recorded. OF density grid is not QE wavefunction cutoff.',
        'comparison_scope':'User explicitly retains different XC/pseudopotentials: QE PAW/PBE versus OF local/LDA. Compare within-method formation/binding energies and structures, never subtract raw absolute total energies across methods.'}
    files['qe_manifest.json']=(json.dumps(qe,indent=2)+'\n').encode()
    files['dftpy_manifest.json']=(json.dumps(of,indent=2)+'\n').encode()
    for rel,data in files.items():put(folder/rel,data)
    return {'campaign':str(folder),'cases':g['case_order'],'geometry_sha256':sha(files['geometry.json']),'qe_sha256':sha(files['qe_manifest.json']),'dftpy_sha256':sha(files['dftpy_manifest.json'])}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--paw-source',required=True)
    print(json.dumps(build(parser.parse_args().paw_source),indent=2))
