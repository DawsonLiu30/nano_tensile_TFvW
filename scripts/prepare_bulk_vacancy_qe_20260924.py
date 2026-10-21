#!/usr/bin/env python3
"""Prepare an immutable fixed-cell QE relaxation campaign, without executing QE."""
from pathlib import Path
import hashlib
import json
import math

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / 'campaigns/al_bulk_vacancy_relax_20260924'
RY_EV = 13.605693122994
BOHR_A = 0.529177210903

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def put(path, data):
    data = data.encode('utf-8') if isinstance(data, str) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f'Refusing to overwrite a different scientific input: {path}')
    if not path.exists():
        path.write_bytes(data)

def main():
    g = json.loads((CAMPAIGN / 'geometry.json').read_text(encoding='utf-8'))
    pp = CAMPAIGN / 'pseudo/Al.pz-vbc.UPF'
    put(pp, (REPO/'campaigns/qe_divacancy_20260923/pseudo/Al.pz-vbc.UPF').read_bytes())
    threshold = 0.0005 / (RY_EV / BOHR_A)
    protocol = {
        'calculation':'relax', 'boundary':'fixed cell; ions relaxed from ideal FCC',
        'a_ref_A':g['a_ref_A'], 'cell_edge_A':g['cell_edge_A'],
        'a_ref_note':'common previous reference; not a new QE or TFvW equilibrium lattice constant',
        'parent_sites':108, 'xc':'LDA Slater/PZ from Al.pz-vbc.UPF',
        'valence_electrons_per_atom':3, 'ecutwfc_Ry':60, 'ecutrho_Ry':240,
        'kmesh':[3,3,3], 'koffset':[0,0,0], 'smearing':'mv', 'degauss_Ry':0.02,
        'nbnd':200, 'conv_thr_Ry':1e-9, 'electron_maxstep':200,
        'nstep':200, 'ion_dynamics':'bfgs', 'etot_conv_thr_Ry':1e-6,
        'forc_conv_thr_Ry_bohr':threshold, 'force_component_tolerance_eV_A':0.0005,
        'validation_max_force_norm_eV_A':0.001, 'max_seconds':39600,
        'nosym':True, 'noinv':False, 'tstress':True, 'tprnfor':True,
        'convergence_status':'starting numerical settings; cutoff/k/smearing/size convergence not certified',
        'energy_convention':'QE printed total energy with MV entropy-like contribution; smearing convergence required',
    }
    m = {'schema_version':1,'campaign_id':g['campaign_id'], 'protocol':protocol,
         'geometry_manifest':{'file':'geometry.json','sha256':sha(CAMPAIGN/'geometry.json')},
         'pseudo':{'file':str(pp.relative_to(CAMPAIGN)).replace('\\','/'),'sha256':sha(pp)},
         'case_order':g['case_order'], 'cases':{}}
    for cid in g['case_order']:
        c = g['cases'][cid]
        nat = c['n_atoms']
        assert nat == 108-c['q'] and 200 > math.ceil(3*nat/2)
        source = CAMPAIGN/c['structure_files']['poscar']
        assert sha(source)==c['structure_sha256']['poscar']
        text=f"""&CONTROL
  calculation = 'relax'
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
  nat = {nat}
  ntyp = 1
  nbnd = 200
  ecutwfc = 60
  ecutrho = 240
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
  nspin = 1
  tot_charge = 0.0
  nosym = .true.
  noinv = .false.
/
&ELECTRONS
  conv_thr = 1.0d-9
  electron_maxstep = 200
  scf_must_converge = .true.
  mixing_mode = 'plain'
  mixing_beta = 0.3
  mixing_ndim = 4
  diagonalization = 'david'
  diago_david_ndim = 2
  startingpot = 'atomic'
  startingwfc = 'atomic+random'
/
&IONS
  ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
Al 26.9815385 Al.pz-vbc.UPF

CELL_PARAMETERS angstrom
"""
        text+='\n'.join(' '.join(f'{v:.14f}' for v in row) for row in g['cell_A'])
        text+='\n\nATOMIC_POSITIONS angstrom\n'
        text+='\n'.join('Al '+' '.join(f'{v:.14f}' for v in row) for row in c['positions_A'])
        text+='\n\nK_POINTS automatic\n3 3 3 0 0 0\n'
        dst=CAMPAIGN/f'qe/inputs/{cid}/pw.in'
        put(dst,text)
        m['cases'][cid]={'input':dst.relative_to(CAMPAIGN).as_posix(),'sha256':sha(dst),
            'nat':nat,'q':c['q'],'valence_electrons':3*nat,
            'structure':c['structure_files']['poscar'],'structure_sha256':sha(source),
            'extxyz':c['structure_files']['extxyz'],'extxyz_sha256':c['structure_sha256']['extxyz'],
            'removed_parent_ids':c['removed_parent_ids']}
    put(CAMPAIGN/'qe_manifest.json',json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'cases':len(m['cases']),'manifest':str(CAMPAIGN/'qe_manifest.json'),
                      'status':'prepared_not_submitted','force_threshold_Ry_bohr':threshold}))

if __name__=='__main__':
    main()
