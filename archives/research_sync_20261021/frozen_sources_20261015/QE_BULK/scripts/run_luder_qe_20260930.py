"""Frozen-input QE execution and separate SCF / axial-vc-relax inspection."""
from pathlib import Path
import argparse, hashlib, json, math, os, re, shutil, subprocess, sys, traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

HA=27.211386245988
BOHR=0.5291772105638411
GPA=29421.02648438959
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def require(ok,msg):
    if not ok: raise ValueError(msg)
def verify(campaign):
    m=json.loads((campaign/'qe_manifest.json').read_text())
    require(sha(campaign/m['pseudo_file'])==m['pseudo_sha256'],'PP hash mismatch')
    for c in m['cases']:
        p=campaign/c['input'];require(p.resolve().is_relative_to(campaign.resolve()),'Input path escaped')
        require(sha(p)==c['input_sha256'],'Input changed: '+c['id'])
    return m
def inspect(attempt,c):
    result={'case':c['id'],'accepted':False,'calculation':c['calculation'],'scope':c['kind']}
    try:
        prov=json.loads((attempt/'provenance.json').read_text())
        require(sha(attempt/'pw.in')==c['input_sha256'],'Input hash changed')
        require(sha(attempt/'pseudo/Al_PAW_PBE.UPF')==prov['pseudo_sha256'],'PP hash changed')
        text=(attempt/'pw.out').read_text(errors='replace')
        require('JOB DONE.' in text,'QE did not report JOB DONE')
        require(not re.search(r'Error in routine|convergence NOT achieved|bfgs failed',text,re.I),'QE error/unconverged message')
        p=attempt/'scratch'/(c['prefix']+'.save')/'data-file-schema.xml'
        out=ET.parse(p).getroot().find('output')
        require(out is not None,'Missing XML output')
        conv=out.find('convergence_info/scf_conv')
        require(conv.findtext('convergence_achieved')=='true','Final SCF not converged')
        accuracy_Ry=2*float(conv.findtext('scf_error'))
        require(accuracy_Ry<=1.01e-10,'Final SCF error exceeds target')
        ast=out.find('atomic_structure');nat=int(ast.attrib['nat'])
        require(nat==c['nat'],'Atom count changed')
        require(abs(float(out.findtext('band_structure/nelec'))-3*nat)<1e-8,'Wrong electron count')
        cell=[[float(v)*BOHR for v in ast.find('cell/'+key).text.split()] for key in ['a1','a2','a3']]
        positions=[[float(v)*BOHR for v in a.text.split()] for a in ast.findall('atomic_positions/atom')]
        forces=[float(v)*HA/BOHR for v in out.findtext('forces').split()]
        stress=[float(v)*GPA for v in out.findtext('stress').split()]
        require(len(forces)==3*nat and len(positions)==nat and len(stress)==9,'Incomplete endpoint arrays')
        energy=float(out.findtext('total_energy/etot'))*HA
        require(all(math.isfinite(v) for v in forces+stress+[energy]),'Nonfinite result')
        fmax=max(math.sqrt(sum(x*x for x in forces[i:i+3])) for i in range(0,len(forces),3))
        comp=max(abs(x) for x in forces)
        if c['calculation']=='scf':
            require(max(abs(cell[i][j]-c['cell_A'][i][j]) for i in range(3) for j in range(3))<2e-7,'Fixed SCF cell changed')
            # Atomic positions may be translated by lattice vectors in XML.
            # Compare their fractional displacement modulo integers.
            a,b,d=cell
            cross=lambda u,v:[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
            dot=lambda u,v:sum(x*y for x,y in zip(u,v))
            det=dot(a,cross(b,d));require(det>0,'Invalid cell volume')
            reciprocal=[cross(b,d),cross(d,a),cross(a,b)]
            for now,old in zip(positions,c['positions_A']):
                delta=[x-y for x,y in zip(now,old)]
                frac=[dot(delta,v)/det for v in reciprocal]
                require(max(abs(v-round(v)) for v in frac)<2e-7,'Fixed SCF position changed')
        else:
            require(out.findtext('convergence_info/opt_conv/convergence_achieved')=='true','Ionic/cell optimizer not converged')
            require('Final scf calculation at the relaxed structure' in text,'No final relaxed-cell SCF')
            require(re.search(r'bfgs converged',text,re.I) is not None,'No BFGS convergence marker')
            require(fmax<=.001 and comp<=.0005,'Final atomic force gate failed')
            require(abs(stress[8])<=.05,'Final axial stress gate failed')
            require(cell[2][2]>0 and max(abs(cell[i][j]-c['cell_A'][i][j]) for i in range(3) for j in range(3) if (i,j)!=(2,2))<2e-7,'Transverse box/tilt changed')
        result.update(accepted=True,status='accepted_scf_diagnostic' if c['calculation']=='scf' else 'accepted_axial_relaxation',energy_eV=energy,n_atoms=nat,cell_A=cell,positions_A=positions,forces_eV_A=[forces[i:i+3] for i in range(0,len(forces),3)],fmax_eV_A=fmax,max_force_component_eV_A=comp,stress_QE_GPa=[stress[i:i+3] for i in range(0,9,3)],final_scf_accuracy_Ry=accuracy_Ry,hashes={'pw.in':sha(attempt/'pw.in'),'pw.out':sha(attempt/'pw.out'),'xml':sha(p)},limitations='Fixed-geometry SCFs test numerical sensitivity only; nano results are unconverged with respect to vacuum/period/grid/k-point sampling until tested.')
    except Exception as exc: result.update(status='not_accepted',error=repr(exc))
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','run','inspect']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--root',type=Path);p.add_argument('--attempt',type=Path)
    args=p.parse_args();m=verify(args.campaign)
    if args.mode=='verify':
        print(json.dumps({'verified_cases':len(m['cases']),'case_ids':[c['id'] for c in m['cases']]}));return 0
    c=m['cases'][args.index]
    if args.mode=='inspect':
        r=inspect(args.attempt,c);print(json.dumps(r,indent=2));return 0 if r['accepted'] else 1
    require(sys.platform=='linux' and os.environ.get('SLURM_JOB_ID'),'Execution requires Slurm Linux')
    require(sha(m['binary'])==m['binary_sha256'],'QE binary mismatch')
    root=args.root.resolve();require(not any((x/'.git').exists() for x in [root,*root.parents]),'Output inside Git')
    attempt=root/c['id']/'attempt_001';attempt.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.campaign/c['input'],attempt/'pw.in');(attempt/'pseudo').mkdir();shutil.copyfile(args.campaign/m['pseudo_file'],attempt/'pseudo/Al_PAW_PBE.UPF')
    shutil.copyfile(args.campaign/'qe_manifest.json',attempt/'qe_manifest.json')
    dump(attempt/'provenance.json',{'case':c,'pseudo_sha256':m['pseudo_sha256'],'binary_sha256':m['binary_sha256'],'runner_sha256':sha(__file__),'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),'started_utc':datetime.now(timezone.utc).isoformat()})
    cmd=['srun','--mpi=pmi2','-n','29',m['binary'],'-in','pw.in']
    with (attempt/'pw.out').open('w') as out,(attempt/'pw.stderr').open('w') as err:
        proc=subprocess.run(cmd,cwd=attempt,stdout=out,stderr=err)
    result=inspect(attempt,c);result['returncode']=proc.returncode
    if proc.returncode:result.update(accepted=False,status='process_failed')
    result['finished_utc']=datetime.now(timezone.utc).isoformat();dump(attempt/'result.json',result)
    print(json.dumps({'case':c['id'],'accepted':result['accepted'],'status':result['status'],'error':result.get('error')}));return 0 if result['accepted'] else 1
if __name__=='__main__':sys.exit(main())
