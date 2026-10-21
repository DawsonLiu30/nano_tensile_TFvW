"""Archive the reviewed result and register its frozen workflow. No DFT or remote I/O."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--apply', action='store_true')
A = P.parse_args()
assert A.apply, 'Explicit --apply required after coordinating final reviews.'
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPO = Path('C:/OFDFT/nano_tensile_TFvW')
SHARE = Path('C:/OFDFT/NAS_SHARE')
ROOT = SHARE / 'LUDER_RESULTS_20261019'
PACK = ROOT / 'R13_AXIAL_CONTINUATION_RESULTS'
ANALYSIS = PACK / 'ANALYSIS'
FLOW = REPO / 'workflows/al_nano_R13_axial_continue_20261018_v2'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,t):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(t,encoding='utf-8')

release = read(HERE/'archive_release.json')
assert release['approved_to_finalize'] is True
audit = read(WORK/'independent_audit/audit_summary.json')
qa = read(WORK/'independent_audit/QA.json')
assert audit['accepted_by_saved_data_audit'] and qa['passed']==qa['total']>=591
pair = read(WORK/'pair_analysis/formation_analysis.json')
struct = read(WORK/'pair_analysis/structure_pairs.json')
assert pair['accepted_pairs']==['R11','R13','R15','R17','R20']
assert all(c['passed'] for c in pair['checks'])
assert all(c['passed'] for c in read(WORK/'pair_analysis/PAIR_QA.json')['checks'])
assert release['pair_review_complete'] is True
result = read(PACK/'R13_0V/result.json')
assert result['accepted'] and result['fmax_eV_A']<=.001 and result['filter_fmax_eV_A']<=.001
assert abs(result['stress_GPa'][2][2])<=.05
assert sha(FLOW/'REVISION_MANIFEST.json')=='0873dc84e7e796fea4f743ab37e1f2dcc1a34dd86bee2cf9fef70cc7e3970bbf'
refs = read(HERE/'SOURCE_REFERENCES.prepared.json')
assert all(sha(Path(r['absolute_local_path']))==r['sha256'] for r in refs['references'])
assert all(sha(Path(r['path']))==r['expected_sha256'] for r in refs['immutable_workflow_file_checks'])

# The two review agents own their final analysis folders. Copy only missing identical files;
# do not replace a distinct review result silently.
for source, dest in [(WORK/'independent_audit',ANALYSIS/'INDEPENDENT_R13'),
                     (WORK/'pair_analysis',ROOT/'ANALYSIS/PAIR')]:
    for p in source.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts: continue
        q=dest/p.relative_to(source)
        q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists(): assert sha(q)==sha(p), ('Different existing review file',q)
        else: shutil.copy2(p,q)

scripts=ANALYSIS/'ARCHIVE_SCRIPTS'
for p in [HERE/'prepare_references.py',HERE/'finalize_archive.py',HERE/'archive_release.json',
          WORK/'post_download.py',WORK/'post_download.sh',WORK/'probe.sh']:
    scripts.mkdir(parents=True,exist_ok=True);shutil.copy2(p,scripts/p.name)
figs=ANALYSIS/'FIGURES';figs.mkdir(parents=True,exist_ok=True)
shutil.copy2(WORK/'plot_relaxation.py',figs/'plot_relaxation.py')
write(ROOT/'SOURCE_REFERENCES.json',refs)

now=datetime.now(timezone.utc).isoformat()
rs=next(x for x in struct['rows'] if x['radius']=='R13')
evidence={
 'updated_utc':now,'job_id':'2255218','case':'R13_0V','attempt':result['attempt'],
 'status':'accepted_numerical_endpoint_for_this_case','slurm_state':'COMPLETED',
 'finished_Taiwan':'2026-10-18T22:21:11+08:00','elapsed_slurm':'04:01:12',
 'physical_endpoint_accepted':True,'scientific_scope':audit['scope'],
 'result_archive':str(PACK),'result_sha256':sha(PACK/'R13_0V/result.json'),
 'workflow':str(FLOW),'revision_manifest_sha256':sha(FLOW/'REVISION_MANIFEST.json'),
 'checkpoint_bundle_sha256':result['artifact_sha256']['checkpoint_bundle.npz'],
 'optimizer_state_sha256':result['artifact_sha256']['optimizer_state.json'],
 'energy_eV':result['energy_eV'],'atomic_fmax_eV_A':result['fmax_eV_A'],
 'filter_fmax_eV_A':result['filter_fmax_eV_A'],'sigma_zz_GPa':result['stress_GPa'][2][2],
 'Lz_A':result['cell_lengths_A'][2],'ionic_steps':result['ionic_steps'],
 'electronic_calls':result['electronic_calls'],'electronic_attempt_summary':result['electronic_attempt_summary'],
 'acceptance':{'atomic_force_eV_A':.001,'filter_force_eV_A':.001,'abs_sigma_zz_GPa':.05},
 'saved_data_QA':{'passed':qa['passed'],'total':qa['total'],'archive':str(ANALYSIS/'INDEPENDENT_R13')},
 'reuse_scope':'Successful R13_0V case using saved coherent checkpoint, z-only BFGSLineSearch and electronic TN. Reuse the frozen scientific core with a reviewed new case manifest; not a guarantee for all columns or a certified global minimum.',
 'frozen_source_policy':'Original manifests and snapshot contents remain unchanged; this evidence record supplies the later acceptance status.',
 'electronic_method':'TN; econv=1e-11 Ha/atom; ncheck=3',
 'ionic_method':'ASE BFGSLineSearch; z-only FrechetCellFilter; maxstep trial increment 0.02 (not an outer-step displacement cap)',
 'fresh_Hessian_at_continuation_start':True,'paired_optimizer_checkpoint_saved':True,
 'R13_pair_review':release['R13_pair_review'],
}
write(REPO/'docs/validated_workflow_evidence/R13_axial_continue_2255218_20261019.json',evidence)
write(ROOT/'WORKFLOW_ACCEPTANCE_2255218.json',evidence)
shutil.copy2(HERE/'HOW_TO_REUSE_ZH.txt',ROOT/'HOW_TO_REUSE_ZH.txt')
shutil.copy2(HERE/'HOW_TO_REUSE_ZH.txt',REPO/'docs/validated_workflow_evidence/HOW_TO_REUSE_R13_ZH.txt')
previous=read(SHARE/'LUDER_RESULTS_20261018/CURRENT_STATUS.json')
status={
 'updated_utc':now,'review_date_Taiwan':'2026-10-19','data_folder':str(ROOT),
 'remote_queue_at_read':[],'remote_read_utc':read(PACK/'ACCOUNTING/probe.json')['utc'],
 'latest_R13_job_id':'2255218','latest_R13_job_status':'COMPLETED and independently accepted',
 'R13_axial_continuation':evidence,'OF_accepted_endpoints':10,'OF_total_endpoints':10,
 'OF_complete_endpoint_pairs':['R11','R13','R15','R17','R20'],
 'OF_pair_analysis':str(ROOT/'ANALYSIS/PAIR'),
 'R13_formation_energy_eV':release['R13_pair_review']['formation_energy_eV'],
 'OF_size_interpretation':release['five_size_interpretation'],
 'QE_k9_unchanged':previous['QE_k9'],'new_DFT_submissions_this_review':[],
 'new_cancellations_this_review':[],'automatic_monitoring':False,'NAS_uploaded':False,
 'email_sent':False,'historical_packages_modified':False,
 'old_canonical_tables_modified':False,
 'next_scope':'Interpret the five OF endpoint pairs with structure changes. QE sampling/representative re-relaxation remains separate, with six new-size endpoints unreleased.'}
write(ROOT/'CURRENT_STATUS.json',status)

memory='''<!-- BEGIN R13_ACCEPTED_RESULTS_20261019 -->
## 2026-10-19 R13 續算驗收完成（最新）

- 2255218於10/18 22:21:11台灣時間正常完成（4h01m12s）；本輪10/19單次SSH查核佇列空。46個完整離子步、185次電子原生成功、無HEG重試；Fmax=.000976534、filter=.000976680eV/Å、σzz=.000040852GPa，591/591保存資料檢查通過。維持.001力與.05GPa軸應力標準，R13_0V正式升級為本模型下數值合格端點。
- 已實際成功的收尾為z-only FrechetCellFilter+BFGSLineSearch（試探增量.02）配合電子TN/econv1e−11/ncheck3；不是FIRE。從2254539 step12 coherent bundle續算，fresh Hessian，新optimizer_state與step46結構／密度bundle一致。永久凍結入口workflows/al_nano_R13_axial_continue_20261018_v2，manifest0873dc84e7e796fea4f743ab37e1f2dcc1a34dd86bee2cf9fef70cc7e3970bbf；原source不改，後驗收證據docs/validated_workflow_evidence/R13_axial_continue_2255218_20261019.json。成功範圍限此R13案例，不把所有柱或全域最低能保證化。
- 此續算能量下降.950994eV，終態Lz=13.739937Å；原子有明顯整體重排。R13的0V/1V相對各自ideal RMS約.75887/.76305Å，但兩個終態去仿射配對RMS=.029942、最大=.105318Å，缺格parent_site_id33仍保留。應先討論重排後結構與空缺造成的增量反應，結晶相分類仍待專門確認。
- 五個OF尺寸已有10/10數值合格端點，R13同方法μAl=-62.943857545eV，Ef=.616366216eV。五組依R11/R13/R15/R17/R20為.424814871/.616366216/.598150394/.628738683/.631826061eV，並非隨尺寸嚴格單調；R11亦有廣泛空缺誘發重排。R13_0V新TN較嚴與原1V LBFGS電子精度差異有列來源；網格／真空／取樣及QE一致性仍另驗證。
- 新raw42檔與遠端SHA一致，資料、結構、逐步分析、五配對能量分項／位移、圖與腳本在C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261019。10/17、10/18舊包與舊canonical表原封保留，大密度／PP／父bundle按hash引用。不新增DFT、不定時監控、未上傳NAS或寄信。QE k9仍為固定幾何診斷（力未過），六個新尺寸未放行。
<!-- END R13_ACCEPTED_RESULTS_20261019 -->

'''
memory=memory.replace('591/591',f"{qa['passed']}/{qa['total']}")
backup=ROOT/'RESEARCH_STATUS_PREVIOUS'
for rel in ['AGENTS.md','docs/VALIDATED_WORKFLOWS.md','docs/validated_workflows.json']:
    p=REPO/rel;q=backup/rel;q.parent.mkdir(parents=True,exist_ok=True)
    if not q.exists():shutil.copy2(p,q)
for rel in ['AGENTS.md','docs/VALIDATED_WORKFLOWS.md']:
    p=REPO/rel;s=p.read_text(encoding='utf-8-sig')
    if '<!-- BEGIN R13_ACCEPTED_RESULTS_20261019 -->' in s:
        s=s.split('<!-- END R13_ACCEPTED_RESULTS_20261019 -->',1)[1].lstrip('\n')
    txt(p,memory+s)
reg=read(REPO/'docs/validated_workflows.json')
reg['updated']='2026-10-19'
reg['latest_accepted_workflow_key']='r13_axial_continuation_reviewed_entrypoint'
reg['reuse_guidance']='For compatible atomic-plus-axial recovery, consult the accepted R13 BFGSLineSearch/TN case and its HOW_TO_REUSE_R13_ZH.txt first. Existing top-level smallstep entry is a historical failed case; do not treat it as the current preferred successful method.'
entry=reg['r13_axial_continuation_reviewed_entrypoint']
entry['evidence']['status']='completed_independently_accepted_R13_0V'
entry['evidence']['physical_endpoint_accepted']=True
entry['evidence']['scope']=evidence['reuse_scope']
entry['evidence']['latest_result']=evidence
entry['evidence']['acceptance_evidence_file']=str(REPO/'docs/validated_workflow_evidence/R13_axial_continue_2255218_20261019.json')
write(REPO/'docs/validated_workflows.json',reg)

readme=f'''2026-10-19 最新計算資料與分析

R13 完整柱已通過原定門檻。
作業2255218正常完成，用時4小時1分12秒；完成46個離子步、185次電子計算。
最大原子力0.000976534 eV/Å，軸向應力0.000040852 GPa。
獨立核查591/591項通過。這些是現有資料的檢查項目，不是另外跑591組DFT。

本次真正補上的結果
OF現在有R11、R13、R15、R17、R20共五組完整的0V/1V數值合格端點。
使用各自OF總能與OF bulk μAl計算，R13空缺形成能為0.616366216 eV。
五組形成能為：R11 0.424814871、R13 0.616366216、R15 0.598150394、R17 0.628738683、R20 0.631826061 eV。

如何解讀
{release['summary_ZH']}
{release['pair_scope_ZH']}
R13_0V電子TN比舊R13_1V電子LBFGS使用更嚴精度；方法與來源差異已逐項記錄。
QE最新仍是R15 k9固定幾何診斷；其力還高於鬆弛標準，六個新尺寸計算未放行。

資料分類（本資料夾內）
R13_AXIAL_CONTINUATION_RESULTS/R13_0V：原始result、能量分項、log、結構、trajectory、力、密度/結構bundle與optimizer state。
  final.extxyz／final.vasp：最終結構；relax.traj：完整47幀含初始。
  result.json：最終能量、力、應力與計算驗收。
  inputs/：實際使用的程式與輸入副本，保持原檔。
R13_AXIAL_CONTINUATION_RESULTS/ACCOUNTING：Slurm完成狀態與stdout/stderr。
R13_AXIAL_CONTINUATION_RESULTS/ANALYSIS/INDEPENDENT_R13：591項查核、逐步診斷與可重跑腳本。
ANALYSIS/PAIR：五尺寸能量／分項／結構表、逐原子配對位移與分析腳本。
R13_AXIAL_CONTINUATION_RESULTS/ANALYSIS/FIGURES：本次收斂圖、繪圖數據與腳本。
CURRENT_STATUS.json：本輪完整狀態，QE沿用10/18已核對結果。
WORKFLOW_ACCEPTANCE_2255218.json：本次成功流程及版本證據。
HOW_TO_REUSE_ZH.txt：為何這次成功、與歷史方法的差異，以及下次如何複用。
SOURCE_REFERENCES.json：10/17、10/18與Al_QE_DFTpy_DATA來源位置和hash。

保存與使用
這是增量資料包；請與既有LUDER_RESULTS_20261017、LUDER_RESULTS_20261018及Al_QE_DFTpy_DATA並列保存。
先前資料與凍結腳本沒有覆寫。原有大密度、PP及父checkpoint以位置+SHA引用，避免重複占空間。
新端點完整電子密度已包含在checkpoint_bundle.npz；遠端final_density/checkpoint_density兩份同內容檔未重複下載。
本輪沒有提交新DFT、定時監控、上傳NAS或寄信。
'''
readme=readme.replace('591/591',f"{qa['passed']}/{qa['total']}").replace('591組DFT',f"{qa['total']}組DFT").replace('591項',f"{qa['total']}項")
txt(ROOT/'先讀我_2255218已完成.txt',readme)
txt(PACK/'先讀我_2255218結果.txt',readme)
txt(ANALYSIS/'REPRODUCE.txt','''分析重建（只讀現有資料，不執行DFT）
INDEPENDENT_R13/audit_saved_results.py 與日期包根目錄ANALYSIS/PAIR的腳本使用Python/NumPy/ASE；各自README保留當次執行參數與來源。
FIGURES/plot_relaxation.py --data <R13_0V目錄> --output <圖片輸出目錄> 重新繪製收斂圖。
目前可用WSL Python：/var/tmp/al-defects-runtime-20260907/env/bin/python。
如果搬到NAS，應保持三個日期資料包及Al_QE_DFTpy_DATA的並列關係，按各腳本README調整root參數；絕對路徑保留原始計算來源。
SOURCE_REFERENCES.json位於此日期包根目錄。原始raw的遠端hash在REMOTE_FILE_MANIFEST.json。
''')

# Final seals exclude their own outputs, avoiding self-reference.
excluded={'ARCHIVE_MANIFEST.json','ARCHIVE_QA.json','SHA256SUMS.txt'}
entries=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name not in excluded:
        entries.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'updated_utc':now,'files':entries,'excluded_generated_files':sorted(excluded),
          'historical_sources':'SOURCE_REFERENCES.json','scope':'New 20261019 incremental package only; original prior packages untouched.'}
write(ROOT/'ARCHIVE_MANIFEST.json',manifest)
txt(ROOT/'SHA256SUMS.txt',''.join(f"{e['sha256']}  {e['path']}\n" for e in entries))
verified=[sha(ROOT/e['path'])==e['sha256'] for e in entries]
assert all(verified)
write(ROOT/'ARCHIVE_QA.json',{'files_passed':sum(verified),'files_total':len(entries),
 'manifest_sha256':sha(ROOT/'ARCHIVE_MANIFEST.json'),'reference_files_checked':len(refs['references']),
 'frozen_workflow_files_checked':len(refs['immutable_workflow_file_checks']),
 'raw_source_files_modified':False,'new_DFT':False})
print(json.dumps({'archive':str(ROOT),'files_verified':len(entries),'workflow_manifest':evidence['revision_manifest_sha256']},ensure_ascii=False))
