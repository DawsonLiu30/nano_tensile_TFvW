"""Document the prepared inputs and bounded interpretation of existing results."""
from pathlib import Path
import json
import shutil
import hashlib
import html

W = Path(__file__).resolve().parents[1]
R = W / 'outputs/bulk_k4_preparation_20261003'
old = W / 'outputs/luder_review_20261003'
(R / 'existing_analysis').mkdir(exist_ok=True)
for name in ['summary.json', 'nano_formation_and_paired_structure.csv', 'analysis_checks.json']:
    shutil.copy2(old / '02_analysis' / name, R / 'existing_analysis' / name)
for name in ['paired_displacements.png', 'nano_energy_and_length.png']:
    shutil.copy2(old / '03_figures' / name, R / 'existing_analysis' / name)
shutil.copy2(__file__, R / 'reproduce' / Path(__file__).name)
s = json.loads((R / 'existing_analysis/summary.json').read_text())
qe, of = s['nano_paired'][:2]
assert qe['method'] == 'QE' and of['method'] == 'DFTpy'
gap = 1000 * (qe['Ef_eV'] - of['Ef_eV'])
ratios = {v: 100 * (of[v] / qe[v] - 1) for v in ['vacancy_induced_u_max_A', 'vacancy_induced_u_rms_A', 'vacancy_induced_u_p95_A']}
qa = json.loads((R / 'preparation_QA.json').read_text())

readme = f'''Al vacancy：10/3 意見核對與五態 k4 準備
狀態：本機準備完成，未提交、未執行新 DFT。沒有遠端查詢或監控。

一、現在可以跟 Luder 說什麼
目前小柱 R15 的 QE 0V/1V 通過既定鬆弛驗收。形成能為 {qe['Ef_eV']:.6f} eV，OF pilot 為 {of['Ef_eV']:.6f} eV，差 {gap:.3f} meV。
加入空缺後，QE 軸長增加 {qe['delta_Lz_percent']:.4f}%，OF 減少 {abs(of['delta_Lz_percent']):.4f}%。
逐原子比較已鬆弛的 0V 與 1V，最大局部位移為 QE {qe['vacancy_induced_u_max_A']:.4f}、OF {of['vacancy_induced_u_max_A']:.4f} Å。OF 最大值高 {ratios['vacancy_induced_u_max_A']:.1f}%，RMS 高 {ratios['vacancy_induced_u_rms_A']:.1f}%，95 百分位高 {ratios['vacancy_induced_u_p95_A']:.1f}%。
這些是目前協定下的差異，還不能指定物理原因。QE 柱子的 k 點、smearing、真空與軸向週期仍待驗證；OF stress production validation 未完成。兩方法的 XC、偽勢及實際半徑也不同。
數據來源：existing_analysis/summary.json；原報告時間為 2026-10-03 16:40 台灣時間。本包沒有更新遠端狀態。

二、採用三層整理，但不畫成已證實的因果鏈
1. 理想 0V → 已鬆弛 0V：看完美柱本身如何改變。現有最大非仿射位移 QE 0.332857 Å、OF 0.124563 Å；尚需徑向/位置分布才能定位哪些表面原子主導，不能單憑最大值解釋形成能。
2. 已鬆弛 0V → 已鬆弛 1V：先把 0V 仿射映射到 1V 晶胞，依共同 parent ID 配對，取最小影像位移並扣掉共同平移。它扣除完美柱的基準，仍保留空缺引起的表面變化。軸長差另列。
3. 同方法的形成能：E_f = E_column(1V) − E_column(0V) + μ_bulk。QE 用 QE 的 μ，OF 用 OF 的 μ。

三、附件中需修正的四句話
「表面鬆弛拿掉了」→ 扣除完美柱基準，沒有消除所有表面效應。
「小柱結果推翻 bulk 的兩倍位移」→ 兩者的系統與觀察量不同。bulk 1NN 的 0.159/0.078 Å 原數字保留；不能據此推出 OF 過度塌陷造成錯誤結合。
「不是 bulk μ 的問題」→ 只知道將已測的 k3 μ 換成 k4 μ，R15 E_f 只差 −2.254 meV；不能排除所有原子庫系統誤差。
「k4 仍有各向異性就是真實空缺彈性」→ 仍需排除 k 點、有限週期影像與局部極小值影響。k4 的形變變化是診斷，不是孤立空缺物理的證明。
補充：R15 代表切割半徑 R/a0=1.5，不是 15 Å；QE 約 6.063 Å，OF 約 5.921 Å。相同的是 96 格點拓撲。

四、五態 k4 輸入
已生成 0V（108 原子）、1V（107）、1NN/2NN/5NN（各 106）。全部三方向 PBC，全晶胞 vc-relax，保留各自已歸檔 k3 終態作起點。
位置：inputs/<case>/pw.in；每個 case 自帶 pseudo/Al_PAW_PBE.UPF、initial.extxyz 及 POSCAR。
完整共同條件見 protocol.json，逐檔來源/hash 見 manifest.json，簡表見 five_case_review.csv。

保持：PBE/PAW；ecutwfc=58.7989154810 Ry（800 eV）；ecutrho=470.3913238484 Ry；MP 0.02 Ry；nbnd=200；nosym/noinv；4×4×4 無位移；SCF 1e−10 Ry；nstep=200。
保持原子門檻：QE forc_conv_thr=1.94469038114889e−5 Ry/bohr（0.0005 eV/Å 每分量）；另驗收力 norm≤0.001 eV/Å。能量停止門檻 1e−6 Ry。
晶胞：BFGS、零外壓、cell_dofree=all、press_conv_thr=0.5 kbar。最終驗收全應力六分量 |σij|≤0.05 GPa，不套用 nano 只查 σzz 的規則。
顯式 optimizer 設定：五態統一 trust_radius_min=1e−6 bohr，沿用成功的 1V 修復做法；fresh BFGS 與電子初態。這項設定須列入方法，不能聲稱相對舊 k3 生產協定只有 k 點改變。
時間/資源提案：ct56，每組 1 node、29 ranks、64 GiB；Slurm 72 h，QE max_seconds=255600。QE 時間計數不是精確的 Slurm 牆鐘倒數。五組可並行，不限一組；實際由排程配額決定。
記憶體依據：已有五態 k4 SCF 印出 >36.58–37.04 GB 估計；64 GiB 是提案，不是最大需求保證。提交後要查實際 peak/failcnt/ranks。

五、承諾輸出及界線
只有合格的 k4 relaxed endpoints 才能組成：
μ4 = E0(108)/108
Ef1 = E1(107) − 107 μ4
Ef2(r) = E2(r,106) − 106 μ4
Eb(r) = 2 Ef1 − Ef2(r)，r=1NN/2NN/5NN
D12 = Ef2(1NN) − Ef2(2NN)
另列 5NN−1NN 的形成能差、各态 ΔV、晶胞長寬/角度、相對 k3 起點的變形及原子位移。能量採一致的 QE MP smearing functional，不偷偷混入零溫外推能量。
這五態不認證六配對完整排序，尤其不能寫「5NN 是全域最穩定」。3NN 是有條件的第六態，未生成或提交。
下一步 k5 SCF 必須從新的合格 k4 幾何生成，五態共同換 k；核對 Ef/Eb/D12、力及全應力的變化。現在沒有 k4 終態，所以沒有提前生成冒名的 k5 輸入。
k5 單點不能取代 k5 晶胞鬆弛；若殘餘力或應力仍大，需再審查是否鬆弛。單次 k4/k5 接近也不證明尺寸/smearing/孤立空缺極限都通過。近零結合能的符號需精度足以分辨才可討論。

六、提交前最後檢查
本包為輸入審查版，沒有 sbatch 或新 runner；旧 campaign-ID 鎖定的 bulk runner、nano 的 axial-only inspector 不能直接拿來執行這五態。
確定提交後，須建立獨立來源快照與對應全應力 inspector，查即時 queue 避免重複，核對 QE binary/偽勢/MPI/配額，再部署與提交。不要改原 campaign 或 canonical。
本次本機查核：{qa['check_count']} 項通過，包含來源 hash、參數變更 allowlist、ASE 幾何與獨立 canonical 終態比對。這不等於科學收斂通過。
重驗：python reproduce/prepare_bulk_k4_review_20261003.py --verify-only --root <本包目錄>（需 ASE/NumPy）。

七、後續順序
先處理現有 R20 終態/停止原因（待下次使用者要求時單次查核），再審查 bulk 五態 k4；R15 配對 axial k/smearing 檢查与 OF stress 候選 production validation 為後续增量。R20 QE 未驗收前不談 QE 半徑趨勢。B0 bonus 放最後。
先前 cutoff 測試有來源，保留其結果及適用範圍；本意見不構成整套 cutoff 全面重跑的要求。
'''
(R / 'README_ZH.txt').write_text(readme, encoding='utf-8')
english = '''The QE small-column pristine and single-vacancy endpoints pass the present relaxation acceptance criteria. Under the current protocols, the vacancy formation energies are 0.747 eV (QE) and 0.598 eV (OF pilot). Their axial responses have opposite signs: +0.076% and −0.147%, respectively.

Using atom-matched relaxed-pristine references, the maximum vacancy-associated displacements are 0.134 Å (QE) and 0.145 Å (OF). OF has approximately 27% higher RMS and 28% higher 95th-percentile displacement. This analysis removes the pristine baseline but retains vacancy-induced surface response. It does not overturn the separate bulk displacement measurements or establish a cause for the energy difference.

The old bulk QE pair ordering remains unvalidated: several binding energies change by more than 0.2 eV between k3 and k4 at fixed geometry. Five k4 variable-cell inputs (0V, 1V, 1NN, 2NN and 5NN) are prepared for review, with all existing results preserved. They have not been submitted. Any subsequent k5 checks must use the newly accepted k4 geometries.

Both the nanocolumn sampling and the OF stress pipeline still require validation. The methods also use different exchange-correlation functionals and pseudopotentials, and their topology-matched columns have slightly different physical radii. The present differences should therefore be reported as observations, without assigning a mechanism yet.
'''
(R / 'LUDER_UPDATE_DRAFT_EN.txt').write_text(english, encoding='utf-8')
case_links = ''.join(f'<li><a href="inputs/{c}/pw.in">{c} — pw.in</a></li>' for c in ['0V','1V','2V_1NN_D110','2V_2NN_D100','2V_5NN_D310'])
page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>五態 k4 輸入與評論核對</title>
<style>body{{max-width:1080px;margin:40px auto;padding:0 28px;color:#23313d;font:17px/1.65 'Microsoft JhengHei',sans-serif}} h1{{font-size:30px}} h2{{font-size:23px;margin-top:34px}} table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccd5df;text-align:left}} .status{{background:#eef4fa;border-left:5px solid #2476a7;padding:16px}}img{{max-width:100%}}pre{{white-space:pre-wrap;font:inherit}}a{{color:#135d92}}</style>
<h1>把差異拆清楚，再補關鍵驗證</h1><p class="status">五態輸入已準備，{qa['check_count']} 項本機查核通過；尚未提交。<br>本頁沿用 10/3 16:40 的歸檔資料，沒有查詢新的作業狀態。</p>
<p>這份意見的主要方向可以採用：先看無空缺的柱子，再看空缺額外造成的變化，最後比較形成能。它們是三組互相對照的證據，還不是已證明的因果鏈。</p>
<h2>小柱的差異已經清楚，原因還沒有定案</h2>
<table><tr><th>R15 指標</th><th>QE</th><th>OF pilot</th></tr>
<tr><td>單空缺形成能（eV）</td><td>{qe['Ef_eV']:.6f}</td><td>{of['Ef_eV']:.6f}</td></tr>
<tr><td>空缺造成的軸長變化</td><td>{qe['delta_Lz_percent']:+.4f}%</td><td>{of['delta_Lz_percent']:+.4f}%</td></tr>
<tr><td>配對位移最大值（Å）</td><td>{qe['vacancy_induced_u_max_A']:.4f}</td><td>{of['vacancy_induced_u_max_A']:.4f}</td></tr>
<tr><td>配對位移 RMS（Å）</td><td>{qe['vacancy_induced_u_rms_A']:.5f}</td><td>{of['vacancy_induced_u_rms_A']:.5f}</td></tr></table>
<p>位移以同尺寸、已鬆弛的無空缺柱為基準；扣掉均勻晶胞形變與共同平移。空缺引起的表面反應仍包含其中。原有 bulk「約兩倍最大位移」數字保留，但不能把它當成錯誤結合能的原因。</p>
<img src="existing_analysis/paired_displacements.png" alt="以已鬆弛無空缺柱為基準的逐原子位移">
<p>兩方法的 XC、偽勢、實際半徑不同；QE 柱子取樣與 OF stress 的驗證也未完成。因此，現在能說的是「能量和軸向反應都不同」，還不能說哪個機制已被證明。</p>
<h2>五態 k4 的審查內容</h2><ul>{case_links}</ul>
<p>保留 800 eV、PAW/PBE 與原門檻。三方向 PBC、全晶胞鬆弛；從各自舊終態出發。提案為每組 29 ranks／64 GiB、Slurm 72 小時，五組可並行。64 GiB 仍需實際記憶體查核。</p>
<p>可比較單空缺形成能、三種配對結合能、1NN−2NN 與 5NN−1NN；不能認證六配對完整排序。後續 k5 輸入須等新的 k4 終態通過驗收後生成。</p>
<p><a href="protocol.json">完整協定</a> · <a href="five_case_review.csv">五態條件表</a> · <a href="manifest.json">來源與 hashes</a> · <a href="preparation_QA.json">117 項查核</a></p>
<h2>給 Luder 的簡短更新草稿</h2><pre lang="en">{html.escape(english)}</pre>
<p><a href="README_ZH.txt">完整中文說明與承諾輸出</a> · <a href="LUDER_UPDATE_DRAFT_EN.txt">英文草稿</a> · <a href="existing_analysis/summary.json">本次引用的原分析數據</a></p></html>'''
(R / 'REVIEW_ZH.html').write_text(page, encoding='utf-8')
# A package inventory is separate from the immutable input/source manifest.
files = {p.relative_to(R).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.rglob('*')) if p.is_file() and p.name != 'package_sha256.json'}
(R / 'package_sha256.json').write_text(json.dumps(files, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'gap_meV': gap, 'OF_relative_excess_percent': ratios, 'documented_files': len(files)}, ensure_ascii=False))
