# Al bulk 空缺：2026-09-25 晶胞鬆弛修訂

**9/28 00:44 修復進度（優先於以下舊快照）：** 原兩批已終結，DFTpy8/8、QE7/8通過；缺QE1V。使用者已授權查因、修正及重新提交直到通過。針對BFGS小下降步触發history連續reset，已於00:42:37提交單一1V job **2162428**，00:43:08開始RUNNING。新campaign `al_bulk_vacancy_1v_repair_20260928`，明設trust_radius_min=1e-6 bohr、conv_thr加嚴1e-10 Ry，其他force/stress/物理設定保留。只用專用repair wrapper驗收，48hSlurm/47hQE、ct56、29ranks64GiB。00:52共享SSH連線失效，已請使用者重連；批次計算不依賴這個查詢連線。未恢復定時Codex監控、不能把已重送當已通過。

**最新接續（9/26 18:28）：** 正式array **2159026**，1V／1NN／3NN／5NN四組皆RUNNING；48h Slurm／47h QE、0-3%4、29ranks/64GiB。使用新campaign `al_bulk_vacancy_vcrelax_continue_20260926` 及snapshot `al_bulk_vacancy_vcrelax_continue_20260926_20260926T095326Z`，對應runner `run_bulk_vacancy_qe_continue_20260926.py`。原子/晶胞皆從舊最後完整SCF幾何以新BFGS和新電子初態出發，不讀未完整trial step，不改科學收斂值。下文舊job/並行/時間描述為歷史，完整此次回執歸檔至 `C:/OFDFT/RESEARCH_UPDATES/AL_BULK_QE_CONTINUE_20260926`。

## 使用者指定的方法

最新更正是 **QE 保留 PAW／PBE；DFTpy 保留 LDA／local recpot**。較早「QE 保持 LDA」的答覆已被撤回。本次不統一兩套 XC 或偽勢。

| 項目 | QE | DFTpy |
|---|---|---|
| 方法 | PBE、Al_PAW_PBE.UPF | LDA、al.lda.recpot，Ts=0.9 T_TF+0.1 T_vW |
| cutoff／網格 | ecutwfc=58.7989154810 Ry≈800 eV；ecutrho=470.3913238484 Ry | 固定78³網格；初態spacing=0.155378769 Å，隨晶胞鬆弛變化 |
| k 點／展寬 | 3³、offset 0；MP 0.02 Ry | 實空間密度最佳化 |
| 電子收斂 | 1e−8 Ry、最多300步 | 1e−9 Ha/atom、ncheck=3、最多1000步 |
| 原子鬆弛 | BFGS；力分量0.0005 eV/Å；另驗證力向量≤0.001 eV/Å | LBFGS；真實原子力與cell filter generalized force皆≤0.001 eV/Å |
| 晶胞 | vc-relax、BFGS、cell_dofree=all | FrechetCellFilter，全strain自由度 |
| 目標壓力 | 0；press_conv_thr=0.5 kbar；另驗證完整殘餘stress tensor | 0；完整stress最大絕對分量≤0.05 GPa |

QE 數值來源是已找回的 `/home/dawson666/qe_divacancy_vcrelax_conv3x3x3_rscan_20260616/pair_scan/pair_01_r2p8566A/pristine_vcrelax/vc-relax.in`。該輸入確實使用 800 eV、8倍density cutoff、3³k點與PAW/PBE；800 eV也在PPT bulk EOS測試範圍內。新批次保留9月24日較嚴格的力／能量停止值、200bands、mixing_ndim=4與步數／時間上限，差異在manifest明列。

DFTpy 78³ 的直接依據：

- `C:/OFDFT/RESEARCH_UPDATES/VACANCY_CONDENSATION_20260915/analysis/local_calculations/summary.json`：同LDA/local、0.9/0.1模型，60³→78³，1NN與2NN結合能分別改−0.1705與−0.0759 meV。
- `C:/OFDFT/RESEARCH_UPDATES/VALIDATION_20260916/analysis/eos/eos_analysis.json`：78³與96³、各7體積點，擬合a₀差4.97×10⁻⁶ Å、B₀差0.000627 GPa。
- `C:/OFDFT/RESEARCH_UPDATES/LUDER_RESPONSE_20260917/controlled/grid_check.json`：數個雙空缺在78³終態幾何上的96³單點，結合能最大差0.00736 meV。這不代表所有新vc-relax軌跡均已做過96³重鬆弛。

OF density grid與QE orbital ecutwfc不具有相同物理意義；不以相同名目eV值宣稱相同精度。PPT已測條件保留其適用範圍，也不把primitive單原子EOS的40³k點直接搬到108格點超晶胞。此輪尚未聲稱空缺k點、有限尺寸或所有終態stress的完整收斂驗證。

## 八組共同初態

3×3×3 conventional FCC，108格點，三方向PBC；起始a=4.039848 Å、邊長12.119544 Å。這是已存在PBE參考的起始尺寸，**不是新DFTpy或QE的平衡a₀結果**。兩方法均从理想結構重新鬆弛到各自的平衡晶胞，不混用先前固定晶胞能量。

| 構型 | 原子數 | 移除的0-based parent IDs | 起始MIC距離（Å） |
|---|---:|---|---:|
| 0V |108|無|—|
| 1V |107|43|—|
| 2V 1NN [110] |106|43,64|2.856604|
| 2V 2NN [100] |106|43,79|4.039848|
| 2V 3NN [211] |106|43,82|4.947783|
| 2V 4NN [110] |106|43,85|5.713208|
| 2V 5NN [310] |106|43,100|6.387561|
| 2V 6NN [111] |106|43,86|6.997222|

第一空缺在fractional(0.5,0.5,0.5)。5NN有兩個等距最近週期影像，保留為影像診斷；任何構型都不稱作孤立／無限分離雙空缺。12 Å滿足指定初始模型大小，本身不構成影像效應已收斂的證明。

每個計算保留cell矩陣、體積、力、stress、電子收斂與atom-parent對應，區分整體晶胞應變與非仿射原子位移。同方法形成能用E(q)−(108−q)E(0)/108，結合能用2E(1)−E(2)−E(0)，限同協定零壓結果。跨方法比較這些能量差與幾何；不可直接相減PAW與local偽勢的原始總能量，也不可把全部差異歸因於TFvW。

## 排程與舊資料

2026-09-26 操作規則更新：後續**新部署**的 bulk QE vc-relax 使用 Slurm **48 小時**，runner 的 QE `max_seconds` 預設及上限為 **169200 秒（47 小時）**；sbatch 明確傳入此值，預留一小時收尾。可用 `--max-seconds` 指定較短的正值。這僅改執行時間預算，不改力、應力、SCF、步數、XC、偽勢、cutoff 或 k 點。

既有 immutable 遠端快照及已提交作業仍為 Slurm 12 小時／QE 39600 秒（11 小時），本機修改不會延長它們。原 campaign manifest／來源輸入保留原 hash 與 39600 秒作為歷史來源；新 attempt 的 `pw.in` 與 `attempt.json.max_seconds` 記錄實際覆寫值，runtime input hash 隨之改變，但科學輸入 hash 不變。`attempt.json.protocol.max_seconds` 是原 manifest 的來源值，不能用它判讀新 attempt 的實際執行預算。須重新凍結、部署這次 runner 與 sbatch 才生效，並先查現有作業避免重複提交。

舊固定晶胞array2151591：0V與1NN已完成，1V遇BFGS trust-radius failure；輸出全保留。個別hold請求失敗後，已用 `scancel --state=PENDING 2151591` 成功取消尚未開始的3–7，並保存遠端回執，避免繼續做已被替代的固定晶胞計算。

新QE使用已實測可執行的QE7.5/IntelMPI profile：ct56、1node、29ranks、64GiB、每case12h、array一次1個。先前1NN僅固定晶胞已耗時約2.5h，因此不把這批完整QE vc-relax強塞進2h的ctest。

新DFTpy使用ctest、1CPU、24GiB、2h。3個array tasks、同時最多2個，各task依序計算2–3個構型，符合使用者2執行+1等待偏好。6600秒是整組共同計算預算；剩餘不足時保留狀態，不能標成完成。環境已查核為DFTpy2.1.2、ASE3.25.0、SciPy1.10.1。

Nanocolumn後續維持柱軸無限PBC、橫向5 Å真空，QE只鬆弛柱軸（cell_dofree=z），OF cell filter也應只開zz，不能用本bulk的全自由cell設定縮掉真空。生成nanocolumn時需明記5 Å為總影像間隙或單側留白；本次只提交bulk八構型，尚未新建nanocolumn。

## 真實執行回執

2026-09-25 17:19（台北）已提交：

| 方法 | 真實array ID | 分區／配置 | 17:19:51查核狀態 |
|---|---|---|---|
| DFTpy |2154593|ctest、0–2%2、每組2h|task 0與1 RUNNING、task 2 PENDING（JobArrayTaskLimit）|
| QE |2154594|ct56、0–7%1、每case12h|八case PENDING（QOSGrpJobsLimit）|

DFTpy 0V已完成初始及第二次電子最小化（flag=0、324電子、78³），晶胞已由12.119544 Å更新至約12.119061 Å；1V也已完成初始電子最小化（flag=0、321電子）。這些是啟動證據，不是終態結果或收斂能量。QE尚未開始，排程估時不可當作保證。

部署快照為 `al_bulk_vacancy_vcrelax_20260925_20260925T091840Z`，遠端位於 `/work/dawson666/src/campaign_snapshots/`；51個來源檔案逐檔SHA通過。遠端QE 15項、DFTpy 11項測試與兩方法八case預檢通過。第一份未提交快照091822Z僅因測試複製到暫存區仍保留0444權限而未通過；修正tests使用可寫副本後重新凍結，科學輸入與runner未因此更改。

- QE outputs：`/work/dawson666/qe_cases/qe_runs/AL_BULK_VACANCY_VCRELAX_20260925`
- DFTpy outputs：`/work/dawson666/dftpy_project/relax/AL_BULK_VACANCY_VCRELAX_20260925`
- 各root的 `submission.json` 保留原始sbatch args、來源及真實job ID；不能把 `--test-only` 顯示的2154591／2154592誤當成production作業。
- 查看狀態：`squeue -j 2154593,2154594`。重跑前先查回執、queue與各case的attempt，不再次執行同一提交腳本。
