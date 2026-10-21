# Bulk Al 空位固定晶胞鬆弛：執行紀錄（2026-09-24）

本輪 campaign 為 `al_bulk_vacancy_relax_20260924`：在同一個約 12 Å 的理想 FCC 晶胞中，分別以 DFTpy TFvW 與 QE 從相同初始結構做**固定晶胞、原子位置鬆弛**。8 組幾何與 QE 輸入已準備完成；DFTpy 已在本機 WSL 執行，QE 已提交 Taiwania3。提交、電子收斂、原子收斂與數值／尺寸收斂是不同狀態。

## 執行快照與位置

| 項目 | 本輪位置／識別 |
|---|---|
| Windows 可編輯 repository | `C:\OFDFT\nano_tensile_TFvW` |
| 初態、parent-site 對應與 QE 條件 | `campaigns/al_bulk_vacancy_relax_20260924/geometry.json`、`qe_manifest.json` |
| 本機凍結來源快照 | `C:\OFDFT\LOCAL_MACHINE_20260907\snapshots\al_bulk_vacancy_relax_20260924_20260924T055907Z` |
| 本機 DFTpy 持久輸出 | `C:\OFDFT\LOCAL_MACHINE_20260907\runs\al_bulk_vacancy_relax_20260924` |
| WSL Python | `/var/tmp/al-defects-runtime-20260907/env/bin/python` |
| Taiwania3 帳號 | `dawson666@twnia3.nchc.org.tw` |
| 遠端凍結來源快照 | `/work/dawson666/src/campaign_snapshots/al_bulk_vacancy_relax_20260924_20260924T055907Z` |
| 遠端 QE 持久輸出 | `/work/dawson666/qe_cases/qe_runs/AL_BULK_VACANCY_RELAX_20260924` |
| QE Slurm array | **`2151591_[0-7%1]`**；一次最多執行 1 個新案例 |
| QE binary | `/home/u1871490/qe-7.5/bin/pw.x` |
| QE batch script | `hpc/bulk_vacancy_relax_20260924.sbatch` |

Windows 持久資料在 WSL 下對應 `/mnt/c/OFDFT/...`。計算來源使用已凍結 snapshot；後續修改 Windows repository 不會自動改變已提交的計算。

## 本輪 8 組初始結構

共同晶胞為 3×3×3 傳統 FCC cell，108 個母晶格位點，三方向週期邊界。共同參考晶格尺度 `a_ref = 3.9545804060131293 Å`，邊長 `L = 11.863741218039388 Å`；這是沿用的比較尺度，尚不是本輪各方法重新擬合的平衡 `a0`。

| Array index | Case | Al 原子数 | 移除 parent IDs（0-based） | 初始 MIC 距離（Å） |
|---:|---|---:|---|---:|
| 0 | `0V` | 108 | 無 | — |
| 1 | `1V` | 107 | 43 | — |
| 2 | `2V_1NN_D110` | 106 | 43, 64 | 2.796310622 |
| 3 | `2V_2NN_D100` | 106 | 43, 79 | 3.954580406 |
| 4 | `2V_3NN_D211` | 106 | 43, 82 | 4.843352071 |
| 5 | `2V_4NN_D110` | 106 | 43, 85 | 5.592621244 |
| 6 | `2V_5NN_D310` | 106 | 43, 100 | 6.252740637 |
| 7 | `2V_6NN_D111` | 106 | 43, 86 | 6.849534186 |

第一空位位於中心 parent ID 43。每組 `POSCAR`、`structure.extxyz` 和 QE 輸入均由同一母結構刪除指定原子；`geometry.json` 保存剩餘原子的 parent IDs，供比較位移與近鄰重排。MIC 是**初始移除格點**的最小影像距離，不是鬆弛後缺陷距離或反應座標。沿 [110] 的第二距離點屬 FCC 第 4 近鄰，不是第 2 近鄰。

`2V_5NN_D310` 有兩個同樣近的週期影像，列為 periodic-image diagnostic；不能拿它或最遠的 `2V_6NN_D111` 當作已知的無限分離參考。更多幾何檢查見 [初始結構包說明](../campaigns/al_bulk_vacancy_relax_20260924/README.md)。

## 與前一輪 QE SCF 的區別

| 項目 | 本輪 `al_bulk_vacancy_relax_20260924` | 前一輪 `qe_divacancy_20260923` |
|---|---|---|
| QE calculation | `relax`，固定 cell，放鬆所有原子 | `scf`，原子與 cell 均固定 |
| 初態 | 新建的理想 FCC 0V／1V／6 組 2V | 先前 TFvW 鬆弛後的 4 組 2V 幾何 |
| 可回答的問題 | 各方法獨立鬆弛後的形成／結合能與結構差異 | QE 在指定 TFvW 幾何上的單點能量差 |
| Runner | `scripts/run_bulk_vacancy_qe_20260924.py` | `scripts/qe_portable.py` |

舊 1NN 已有完成紀錄，舊 2NN／[310]／中距 [110] 的作業另由既有佇列管理；本次新 array 沒有取消、覆寫或取代它們。兩輪資料不得直接混成一條「QE 已完全鬆弛」曲線，也不與八月的變晶胞資料混用。

## 方法與驗收條件

DFTpy：`Ts = 0.9 T_TF + 0.1 T_vW`，LDA，`al.lda.recpot`，78×78×78 固定網格（約 0.152099 Å）。每個新案例從理想 FCC 與 HEG 電子密度開始；單次鬆弛內可重用上一電子密度。電子 LBFGS 的 `econv` 為 10⁻⁹ Ha/atom、`ncheck=3`、`maxiter=1000`；原子 LBFGS 目標 `fmax≤0.001 eV/Å`，最多 1000 步。`accepted=true` 同時要求最終力門檻與全部電子評估通過，不只檢查最佳化器退出碼。

QE：LDA `Al.pz-vbc.UPF`、3 valence electrons/Al、60/240 Ry cutoffs、3×3×3 無位移 k 點、MV 0.02 Ry、200 bands。電子 `conv_thr=10⁻⁹ Ry`、`electron_maxstep=200`；原子 BFGS 最多 200 步，`etot_conv_thr=10⁻⁶ Ry`。`forc_conv_thr=1.9446903811488877e-5 Ry/bohr`（0.0005 eV/Å）；輸出再驗證最大原子力向量長度 ≤0.001 eV/Å。詳細條件與輸入 SHA-256 以 `qe_manifest.json` 為準。

QE 每案例配置 1 node、29 MPI ranks、1 thread/rank、64 GB、12 h、`--no-requeue`；QE `max_seconds=39600` 留出正常結束與記錄時間。compute node 載入 `intel/2022`、`intelmpi/2021.11`，經 `srun --mpi=pmi2` 啟動。新的 attempt 不讀舊波函數或 scratch；runner 重新設定自己的 `pseudo_dir`、`outdir`、prefix 與 fresh-start 電子初值。

QE 的 `JOB DONE` 不等於原子鬆弛收斂。Runner 分開記錄電子收斂、BFGS 原子收斂、退出碼與限時／步數狀態；只有 `usable=true` 才可納入已完成的比較。QE 自身停止準則通過但力向量門檻未過時，標成 `converged_needs_review`。未完成作業的最後能量僅供診斷。

## 已核實進度

<!-- Execution-status block: update from finished result.json and fresh Slurm evidence. -->

- **DFTpy**：8/8 `attempt_001` 均為 `status=converged, accepted=true`；全部電子評估、最大原子力 ≤0.001 eV/Å 與來源／產物 hashes 通過後處理檢查。
- **QE**：array `2151591_[0-7%1]`，最新查核 UTC 2026-09-24T08:28:10.018896+00:00；詳細佇列如下。

```text
JOBID                                NAME        STATE       TIME         NODELIST(REASON)
             2151591_0          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_1          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_2          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_3          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_4          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_5          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_6          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
             2151591_7          al-bulk-vac-relax-20260924      PENDING       0:00        (QOSGrpJobsLimit)
```

- 已有本輪 DFTpy 形成能、結合能、電子動能差及 matched-site 位移表；尚無已完成的本輪 QE relaxation 交叉比較。

<!-- End execution-status block. -->

## 檔案與 hash 追溯

| 凍結來源 | SHA-256 |
|---|---|
| `geometry.json` | `aa5b79d49467c6aa36090eb5f423849c1af3f5ed42188069bb708b001bc67870` |
| `qe_manifest.json` | `0a32d9ee5e91032391cc9ce7c4196061446f95ccc30f4e58a09abd53bd03a815` |
| QE `Al.pz-vbc.UPF` | `4eab06b63f87f07ede2d5a193e6d993a09107167fd6b8647afa807342501d6e5` |
| DFTpy `al.lda.recpot` | `c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803` |
| QE runner | `22265085c54b1873d52e3b9512781a0470e8ef2e9b7d97d790fbc88bd6e8ba49` |
| DFTpy runner | `c796adaecdb89b46fd05e1098629f7b3a6256c6621f8494f1887e38130bff6ac` |

QE protocol hash：`cf2086ff53828724b3a8ae153b4ba6ba5bd562a9d6f801d048e4a699a6353121`。每份 `pw.in`、POSCAR、extxyz 的個別 hash 在 `qe_manifest.json`；runner 也逐原子比對 POSCAR 與 QE 初態坐標。DFTpy 每次 attempt 的 `provenance.json` 記錄實際 runner、engine、偽勢、幾何與展開後設定的 hash。上表識別本次執行快照，不應因日後 repository 改版而倒填新 hash。

QE 輸出採 `<case>/<UTC timestamp>-<UUID>/`，保存 `attempt.json`、`source/`、runtime `pw.in`、`pseudo/`、`pw.out`、`pw.err`、`result.json`；可解析時另存 `trajectory.json`、`final_geometry.json`、`final.extxyz`。Source/runtime 輸入唯讀，運行前後查核 hash；每 case 有 exclusive lock，失敗紀錄與舊 attempt 不覆寫。

DFTpy 輸出採 `<case>/attempt_001/`，保存 `provenance.json`、`config.json`、電子與原子 JSONL 評估、`relax.traj`、`relax.log`、`checkpoint.*`、完整 DFTpy log。完成後另存 `final.extxyz`、`final.vasp`、`final.traj`、`final_density.npz`、`final_forces_eV_A.npy` 及 `result.json` 的逐檔 artifact hashes。這些檔案在 Git 外持久保存。

## 重現驗證與只讀監控

以下指令不提交新作業；現有 array 不需要重提。

在 repository 或凍結快照中檢查 QE 輸入及 runner：

```bash
python3 scripts/run_bulk_vacancy_qe_20260924.py verify
python3 scripts/run_bulk_vacancy_qe_20260924.py dry-check --case 2V_1NN_D110
python3 -m unittest discover -s tests -p test_bulk_vacancy_qe_20260924.py -v
```

本次 QE runner 的 12 項測試通過；測試用真實輸入與模擬 subprocess，沒有啟動 production QE。重新檢查 DFTpy 設定可在已安裝 DFTpy/ASE 的 WSL 環境執行：

```bash
cd /mnt/c/OFDFT/LOCAL_MACHINE_20260907/snapshots/al_bulk_vacancy_relax_20260924_20260924T055907Z
/var/tmp/al-defects-runtime-20260907/env/bin/python scripts/run_bulk_vacancy_dftpy_20260924.py --case 0V --dry-check
```

Windows PowerShell 讀取本機各 attempt 狀態：

```powershell
$runRoot = 'C:\OFDFT\LOCAL_MACHINE_20260907\runs\al_bulk_vacancy_relax_20260924'
Get-ChildItem -LiteralPath $runRoot -Filter result.json -File -Recurse | ForEach-Object {
    $result = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
    $result | Select-Object case, attempt, status, accepted, ionic_steps, final_fmax_eV_A, finished_utc
}
```

登入 Taiwania3 後查看排程與完成回執：

```bash
squeue -r -j 2151591 -o '%.20i %.10T %.12M %.10l %R'
sacct -j 2151591 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P
```

用凍結來源檢查已有 QE attempt。`find` 只列出已建立的 attempt；pending 案例還沒有 attempt 是正常情況：

```bash
task_source=/work/dawson666/src/campaign_snapshots/al_bulk_vacancy_relax_20260924_20260924T055907Z
task_runs=/work/dawson666/qe_cases/qe_runs/AL_BULK_VACANCY_RELAX_20260924
find "$task_runs" -mindepth 3 -maxdepth 3 -name attempt.json -print
```

對選定且已結束的 attempt，把實際完整路徑代入 `--attempt`：

```bash
/home/dawson666/miniconda3/bin/python3 "$task_source/scripts/run_bulk_vacancy_qe_20260924.py" inspect \
  --campaign "$task_source/campaigns/al_bulk_vacancy_relax_20260924" \
  --attempt /absolute/path/to/selected/case/attempt-directory
```

`inspect` 重新核對 immutable files、stdout/stderr、result hashes 與電子／原子收斂，不單靠 Slurm `COMPLETED` 或已存在的最後能量。

## 本輪結果的科學邊界

本輪先取得共同有限週期晶胞下的能量與結構資料。108-site cell 的 vacancy fraction 為 1/108 或 2/108；尚未證明缺陷影像、濃度與 cell size 影響已消失。QE cutoff／k 點／smearing、DFTpy 網格／KEDF 選擇及平衡晶格尺度也仍需獨立查核。**作業收斂不等於 finite-size 或物理準確性收斂。**

形成能與結合能只能在同方法、同晶胞及一致 reference convention 下計算；不可直接以兩套偽勢的原始總能量互相相減。六個 2V 案例是獨立終態候選，最佳化 trajectory 不是具物理時間的 vacancy condensation 動力學；遷移能障需另做路徑計算。完整 proposal 還包含 crystalline Al nanocolumns，當前這一批只執行 bulk 分支，未宣稱 nano 分支已完成。
