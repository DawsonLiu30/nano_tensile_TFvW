# 用 GitHub 在另一台電腦或 iService 接續 QE 計算

**2026-09-24 更新：** 使用者已登入 Taiwania3 並核對 clone。當前目的地請優先使用 [Taiwania3 專用部署說明](TAIWANIA3_20260924.md)；下文保留一般跨裝置流程。

本次移轉的是 `campaigns/qe_divacancy_20260923`：**固定 DFTpy 鬆弛後的原子位置與晶胞，只做 QE 電子自洽（SCF）計算**。目的先回答兩組雙空位排列哪一個能量較低；不是舊版 PBE 的 `vc-relax`，也不會重新執行拉伸試驗。

GitHub 保存程式、環境規格、精確輸入、偽勢與精簡證據。計算輸出及大型 scratch 放在 Git repository 外面。以下命令分成檢查、準備、執行；**只有明確加上 `run --execute` 或自行呼叫 `sbatch` 才會開始計算**。

## 1. 這一批要算什麼

2026-09-23 的來源狀態如下；不是新電腦已執行的狀態。

| Case ID | 結構 | 來源狀態 | 下一步 |
|---|---|---|---|
| `2V_1NN_D110` | 第一近鄰 [110] | 已有完成的 SCF 輸出 | 保留為參考，先核對方法、版本及收斂 |
| `2V_2NN_D100` | 第二近鄰 [100] | 舊執行中斷，未完成 | 在目的地從新 SCF 開始 |
| `2V_D310_r1` | [310] | 僅準備輸入 | 補算教授關心的低點 |
| `2V_D110_r2` | 中距 [110]，第四近鄰殼層 | 僅準備輸入 | 與 [310] 比較 |

共同條件：108 格點、移除兩原子後 106 原子；固定晶胞；LDA、`Al.pz-vbc.UPF`；60/240 Ry cutoff；3×3×3 k 點；MV 展寬 0.02 Ry；200 bands；電子門檻 `1e-9 Ry`。精確設定與 SHA-256 在 campaign 的 `manifest.json`，以檔案為準。

先比較同原子數的總能差，不需要先補 pristine 或單空位。正式形成能、結合能則仍需要共同條件下的 0V／1V 參考。既有 1NN 結果只有在設定、偽勢、QE 版本與收斂核對後才能加入比較；如果目的地版本／編譯環境不同，應增加同端點重算來檢查差异。完成 SCF 也不等於 cutoff、k 點或模型準確性已驗證。

## 2. 取得同一份程式與輸入

在 Linux 終端，或 Windows 的 WSL Ubuntu 終端中執行。新電腦的 WSL／Linux 安裝與 Git 登入須先完成；不要把存取權杖寫進命令或 repository。

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone --branch codex/portable-qe-20260923 --single-branch \
  https://github.com/DawsonLiu30/nano_tensile_TFvW.git
cd nano_tensile_TFvW
git status --short
git rev-parse HEAD
```

已有這個 clone 時先看 `git status`，保存自己的修改，再 `git fetch origin` 並切換上述分支。不要使用 `reset --hard` 蓋掉另一台電腦的新工作。`git push` 不會替另一台電腦更新檔案；它仍需自行 fetch／pull。正在執行的 attempt 應保留自己的輸入副本，不隨 repository 更新而變動。

## 3. 在一般 Linux／WSL 電腦建立最小環境

已有 conda 或相容的環境管理工具時：

```bash
conda env create -f env/qe-linux.yml
conda activate al-qe-portable
command -v python
command -v pw.x
conda list qe
```

這個最小環境是 Python 3.11 + QE 7.5，足以使用目前的 portable SCF runner；不是完整 DFTpy 分析環境。不要直接複製舊電腦的 `.venv` 或 conda 環境目錄。

HPC 若有站方 QE／MPI module，優先採該站方支援的組合。不要把某套 MPI 的 `mpirun` 與另一套 MPI 編譯的 `pw.x` 混用；也不要沿用舊腳本的 `LD_PRELOAD`。請先確認版本與 module 設定。repository 舊資料中的 `ct56`、account、Intel MPI 與個人 `/work/...` 路徑是歷史紀錄，**本次未確認仍適用**。

## 4. 先驗證與準備，不開始 QE

以下工作目錄要位於持久儲存區、repository 外面，不能用容易被清除的 `/tmp`。WSL 建議在 Linux 檔案系統計算，並將精簡結果另行備份到 Windows；大型 scratch 不必放在 OneDrive 同步資料夾。

```bash
export PROJECT_ROOT="$(pwd -P)"
export QE_WORK_ROOT="$HOME/qe-runs/divacancy_20260923"
export QE_CASE=2V_2NN_D100

python scripts/qe_portable.py verify \
  --campaign campaigns/qe_divacancy_20260923

python scripts/qe_portable.py prepare \
  --campaign campaigns/qe_divacancy_20260923 \
  --case "$QE_CASE" --work-root "$QE_WORK_ROOT"
```

檢查顯示的 case、輸入、偽勢與工作路徑。先從 2NN 開始；另外两個待算 case 使用同樣流程，只替換 `QE_CASE`。不要修改共享 campaign 輸入來反覆覆寫同一個案例。如果要改 cutoff、k 點或研究條件，需另外建立可識別的 campaign／manifest。

## 5. 明確開始一般電腦上的計算

先使用一個程序、兩個 OpenMP 執行緒的保守設定；它不是最優效能保證。確認目的地記憶體足夠。BLAS 固定一個執行緒，避免 MPI、OpenMP 與 BLAS 各自擴增造成超額使用。

```bash
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/qe_portable.py run \
  --campaign campaigns/qe_divacancy_20260923 \
  --case "$QE_CASE" --work-root "$QE_WORK_ROOT" \
  --pw 'pw.x' --threads 2 --max-seconds 39600 --execute
```

命令在前景執行；保持終端可用。若需要 MPI，先核對 MPI/QE 相容性及實體核心、記憶體，再使用例如 `--pw 'mpirun -np 2 pw.x'`，並相應降低每個 rank 的 OpenMP 執行緒數。`--pw` 是命令與參數，不是任意 shell 程式，不要放 `&&`、重導向或 shell substitution。

輸出會寫入新的 attempt；不能把「程式退出」或單獨的 `JOB DONE` 當作 SCF 已收斂。以 runner 彙整的收斂狀態、QE 原始輸出及所用輸入共同判定。

## 6. 在 iService 或其他 Slurm 主機上執行

使用者確認的國網登入指令是：

```bash
ssh dawson666@twnia3.nchc.org.tw
```

2026-09-24 使用者已完成互動登入，貼回 Python、partition、計畫關聯與舊提交腳本；QE modules、執行檔與實際計算仍待現場檢查。Codex 的獨立 SSH 程序不會自動取得使用者另一個終端的登入狀態。
登入憑證不能隨 GitHub 搬移，也不要把密碼／一次性驗證碼寫進 repository。

先在登入節點 clone、驗證並準備，**不要在登入節點直接跑大型 `pw.x`**。`hpc/qe_portable.sbatch` 預設請求 1 task、每 task 2 CPU、16 GB、12 小時；沒有填入猜測的 account 或 partition，也不會自行提交。

若要自動啟用已經確認的 module／環境，可建立 **不追蹤進 Git** 的 `env/local-qe.sh`。內容只放環境初始化，例如站方已確認的 `module load` 或 conda activation；不要放密碼、SSH 私鑰、token。這個檔案會被 Slurm template source，應由本人檢查。

在目的地主機設定真實的絕對路徑與啟動命令：

```bash
# 先 cd 到剛 clone 的 repository 根目錄。
export PROJECT_ROOT="$(pwd -P)"
export QE_WORK_ROOT="$HOME/qe-runs/divacancy_20260923"
export QE_CASE=2V_2NN_D100
export PW_COMMAND='srun pw.x'
export QE_JOB_SECONDS=43200
export QE_MAX_SECONDS=39600

# scheduler 自己的 stdout/stderr 也放到 Git 外。
mkdir -p "$QE_WORK_ROOT/slurm-logs"
cd "$QE_WORK_ROOT/slurm-logs"

# 確認本站 account / partition、檔案系統和 QE/MPI 之後，才執行：
sbatch --export=ALL "$PROJECT_ROOT/hpc/qe_portable.sbatch"
```

若站方要求 account、partition，以你實際有權限的值加在 `sbatch --account=... --partition=...`。`srun pw.x` 只是待核對的示例；某些站方要求其指定的 MPI launcher。主機 SSH alias `iservice` 也必須在該電腦上自行設定，不能由 GitHub 憑空建立登入權限。

可用 `sbatch` 參數改 CPU、記憶體與時間；必須一起檢查啟動命令、rank 數、每 rank 執行緒與資源是否一致。若變更 `--time`，同步調整 `QE_JOB_SECONDS` 為該限制的秒數，讓 `QE_MAX_SECONDS` 至少少 600 秒。模板預設 QE 11 小時停止，留下 1 小時給正常收尾；`max_seconds` 不是完成時間保證，硬性終止仍可能留下不完整 checkpoint。

每個 case 建議一次只啟動一份，保留 job ID。提交另一 case 時修改 `QE_CASE` 再提交；不要同時在兩台主機寫入同一個 attempt 路徑。

## 7. 收集結果並帶回另一台電腦

在 repository 根目錄執行 collector（不會啟動 QE）：

```bash
python scripts/qe_portable.py collect \
  --campaign campaigns/qe_divacancy_20260923 \
  --work-root "$QE_WORK_ROOT"
```

保留以下可追查證據：該次精確輸入、`pw.out`／錯誤 log、run manifest 與其來源 campaign SHA-256、使用的 Git commit、偽勢雜湊、QE 版本、收斂與退出狀態、collector 的結果。只帶回摘要能量而没有輸入與輸出，無法判斷兩台機器是否算同一個問題。

用檔案傳輸先帶回選定的 attempt 證據，再整理到專門的 `evidence` 目錄；確認檔案大小、路徑不含憑證、manifest 的 hash 能核對後，才用 `git add` 明確指定要保存的檔案、commit、push。另一台機器再 pull。不要 `git add` 整個工作目錄或把 `*.save`、wavefunction、電荷密度、環境安裝包與 Slurm scratch 全部推上 GitHub。

本次 **從新 SCF 開始**，不將舊中斷的約 4 GB 2NN checkpoint 當成可攜結果。QE 官方指出資料目錄包含二進位資料，跨機器可讀性不保證；正常停止也比強制終止更能保護檔案。未來若要真正接續 wavefunction checkpoint，需另外確認完整性、QE 編譯／版本與資料格式，而不是只搬 `.save` 就假定可繼續。參考：[QE 官方 Data files](https://www.quantum-espresso.org/Doc/pw_user_guide/node9.html)。

## 8. 完成後可以回答哪些問題

- 1NN 與 2NN 都有相容、收斂的結果後，才能比較兩個近鄰排列的能量排序。
- [310] 與中距 [110] 都完成後，才能用自己的 QE 檢查教授問的低點差異；兩者距離也不同，不能全部歸因於晶向。
- 尚未完成 cutoff／k 點／展寬收斂、QE 原子鬆弛與共同 0V／1V 參考前，不宣稱完整準確性驗證或正式 QE 結合能已完成。
