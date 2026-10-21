# 2026-10-01 22:03：確認空轉後停止並重新提交

使用者明确授權：確認真的空轉則取消重來，檢查log與記憶體，詳細解釋補算及問題、查閱官方文件。本紀錄優先於前段「等取消確認」。

## 已做的操作

- 21:52對原四個allocation各取20秒前後程序快照。pw.x各剩28/25/18/27，應為29；存活程序CPU約99.6%，全程序I/O與pw.out的hash/size/mtime不變。結合約18h無科學進度及缺失MPI ranks，確認空轉。
- job層cgroup limit 64GiB，峰值達限；memory.failcnt為14123/52237/126716/104802。這是觸碰限制次數，不是死亡程序數；OOMkill counter仍0，未取得各rank直接死亡原因，不說已證明kernel oom-kill。
- 21:53備份完整小型輸出後取消2173492_11–14，sacct確認CANCELLED。舊前11個完成SCF未取消，全部舊輸出/scratch保留。
- **21:57正式重新提交**：array **2180070**（indices0–1，R15_0V/R15_1V）每task128GiB；array **2180073**（indices2–3，R20_0V/R20_1V）每task256GiB。各29ranks/1node、ct56、48h/內部47h，每批%2，總共可4組並行。
- 最後22:03快照四組PENDING/Priority。尚未啟動，沒有聲稱新allocation實際guard已執行成功，不等待排隊／不開Codex監控。後續查fresh queue再操作，不重送。

## 資源與科學協定

- 小柱估計72.60/72.10GB，總量在pw.out L371/L367；大柱156.45/155.76GB在L531/L527。前100行的available memory屬主機剩餘，不是job權限。逐case的numbered first100和memory lines保存於新報告evidence。
- Slurm25.11.0，KillOnBadExit=0，task/cgroup，ct56 MaxNodes1/MaxCPUs56/MaxTime4days。現場節點RealMemory至少384564MiB，cpn3028另772412MiB；不能沿用舊192GB硬體介紹代替實際排程。128/256GiB的test-only與正式submit均通過。
- 四份新pw.in與舊版逐byte/hash相同，cutoff/PAW/PBE/k/MP/SCF/force/stress/結構/真空/axis vc-relax不變。新的recovery manifest加入resources及舊arrayid，未倒改旧manifest。
- 大記憶體來源包括PAW addusforce，約1981/4162MB per rank；disk_io medium僅部分減wavefunction記憶體，增加k pools不保證減augmentation/FFT等所有部分。此次保持MPI29與物理條件，先增加確實不足的配額。
- 新runner啟動前驗SLURM_MEM_PER_NODE與沿cgroup祖先鏈最嚴限制≥128/256GiB；不足則拒跑。srun加 --kill-on-bad-exit=1 --oom-kill-step=1 --wait=60，本機台help支援；沒有修改全系統Slurm或安全設定。
- 從相同初始幾何fresh啟動。原中斷作業無完整force/ionic step、無正常restart端點，不硬接零碎wavefunction。驗收沿用run_luder_qe_20260930.py inspect，絕不以Slurm COMPLETED替代。

## 新檔與來源

- Campaign：campaigns/al_nano_qe_memory_recovery_20261001
- Runner：scripts/run_nano_qe_memory_recovery_20261001.py
- Sbatch：hpc/nano_qe_memory_recovery_20261001.sbatch
- Tests：tests/test_nano_qe_recovery_20261001.py；本機/遠端各5test通過。第一次remote preflight因測試臨時副本繼承唯讀權限失敗，沒有提交；修測試後另凍結snapshot重新驗證。
- 正式snapshot：`/work/dawson666/src/campaign_snapshots/al_nano_qe_memory_recovery_20261001_20261001T135638Z`；SOURCE_MANIFEST sha=ce639dcf9d53e0f82e0c734687564e1463439b19ba4f0a6b3b7ada337f7749cc，46source檔。
- 新run root：`/work/dawson666/qe_cases/qe_runs/AL_NANO_QE_MEMORY_RECOVERY_20261001`，分開的submission_R15/R20.json防重送。
- 本對話outputs/qe_nano_recovery_20261001保存診斷、取消、提交、部署、最後狀態與中英報告。

## 已完成補算的解釋與OF候選

沿用docs/LUDER_RESULTS_AUDIT_20261001_ZH.md：三個k3對照重現舊值，全八態k4加匹配參考得到巨大的Eb/force/stress敏感性；k4不是重新鬆弛或已驗證極限，不宣稱5NN/1NN物理機制已被確定。本輪只有4個QE nano resource recovery，沒有新bulk/k5/smearing/OF生產計算。

DFTpy vW stress另準備未部署候選，從同一離散sqrt(rho)能量的二階頻域算符求導，保留Nyquist項。2個奇/偶網格、斜盒合成密度及4個保存nano密度共36個獨立tensor分量與finite strain吻合，最大差1.241895e-9GPa。只評估保存rho，無SCF；尚缺全total-stress/force及grid/vacuum/production接入驗證，不倒改原result.json或現役dftpy。

## 官方文件已查

- [QE troubleshooting](https://www.quantum-espresso.org/Doc/pw_user_guide/node21.html)：授權記憶體可能小於主機可用；正常退出才有可靠restart。
- [QE parallelization](https://www.quantum-espresso.org/Doc/pw_user_guide/node18.html)、[input](https://www.quantum-espresso.org/Doc/INPUT_PW.html)：記憶體分佈與disk_io。
- [Slurm cgroup](https://slurm.schedmd.com/cgroup.conf.html)、[srun](https://slurm.schedmd.com/srun.html)：部分程序被限制後其他程序可能繼續；整step停止旗標。
- [國網queues](https://man.twcc.ai/@TWCC-III-manual/ryyo0tsuu)、[iService Slurm FAQ](https://iservice.nchc.org.tw/nchc_service/nchc_service_qa.php?target=118)。
- [Smearing比較研究](https://arxiv.org/abs/2109.01196)：支持針對能量/力檢查k/smearing，不當作本案誤差值。
- [DFTpy官方vw.py](https://github.com/Quantum-MultiScale/DFTpy/blob/master/src/dftpy/functional/kedf/vw.py)、[release notes](https://dftpy.rutgers.edu/releases.html)：當前stress公式對照與2.1.3 line-search改進不等同vacuum stress已修。
