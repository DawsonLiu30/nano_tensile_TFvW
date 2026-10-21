# 2026-10-01 晚間：完成結果與執行阻塞查核

使用者要求查已完成計算，不啟用監控。約 21:00–21:08 台灣時間的實際快照：

- QE array2173492：0–10 共11個bulk單點完成，全部frozen runner inspect accepted_scf_diagnostic。11–14為4個nano axial vc-relax，Slurm仍RUNNING，但實際停滯。
- DFTpy2173494與2173496共4個nano計算完成，電子flag0、電子數、力與軸向應力通過原停止條件。新增獨立查核發現橫向應力數值問題，因此不能籠統當作完整科學驗證通過。
- 本次沒有新DFT、沒有重新提交、没有修改cutoff/PP/XC或原canonical、沒有定時追蹤。已詢問使用者是否取消四個卡住QE；本紀錄寫入時尚未收到回答，未執行取消。

## QE bulk 敏感性

三個原k3幾何重算（1NN/3NN/5NN）與舊能量最大差0.00005276meV，支持資料完整性。全八態在舊幾何上的k4SCF使用匹配k4 0V與1V參考：

| 量 | 原k3鬆弛 | 舊幾何k4單點 |
|---|---:|---:|
| 1V形成能/eV | 0.644963 | 0.679875 |
| 1NN結合能/meV | −2.328 | −250.429 |
| 2NN結合能/meV | +22.680 | +37.257 |
| 3NN結合能/meV | +50.146 | −183.825 |
| 4NN結合能/meV | +16.902 | +32.498 |
| 5NN結合能/meV | +56.681 | −181.997 |
| 6NN結合能/meV | +23.315 | +29.413 |
| D12/meV | +25.008 | +287.686 |

k4不是重新鬆弛、更不是已證明收斂的正確答案。缺陷max force約0.030–0.046eV/Å、部分應力>1GPa。舊5NN排序與利用舊QE作為黃金標準判斷OF機制的主張需暫緩；本測試也不能獨立區分取樣、影像與真實物理對4.4%形變的貢獻。PAW/PBE、800eV、8倍rho cutoff、MP0.02Ry保留，SCF1e-10Ry。

## DFTpy nano 初步能量/結構及stress診斷

[111]軸向週期柱，半徑1.5a0/2a0，96/176父格點；DFTpy半徑5.921/7.895Å。橫向每側5Åpadding，名目gap10Å，xy固定，z鬆弛。0V/1V共4組，固定FFT147×147×96及175×175×96。

- 同方法bulk μ=−62.9438575446eV/atom，Ef=E_nano1V−E_nano0V+μ；兩尺寸暫為0.598150643/0.631826232eV。
- 1V比同尺寸0V額外軸向縮短0.0197246/0.0137170Å。終態無nearest-parent換位，空格點仍50/77，無多重佔據；不等於觀察到遷移、凝聚或表面吸收。沒有聲稱整條軌跡無瞬時換位。
- 原total σyy為−8402.9、−4843.0、−21980.2、−178612.6GPa，不可作物理解釋。原runner只gate zz，漏掉這項異常。
- 以保存rho重算：vW能量重現<1e-8eV；local/remote vw.py SHA相同47678fd55e19fd199465f549d1d6e37f491cb0a4ce02c09fbff573f60b0015db。原stress用∇ρ∇ρ/ρ，真空極小rho造成數值放大，ρ<1e-8的區域僅约1e-5電子卻貢獻幾乎全部異常。
- ±1e-4/±1e-5保持電子數的有限strain vW能量導數與sqrt(rho)梯度stress相符<1e-6GPa。只作保存density後處理，沒有新SCF。以後者替換vW的診斷zz僅差0.00020–0.00033GPa；全4組診斷zz<0.0017GPa。但非完整total-stress/force驗證，不倒改原結果或宣稱所有誤差排除。
- 能量與結構保留pilot；橫向stress拒用。真空、軸長、grid與QE比較未完成。production修正與獨立验证後再談彈性或表面應力。

## QE nano 實際停滯

2173492_11/12/13/14的實體job/node分別2174109/cpn3028、2174110/cpn3087、2174111/cpn3152、2173492/cpn3114。原本各29 ranks，實際pw.x剩28/25/18/27。最後輸出10/1台灣04:16/04:19/03:55/03:51，約17h無更新。R15僅完成第一SCF，R20無完整SCF，四組皆無force block。

沿用64GiB配置不足以涵蓋QE自估72.60/72.10/156.45/155.76GB；四組cgroup峰值皆近64GiB，R20_0V有9次memory.failcnt。所取OOMkill counter為0，缺失rank的確切死亡原因未直接證明；不能宣稱已取得OOM kill日誌，但明顯是MPI執行不完整。

先嘗試node SSH唯讀失敗，未互動認證；後用四個既有allocation的單程序短暫srun --overlap讀自己的/proc/cgroup，無新allocation/DFT。沒有修改登入或安全逾時。待確認取消後另規劃足夠memory及MPI配置，保留科學輸入；勿擅自降ecut。

## 本機歸檔

- 新NAS增量：`C:/OFDFT/NAS_SHARE/Al_vacancy_Luder_update_20261001_evening`，入口START_HERE.html（中文）及REPORT_EN.html（英文）。同名zip 92,419,718bytes，SHA256=fb008e36f984e86f48455261890bc0d4a66682b29efe86644abe558de12653bf。
- 永久副本：`C:/OFDFT/RESEARCH_UPDATES/AL_LUDER_RESULTS_20261001`。
- 工作原件：本對話`outputs/luder_results_20261001`。
- 279下載原始檔全hash通過；新包309檔、348項分析核查、27報告連結、ZIP全部payload核對。QA是解析/算術/來源/後處理驗證，不是物理收斂認證。
- 保留原始輸出、DFTpy density/trajectory、QE XML、凍結sources、partial running logs及省略清單。174.44GB大型scratch/重複checkpoint未下载，不是完整wavefunction restart包。
- 原先514MB完整NAS包不覆寫；本增量需一併保存。沒有上傳NAS或寄信。後續先讀本紀錄再操作，避免把舊accepted狀態當完整物理驗證。
