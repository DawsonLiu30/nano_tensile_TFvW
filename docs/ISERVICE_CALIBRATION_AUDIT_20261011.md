# iService與校準連續性查核：2026-10-11

使用者要求全面檢查現役與先前數據，延續既有收斂校準。已先讀AGENTS及CONVERGENCE_BASELINES。此次沒有新增/取消/重送DFT、修改現役輸入、放寬門檻或建立監控。

## 單次遠端快照

台灣時間01:07–01:15，現役只有2221957_0/1：R20 0V/1V，實際job2221958/2221957，cpn3023/cpn3178。各29ranks、256GiB，72h Slurm/71h QE；Slurm EndTime=10/13 11:45:27，不是完成預測。

新增完整SCF/總力區塊13/9；fmax=0.0016875381/0.0056318001 eV/Å，力最大分量0.0016151671/0.0055453558 eV/Å。尚未合格，不出正式R20形成能。BFGS計數39→51、26→34；01:07到01:15電子iteration累計368→372、307→311，stdout持續增加。

短時間唯讀診斷step在既有allocation內完成，沒有新DFT或新allocation。直接SSH到compute nodes被拒後採此方式；該失敗不是rank缺失證據。實測各29/29個pw.x（排除srun和診斷Python程序），job記憶體峰值178.5974/177.9066GiB，QE step峰值162.814/162.175GiB附近，以memory_summary.json精確值為準；job/step failcnt皆0。256GiB目前未觸限，不能保證未來最大需求。

## 參數及來源查核

- 35組主資料的689個既有payload SHA256全部吻合；70份核心輸入/結果與遠端逐份相同。21個已接受QE結果的stdout能量/連續nat列總力與result.json在輸出精度內一致。
- 23個歸檔QE及2個現役QE均維持800eV、8倍rho cutoff、MP.02、PAW/PBE。新restart只改restart_mode與時間，反向還原後和parent輸入逐字相同。12個DFTpy均LDA/local、TF=.9/vW=.1；8 bulk為78³、nano為147×147×96或175×175×96。
- 35個结构PBC皆TTT；8個nano（QE R20用舊最後完整步）在3.3Å近鄰檢查中跨z連續、無跨xy近鄰。這是幾何檢查，不是電子真空收斂測試。
- 9/24舊固定cell/SCF抽查為60/240Ry、MV.02、Al.pz-vbc.UPF；此歷史LDA/NC協定未混進目前35組。May6 EOS已知遠端位置仍未找到，home/formal與研究目錄有界追查也未補得high-k原始輸入。不是使用者未做校準。
- 本次81項來源/參數/解析檢查通過，另35個ASE結構核查通過；不是新DFT精度認證。

## 科學判讀與下一步

既有primitive40³/80³ EOS、QE800eV及DFTpy LDA78³校準仍為基準。主要流程缺口是新108格點k3並未先完成轉用驗證；現有k4為診斷，不是已收斂正式網格。nano1×1×3也未被PPT認證。

QE k3→k4固定幾何的1NN/3NN/5NN Eb差−248.101/−233.971/−238.678meV；單空缺Ef從.644963到.679875eV，不能再把單空缺「約3%一致」當最後準確度保證。舊pair ordering、5NN最穩及OF因果主張暫緩。原canonical與六圖不倒改。

五態k4草案先重審原/新晶胞、倒空間取樣與目標量，不直接提交當正式設定。既有兩個nano續跑確有下降，保持跑完作pilot/幾何起點，再做所需k/grid/vacuum/period驗證。OF nano stress仍需完整production revalidation。沒有自行擴張計算矩陣。

## 新發現：restart的startingwfc設定

兩個pw.out第25/27行警告startingwfc='atomic+random'與startingpot='atomic'；第548/544行印出randomized atomic wfcs，後續確有SCF及BFGS延續。官方QE7.5 `PW/src/input.f90` 838–847會將startingpot修成file，對startingwfc只警告。不能以121份scratch複製驗證成功取代wavefunction讀回證據，也不能僅憑警告斷言後續已自洽能量必錯。

目前不干預下降中的作業。後續restart協定應明設兩項為file，另留新版本、調整來源差異檢查並測試，不能默改frozen snapshot。最終仍需正常電子/幾何與final-cell SCF驗收。

官方依據：[QE input說明](https://www.quantum-espresso.org/Doc/INPUT_PW.html)、[QE7.5實作](https://github.com/QEF/q-e/blob/qe-7.5/PW/src/input.f90#L838-L847)。

## 位置

- 使用者可讀報告及逐組參數：`C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/REPORTS/calibration_audit_20261011`，入口REPORT_ZH.txt。6檔約86.6kB，不複製整套資料。
- 原始此次查核、現役日誌及hash：本對話 `outputs/iservice_audit_20261011`。remote_snapshot是執行中日誌快照，不是正式新端點。remote_reconcile.json記70個遠端對照與node原始讀取；原始process列表含srun與Python，rank數讀memory_summary.json的嚴格pw.x filter結果。
- 本機再現程式：`work/iservice_audit_20261011`。audit_local.py讀本機/已捕捉資料、finish_audit.py產報告、check_structures.py用WSL ASE核查；*.sh是本次唯讀遠端查核，沒有scheduler mutation或DFT提交。
