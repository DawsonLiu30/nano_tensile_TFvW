R15 k9 最小續查包，2026-10-17（準備完成，尚未提交）

理由：已完成的k7 SCF並沒有程式故障，但k5→k7的成對能量差仍變動
10.049289958 meV，且力場仍敏感。使用者要求未通過的工作排查後繼續完成；
因此這次只增加R15 0V/1V的k9單點，供選定重新鬆弛的取樣設定。

只新增：
  index0 nano_R15_0V_k9_SCF_20261017，96原子
  index1 nano_R15_1V_k9_SCF_20261017，95原子

物理輸入從原k7逐字衍生，只改prefix和1×1×7→1×1×9，offset不變。
同一歷史k3終態晶胞/座標、PAW/PBE偽勢、800eV、8倍rho cutoff、MP.02Ry、
SCF1e-10Ry、nbnd、初始化、71h內停都保持原值。初始extxyz也逐位元相同。

沿用而非改寫：
1. 10/15完整52檔凍結snapshot與原科學inspector。
2. 10/17已成功執行k7的reviewed入口，利用其既有父hash/gate驗證。
3. 同一memory allocation guard、29MPI ranks的fail-fast srun命令。
這次新入口只處理兩份輸入overlay與新版本來源，不修改舊guard與gate。

資源保持k7已成功使用的256GiB/29ranks/72h；array0-1%2。
不是新測得的k9記憶體保證。保留完整scratch；新case/attempt_001必須不存在。
每case lock與mkdir防止重複執行。舊k3/k5/k7與六尺寸case不動。

部署檔案：
  run_reviewed_k9.py
  submit_k9.sbatch
  overlay_manifest.json
  prepare_overlay.py / test_k9_overlay.py / program_QA.json
  inputs/<兩個case>/pw.in、initial.extxyz

Linux純程式測試（不算DFT）：
python -B test_k9_overlay.py --parent-snapshot <10/15原snapshot> --root <原QErunroot> --k7-entry <10/17k7snapshot>/run_reviewed_k7.py

檢查：
python -B run_reviewed_k9.py verify --parent-snapshot <10/15原snapshot> --root <原QErunroot> --k7-entry <10/17k7snapshot>/run_reviewed_k7.py

run由sbatch執行，同參數加 --index 0或1；不用重建原snapshot。
sbatch環境：REVIEW_SNAPSHOT、PARENT_SNAPSHOT、K7_REVIEW_SNAPSHOT、QE_RUN_ROOT。
本機資料配置可另帶 --k7-evidence-root <FOLLOWUP_RESULTS>；遠端不需此參數。

14項Linux/WSL純程式測試通過，含實際mesh9及錯誤mesh/prefix/cutoff/幾何拒絕、
電子未收斂/非零退出拒絕、記憶體/ranks、case範圍與原guard保持。
這些是程式測試，不代表k9物理收斂。原子力門檻仍.001eV/Å，分量.0005，軸應力.05GPa。
單點成功只代表該固定幾何的電子/資料驗收；不自動放行六個新尺寸。
得到k9後仍需判斷取樣穩定性，再對代表柱重鬆弛，並在新幾何驗證。

目前overlay manifest SHA256：
fc36ca93d6fc4a25047f2c838b3f3330c013ba92045e75a35d1eddd538a5053a
