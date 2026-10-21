# QE 單空缺未收斂原因與修復

2026-09-28 使用者明確授權診斷、查閱資料、修正並重新提交，直到取得合格結果。此修復保留原始失敗紀錄與已合格七個 QE 構型，不把失敗末步改標為完成。部署與真實作業編號由後續回執記錄。

## 已證實的停止原因

September26 的1V續跑不是逾時，Slurm於1小時30分42秒後FAILED。四個完整SCF的enthalpy依序為−4226.7178840078、−4226.7178840293、−4226.7178840542、−4226.7178840691 Ry；最大原子力分量依序0.000690598、0.000663602、0.000593411、0.000549445 eV/Å，皆持續下降。

後三個移動步印出的trust radius為2.79446e−5、2.58190e−5、2.30773e−5 bohr。`pw.out` 5685、8325–8327、10908–10914顯示過小trust radius、重設history、再次觸發、停止。Theta=.905/.928屬Hessian curvature damping警告，並非已證實的力震盪或物理負曲率。

[QE 7.5 BFGS原始碼](https://github.com/QEF/q-e/blob/qe-7.5/Modules/bfgs_module.f90#L986)的下降分支也會在trust radius低於最小值時重設；連續觸發後停止。因此這是有來源支持的optimizer停止機制，不代表晶體無法找到最低點。

[官方輸入文件](https://www.quantum-espresso.org/Doc/INPUT_PW.html)列預設1e−3，但[7.5版本原始碼](https://github.com/QEF/q-e/blob/qe-7.5/Modules/read_namelists.f90#L544)以及遠端實際`/home/u1871490/qe-7.5/Modules/read_namelists.f90`第548行均為1e−4 bohr。遠端檔SHA256為dacf74da652b2a07b22e27b383c828885f58078287a84143feb5a6c224b9f7ea；`bfgs_module.f90`為80417f1e5cd5f52a84ab81878d50175ea078071ce3b76fcb4a959089ab4d9432。這是可讀source的查核，非重新建立binary的證明；明設新值消除預設差異。

## 本次最小修復

| 參數 | 原attempt | 新attempt | 目的 |
|---|---|---|---|
| IONS.trust_radius_min | 未明設 | 1e−6 bohr | 允許已觀察到的細小下降步，避免連續history reset |
| ELECTRONS.conv_thr | 1e−8 Ry | 1e−10 Ry | 加嚴電子解，改善小步長能量／力一致性 |
| forc_conv_thr | 1.94469038114889e−5 Ry/bohr | 保留 | 約0.0005 eV/Å分量門檻 |
| stress與energy停止值 | 0.5 kbar、1e−6 Ry | 保留 | 不放寬既有離子／晶胞驗收 |
| PP、XC、cutoff、kmesh、smearing | PAW/PBE、800eV、8倍ecutrho、3³零位移、MP.02Ry | 保留 | 同物理模型可比性 |

四步Total SCF correction相對Total force約52%、93%、7.2%、2.7%；這不是電子噪音成因的證明，但支持新起點加嚴SCF，避免初期不夠精確的力干擾新的Hessian。[官方troubleshooting](https://www.quantum-espresso.org/Doc/pw_user_guide/node21.html)也建議檢查能量／力一致性並降低conv_thr或提高upscale。本次不額外改upscale、Wolfe參數或trust radius上限。

從最後已完整SCF的cell/positions（原始行8335/8340）接續，新電子初態與BFGS history，無wavefunction checkpoint重用。全自由晶胞vc-relax、三方向PBC；ct56、29MPI ranks、64GiB，Slurm48小時／QE47小時。單一缺項只需一個工作，不對其他構型設單工限制。

不直接將vc-relax的ion_dynamics替換為FIRE：[QE IONS/CELL定義](https://www.quantum-espresso.org/Doc/INPUT_PW.html)要求cell BFGS配ion BFGS，FIRE列於固定晶胞relax；damped cell dynamics若後續需要，須另定對應方法及驗收。這不是現在最小的修復。

## 完成條件及未成功時的處理

必須同時有明確BFGS convergence、relaxed-cell final SCF、電子精度≤1e−10 Ry、最大力分量≤0.0005 eV/Å、最大力向量≤0.001 eV/Å、完整殘餘stress各分量≤0.5 kbar、正常結束及輸入／輸出／偽勢來源雜湊通過。不得以JOB DONE或Slurm COMPLETED取代科學驗收。

若新attempt仍未達標，先依實際停止分支及同幀力／SCF／應力重新診斷，再選擇可追溯的接續方案，不盲目重送相同輸入。成功後才計算同方法1V形成能與全部六組QE結合能，補入報告。此前14頁報告保持歷史快照，不倒改數值。
