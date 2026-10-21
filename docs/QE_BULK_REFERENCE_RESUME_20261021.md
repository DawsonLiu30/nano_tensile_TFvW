## 2026-10-21 21:33 QE重啟與迭代已實測確認

21:33 台灣時間啟動查核完成：三個v2續算已實際讀到保存密度與波函數，進入第16次SCF；原子初猜警告及讀檔回退均未出現。0V/1V k5均進入第1次SCF。五個計算RUNNING；三個後續k5等待各自k4通過，沒有新完成端點。三個v1正常停止、舊依賴取消。已停止遠端查詢；沒有持續監控。

# QE bulk續算與k點驗證：2026-10-21

最新：QE bulk 續算與 k 點檢查，2026/10/21

本資料夾是執行紀錄；尚未新增已完成的雙空缺結果。
完成資料另在 workitem_QE_bulk_vacancy_reference_20261021；NAS由使用者手動上傳。

目前應追蹤：
2286031_2/3/4：1NN／2NN／5NN 的 k4 全晶胞 vc-relax 續算，三組可同時執行。
2285949_0/1：已完成 k4 0V／1V 的 k5 固定幾何 SCF，沿用第一次提交，不重算。
2286032_2／2286033_3／2286034_4：各雙空缺的 k5 SCF；各自 afterok 對應續算且程式重驗 k4 終態。
2286038：一次性 k4/k5 五態配對分析。它不會再提交計算，不是持續監控。

分區：
04_CORRECTED_RESTART_V2：目前實際續算腳本、原凍結核心、續算輸入、偽勢與 hashes。
05_V2_SUBMISSION_AND_STARTUP：正式回執、程式查核、最終啟動證據。
03_K5_REFERENCE_INPUT_PREVIEWS：0V／1V k5 輸入預覽，與實際執行的結果分開。
01_FROZEN_EXECUTION／02_SUBMISSION_AND_QA：v1完整保留，供追查修正原因。

v1 的三個續算啟動出現「startingwfc atomic+random may spoil restart」警告，遂正常停止；當時尚未取得新的鬆弛端點。其舊後續作業2285950/51/52/53已取消。
v2從原10/15的357個保存檔接續，明訂restart_mode=restart、startingpot=file、startingwfc=file。
舊已成功的nano wrapper保留原子初猜；不應直接把該操作選項套到第一輪SCF中斷案例。
實際是否讀到檔案與迭代前進，以05中的啟動證據為準。v1/v2來源均保持原樣。

科學設定保留：QE7.5、PAW/PBE、800eV、ecutrho=8倍、MP .02Ry、200bands、三方向PBC。
續算：29 MPI ranks、64GiB、k4、零壓力全晶胞BFGS；72h/內部255600s。
k5：29 ranks、128GiB、72h；沿已驗收k4幾何，disk_io=medium為磁碟／記憶體策略調整。
Fmax≤.001 eV/Å、最大分量≤.0005、全應力分量≤.05GPa、最後SCF≤1e−10Ry，另要求BFGS與最後晶胞SCF完成。
k5是固定幾何診斷；電子收斂成功與合格鬆弛終態分開。每個網格各自用 E_ref_Al=E0/108 計算形成／結合能。
5meV是這次敏感性判讀解析度，k4→k5單步通過不等於全面取樣、smearing或有限尺寸收斂。
原胞40³/80³ EOS與800eV校準依docs/CONVERGENCE_BASELINES.md；未把原胞取樣搬成超晶胞收斂證明。

27項程式檢查（原23項加4項file-mode／設定保留檢查）通過，不是27個DFT物理結果。
本輪只查核啟動，不建立持續監控。原始工作與完成workitem不改。
官方依據與程式來源見05/REVIEW/RESTART_FILE_MODE_REVIEW.txt。
