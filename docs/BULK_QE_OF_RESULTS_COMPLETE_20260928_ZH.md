# Al vacancy bulk 結果補齊

2026-09-28 09:59 台北，使用者重新登入後查核 QE 1V 修復 job 2162428。作業已於03:13:58完成，耗時02:30:50；專用 frozen inspector 與獨立 raw 輸出驗收均通過。BFGS 為3個SCF cycles／2步，之後另有第4次 final relaxed-cell SCF。

最終力分量0.000463055718 eV/Å，小於原輸入0.0005；最大力向量norm0.000463055860，小於0.001；SCF accuracy8e-13Ry，小於修復的1e-10。應力通過既定0.5kbar門檻；印出零應力只表示輸出精度內。此次沒有降低力或應力驗收門檻。

本輪bulk現在DFTpy8/8、QE8/8。QE單空缺形成能0.644962821749eV；六組binding（1NN至6NN，正值為有利）分別為−2.327662、22.679874、50.146231、16.902489、56.680501、23.315124meV。1NN接近零，其物理正負仍需finite-size／kmesh／smearing誤差界；5NN存在双等距週期影像，不是孤立雙空缺最佳距離證明。不同方法使用不同XC和偽勢，誤差不能單獨歸因TFvW。

新下載49檔逐一SHA核驗；合計三批746原始檔，1,054,805,876bytes未壓縮，保留charge density/XML及source snapshots。遠端119,171,159,040bytes WFC/mixing未下載未刪除，本包不是完整電子restart備份。

新分類包：`C:/Users/m1331/Documents/Codex/2026-09-24/g/outputs/al_proposal_results_complete_20260928`；舊報告與原始封存保留。140個原數值欄位不變，独立合併交叉核查987項通過；詳細內容見00_README及07_independent_QA。

動畫沒有呈現空位逐漸靠近，因為既有靜態relaxation各幀的格點占據未變。空位移動須由Al原子跳入空格，讓未佔據格點改變；目前optimizer軌跡沒有建立這種遷移路徑，也不代表物理時間。803幀舊軌跡包不含這次新增repair四幀，避免混淆版本。NEB等路徑計算需另設端點與跳躍路徑。

Proposal尚待新協定nanocolumn結果、關鍵能量差的有限尺寸／數值誤差，以及依研究結論所需的彈性或遷移路徑證據。本次只查一次實際狀態、取回和後處理，未新增計算或開啟持續監控。
