# 2026-10-21 Lüder要求先聚焦QE空缺參考

老師指出10/20更新自第3頁起難理解：historical data未說明用途、a0/B0來源與差異未解釋，並要求先做QE vacancy/divacancy reference、完成資料單獨放共享OFDFT下workitem資料夾、補vacancy-site分析、替換mu(Al)記號。

## 本輪處理

- 本機新資料夾：`C:/OFDFT/NAS_SHARE/workitem_QE_bulk_vacancy_reference_20261021`。使用者先前指定NAS手動上傳仍有效；本輪未上傳、未寄信、未查遠端或提交新DFT。
- 分享包只含已完成的QE k4 bulk 0V/1V原始輸入、輸出、XML、結構、偽勢與其後處理；兩頁英文PDF和可攜重算腳本隨包。未完成k4雙空缺、舊k3配對與單點診斷不當成完成參考放入。
- 兩組通過BFGS/final-cell-SCF/XML、Fmax≤.001、component≤.0005eV/Å、全應力≤.05GPa。E0=-58045.77533919631、E1=-57507.63652052896 eV；Ef1=.677935897 eV。這是完成k4鬆弛，較密k點／smearing／有限尺寸驗證仍待完成，整個reference work item未結案。
- k4 1NN/2NN/5NN於10/15因五尺寸優先而正常停止，沒有合格終態。JOB DONE和return0不能替代BFGS/final-SCF。後續先檢查存檔與fresh遠端狀態，從已凍結QE流程續算，不把尚未完成的2V寫入正式binding表。
- 新寫作統一 `E_Al^ref = E_0V/108` 或機讀 `E_ref_Al`，稱完整bulk每原子參考能；值-537.4608827703362 eV/atom。Ef(nV)=E_nV-E_0V+nE_Al^ref；Eb(2V)=2Ef(1V)-Ef(2V)。原始凍結記錄不倒改。
- vacancy site為0V第44原子（zero-based43），分率約(.5,.5,.5)。107個原子唯一對應；去仿射與平移後RMS .017091998Å、12NN平均內移.047497659Å；參考殼半徑2.853648719→2.806151060Å。12近鄰原子CN12→11，cutoff3.1/3.2/3.3Å一致；其餘95原子CN12。初末態空格一致，非遷移路徑證明。終態最近影像12.107006521Å，ΔV=-4.183516836Å³。

## a0/B0說明

舊第3頁將QE primitive EOS的40³/80³ k點與OF108原子78³/96³實空間網格放在同一表，兩套模型與測試量沒有說清楚。後續用具體計算名稱，避免單寫historical。
新QE k4 0V的等體積晶格參數4.038837655116Å，來自vc-relax晶胞，並非新EOS擬合；本輪沒有新B0。早期QE原胞EOS4.039825Å/77.991GPa與OF3.947426710Å/50.191325962GPa各自保留其來源，不能把現行PAW/PBE設定補造為早期high-k未找回的輸入，也不能把兩方法差異全歸因於KEDF。

完整本機查核在本對話 `work/luder_qe_ref_20261021`；分享包只取完成資料與直接評估。未擴研究矩陣，也未修改原成功流程。
