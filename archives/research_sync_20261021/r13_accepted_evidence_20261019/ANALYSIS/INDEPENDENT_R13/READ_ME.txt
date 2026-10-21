2255218 R13_0V 獨立資料驗收（2026-10-19，台灣）

本資料夾讀取已下載的輸出、密度陣列與 ASE SinglePoint 軌跡，沒有重新執行 DFT。

結果：596/596 檢查通過。46 個完整離子步，185 次電子計算皆原生成功；
46 次線搜尋皆 CONVERGENCE，從保存的原子力／應力重建後，下降方向、
Armijo 與曲率條件也逐步通過。最終 bundle、結構、力、電子數、optimizer_state
均對應第 46 步。程式輸入與凍結版本的 hash 保持一致。

Fmax = 0.000976534043 eV/Å；filter Fmax = 0.000976679521 eV/Å。
sigma_zz = +0.000040852086 GPa；原門檻 0.001 eV/Å 與 0.05 GPa 保持。
數值鬆弛端點通過，並不代表已驗證網格／真空／所有尺寸的取樣敏感度。

本輪能量由 -4760.084413242334 降至 -4761.035407359499 eV，
降低 950.994117 meV；最後 5 步只降 0.038779 meV。
Lz 由 13.497656993076 增至 13.739937163017 Å（+1.79498%）。
扣除均勻晶胞形變後，逐原子位移最大 0.80224 Å、RMS 0.55270 Å。
第 8 步有較大的構型移動與能量下降，力一度增加後才逐步降低。
所有保存幀都未出現原子重疊，最短 Al–Al 距離最小 2.64228 Å。

因此本輪包含明顯結構重排，不能僅稱為微小的數值收尾。
R13_0V 可升級為在目前模型下通過力與應力門檻的端點。
後續仍須對照 R13_1V 的結構與局部配位，再解釋兩者的能量差。
本檢查沒有判定 FCC 晶相保持，也沒有證明這是全域最低能結構。

檔案：
  QA.json：596 項檢查。
  audit_summary.json：主要指標、來源 hash 與解釋範圍。
  trajectory_metrics.csv：含第 0 步的 47 幀能量、力、軸長與非仿射位移。
  completed_step_diagnostics.csv：46 個完整步的線搜尋核對與逐步變化。
  audit_saved_results.py：可攜資料分析腳本（numpy/scipy/ASE）。
  ase.filters.py：遠端實際 ASE 3.25 filter 原始碼副本，用於重建廣義力。

重現：在此資料夾執行 python audit_saved_results.py
或指定原始資料根目錄：python audit_saved_results.py /path/to/R13_AXIAL_CONTINUATION_RESULTS
執行會重寫本資料夾的分析 JSON/CSV，保持 R13_0V 原始資料不變。

注意：completed_step_diagnostics.csv 的 trapezoid_residual_meV 只是端點梯度
梯形積分的近似殘差。第 8 步形變大，沿線具有明顯非線性；這一近似不能當成
完整能量／力一致性測試。正式逐步判斷使用保存的原生線搜尋狀態與 Wolfe 條件。
