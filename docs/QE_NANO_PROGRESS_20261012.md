# 2026-10-12 R20 完成與執行中查核

18:58 原始檔快照、19:08 台灣時間最後短查核。完整結果與數據讀本機 [REPORT_ZH.txt](C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/REPORTS/nano_progress_20261012/REPORT_ZH.txt)，新原始資料在 `C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/QE/05_Nanocolumn_R20_restart_20261010`。

- 2221957_0：07:14完成；凍結runner驗收通過。176 atoms，最終SCF能量−94548.01223174046eV，Lz13.80273214568Å，最大原子力.00045886987eV/Å。
- 2221957_1：仍RUNNING；電子迭代在本次兩個快照間增加4次，29 ranks完整。最新完整步最大原子力.00108565444、最大分量.00100864948eV/Å，未達.001/.0005門檻，正式形成能未填。
- 28個新原始檔逐檔hash，80個新完整SCF與65個前次完整SCF做XML/stdout對位；432項完整性及分析檢查通過。不是新增物理精度或取樣收斂驗證。
- QE原子最佳化末步與final relaxed-cell SCF不能混用：本次0V後者高.920791meV，採後者作正式終態能量。
- QE7.5 XML step.scf_error是Ry、output.scf_error是Ha。來源：[add_qexsd_step](https://github.com/QEF/q-e/blob/qe-7.5/PW/src/add_qexsd_step.f90)、[pw_restart_new](https://github.com/QEF/q-e/blob/qe-7.5/PW/src/pw_restart_new.f90)。原production final-output換算正確。
- 未重送、取消、改門檻、增加監控或上傳NAS。原始資料歷史及校準限制不變；後續先讀CONVERGENCE_BASELINES.md。
