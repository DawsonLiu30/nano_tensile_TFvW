# Bulk QE與DFTpy本機歸檔結果

2026-09-26 17:22台北查核後已完成歸檔分析。DFTpy8/8accepted；QE0V、2NN、4NN、6NN可用。1V需修復BFGS，1NN／3NN／5NN需更長接續。QE單空缺形成能及全部binding仍缺合格1V參考。

- [六頁分析報告](../../RESEARCH_UPDATES/AL_BULK_VCRELAX_20260926/Al_bulk_QE_DFTpy_report_20260926.docx)
- [完整數據分析](../../RESEARCH_UPDATES/AL_BULK_VCRELAX_20260926/analysis/summary_zh.md)
- [CSV](../../RESEARCH_UPDATES/AL_BULK_VCRELAX_20260926/analysis/accepted_comparison.csv)
- [原始歸檔清單](../../RESEARCH_UPDATES/AL_BULK_VCRELAX_20260926/ARCHIVE_MANIFEST.json)
- [本機副本校驗索引](../../RESEARCH_UPDATES/AL_BULK_VCRELAX_20260926/LOCAL_ARCHIVE_INDEX.json)

已下載534原始檔，逐檔SHA驗證，包含density及source snapshots。大型波函數／mixing history共71.70GB未搬回、未刪除，完整清單在manifest中。壓縮包334.8MB，不是完整restart備份。

QE已完成的雙空缺形成能為2NN1.267246、4NN1.273023、6NN1.266611eV；2NN與6NN差0.635meV，不据此確定穩定性排序。與OF比較同時包含XC／偽勢／KEDF／展寬／結構差異。不同方法原始總能不直接相減。

後續已將活躍bulk runner及sbatch預設修訂為48hSlurm／47hQE，17項tests與八case預檢通過。既有immutable snapshot與歷史回執維持原12h／11h；尚未新部署或提交。每10分鐘監控仍已刪除。
