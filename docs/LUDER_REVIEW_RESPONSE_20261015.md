# Lüder 回饋處理紀錄，2026-10-15

## 交付

本機增量包：`C:/OFDFT/NAS_SHARE/LUDER_NAS_UPDATE_20261015`。
主簡報：`REPORTS/Luder_response_20261015/20261015_Al_vacancy_review_response.pptx`，13頁，6個可編輯原生圖表。
同資料夾有中文處理說明、修改對照表、逐端點力、位置及週期距離、能量分項、產表程式、官方來源與新計算回執。尚未上傳NAS或寄信。

完整工作目錄：`C:/Users/m1331/Documents/Codex/2026-09-24/g/work/luder_response_20261015`。
原簡報與canonical原始資料均保留；修訂簡報撤下高力固定幾何的配對能量圖。

## 本次核對與科學狀態

- 35筆力資料中，24筆當前鬆弛終態全部達norm力門檻0.001eV/Å；11筆是固定幾何SCF診斷，7筆k4缺陷超標。前者不是所有歷史校準的統計，後者不能當k4平衡配對排序。
- 力達標仍需k點、smearing、真空、軸週期的轉用驗證。OF nano四組繼續列pilot，尚待總stress與重新鬆弛驗證。
- 四個1V最短影像距離13.397763–13.825627Å，全部符合老師≥10Å要求。空缺近軸父格點CN12、未發現換父格點；R15第一近鄰部分本來就位於低配位表面。
- QE R15/R20形成能0.747117974/0.476751894eV；OF0.598150643/0.631826232eV。分別使用本方法bulk每原子化學勢。兩尺寸對比保留，一般半徑趨勢尚待更多尺寸與取樣檢查。
- QE R20最大位移比R15小，第一近鄰平均內移卻較大（0.067309對0.059237Å）；兩種結構量不互相替代。
- OF分解加總已閉合。QE native能量分項已閉合，但one-electron不是純Ts；完整PAW Ts/Vei尚待後處理。官方QE/DFTpy來源在energy_audit/primary_sources.json。
- 五尺寸R/a0=1.1/1.3/1.5/1.7/2.0，完整柱52/76/96/128/176原子。新增三尺寸只有結構，沒有新能量或生產作業。

## 已提交計算

正式array **2245215**，2026-10-15 19:13:44台灣唯一快照：全部PENDING/QOSGrpJobsLimit。沒有監控。

0V、1V、1NN、2NN、5NN；ct56/mst114175，29 ranks、64GiB、72hSlurm/71hQE、並行5。
PAW/PBE800eV、ecutrho8倍、k4、MP0.02Ry；全cell vc-relax；norm力≤.001、分量≤.0005eV/Å；六應力分量≤.05GPa；final rebuilt-cell SCF≤1e-10Ry。
k4是驗證已發現的殘餘力與幾何改變，不是已收斂的最終取樣；後續matched k5等檢查仍需做。

Snapshot：`/work/dawson666/src/campaign_snapshots/al_bulk_k4_validation_20261015_20261015T111240Z`。
Runroot：`/work/dawson666/qe_cases/qe_runs/AL_BULK_K4_VALIDATION_20261015`。
回執：`work/luder_response_20261015/bulk_k4_validation_20261015/submission_receipt`。
test-only2245212不是正式作業。提交前queue為空、無重複campaign；34檔部署hash核對通過，詳細準備/runner測試在package。

## 驗證紀錄

力查核226項；既有幾何44項、新結構57項；能量加總18項；簡報最終資料與來源查核24項。PowerPoint全13頁渲染並逐頁視覺檢查；修訂未變更頁由圖像hash對照。這些是資料/程式/版面查核，不是新的DFT收斂測試。
正式簡報SHA256：977a54c711447ec00cacca6344536daf57098221fd5b8e788024ec12cbf2d2d5。
