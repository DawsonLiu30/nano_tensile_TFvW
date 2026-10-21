# 2026-10-14 proposal 正式稿

成品：`C:/OFDFT/PROPOSAL/鋁空缺與奈米柱研究計畫_正式版_20261014.docx`。

## 本次修訂

1. 第三章依 Sholl 與 Steckel（2009），第 8–15 頁核對 Born–Oppenheimer、電子密度、Hohenberg–Kohn、Kohn–Sham 方程與非交互作用動能。書籍 DOI：10.1002/9780470447710。TFvW 另依 OF 文獻，不把課本未涵蓋的模型歸於課本。
2. 第四章加入前期 KEDF 比較、100 組權重掃描、77 組合格候選與五組代表候選。原 Excel 的完整晶體數值與歷史掃描相互核對；選參歷史參考為 QE 64/63 的 0.601167 eV。說明選 .9/.1 的形成能依據及與晶格最佳候選的取捨。
3. 摘要、第五與第六章採已驗收 R20 終態：Ef=.47675189365281767 eV；與暫定值差 +.8071027696 meV。加入四組奈米柱、540 個共同格點的逐原子比較。結果已完成的結論直接陳述，數值取樣與 OF stress 待驗範圍仍標清。

## 檔案位置

- 新增可分享證據：`C:/OFDFT/PROPOSAL/SOURCES_20261014`，含來源說明與 SHA256 manifest。
- R20 原始資料：`C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/QE/08_Nanocolumn_R20_final_SCF_ACCEPTED_20261014`。
- 形成能、成對位移、驗收記錄：`C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/REPORTS/nano_progress_20261014`。
- 建置／分析程式：`C:/Users/m1331/Documents/Codex/2026-09-24/g/work/proposal_final_20261014`。
- 文件驗證：上述工作目錄 `render/final_document_audit.json`；視覺檢查 `render/final_visual_qa.json`。內部 PDF 與 PNG 是排版檢查檔，不作另一份正式報告。

## 驗證記錄

- 凍結 runner inspect accepted；238 項 R20／四組 nano 分析檢查通過。
- 100 組掃描的形成能由原始能量重算；77 個合格候選與 Excel 三種量一致。四個不合格候選的歷史快照差異保留於來源紀錄。
- 新增 132 個證據檔 hash 核對；39 個既有 canonical 來源未更動。
- 22 項文件自動檢查通過，包括公式逐式編號與解釋、字體、引文、數據來源及 R20 同方法原子庫。
- Word 原生渲染後逐頁檢查；後續小幅版面修訂以逐頁像素 hash 確認未變頁，再檢查變更頁。

本輪不含新 DFT、參數更動、定時監控或對外傳送。
