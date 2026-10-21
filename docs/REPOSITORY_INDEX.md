# 研究程式與資料位置索引

整理日期：2026-10-21。使用分支 `codex/portable-qe-20260923`。

## 程式入口

| 工作 | 入口 | 使用說明 |
|---|---|---|
| FCC supercell 與奈米柱 | `app/ase_nanocrystal.py` | 支援圓形與多邊形截面；半徑採幾何裁切 |
| Bulk／空缺 campaign | `campaigns/`、`scripts/` | 依各日期 manifest 選擇，避免用舊預設取代已驗證設定 |
| 凍結執行版本 | `workflows/` | 保留已用過的版本、父來源、輸入與程式檢查紀錄 |
| 目前 QE bulk 續算 | `workflows/al_bulk_reference_resume_k5_20261021_v2/` | v1 保留為歷史；是否完成以最新結果證據為準 |
| R13 已驗收續算版本 | `workflows/al_nano_R13_axial_continue_20261018_v2/` | 使用範圍與後驗收證據見登錄；不擴張成所有模型均已驗證 |
| 本機／遠端提交模板 | `hpc/` | 保存歷史配置，不因 clone、pull 或檔案驗證而執行 |
| 流程登錄 | `docs/validated_workflows.json` | 機讀來源、狀態、檢查與結果證據 |

先讀 [收斂基準](CONVERGENCE_BASELINES.md)及[流程登錄](VALIDATED_WORKFLOWS.md)。此次整理沒有改寫計算核心，也沒有執行或重送計算。

## 本次補入的原始來源與證據

下列路徑位於 `archives/research_sync_20261021/`。檔案由既有本機資料包逐位元複製；[IMPORT_MANIFEST.json](../archives/research_sync_20261021/IMPORT_MANIFEST.json)記錄原位置、大小與 SHA-256。

| 子目錄 | 內容 |
|---|---|
| `frozen_sources_20261015/` | 登錄所引用的 QE bulk、QE nano、DFTpy nano 凍結程式 |
| `qe_nano_inputs_20261015/` | 10/15 QE 輸入、程式、偽勢、測試 fixture 與 manifest |
| `dftpy_nano_inputs_20261015/` | 10/15 DFTpy 輸入、程式、偽勢與 manifest；大型密度依原索引取用 |
| `qe_bulk_accepted_20261017/` | 已驗收 0V／1V 的原始輸入、輸出、結構與 XML；不含未完成雙空缺結果 |
| `r13_accepted_evidence_20261019/` | R13 後驗收原始紀錄、結構與既有分析；大型 checkpoint 以 hash 引用 |

原文件中的 Windows／HPC 絕對路徑保留為來源證據，不做全文路徑替換。其他電腦可依本索引找到已收錄副本；需要外部密度或 checkpoint 的執行仍須另行取回資料並核 hash。這不是可直接重啟全部作業的完整備份。

## GitHub 外的資料

- `C:/OFDFT/NAS_SHARE/`：本機原始研究資料包與分享版本；目錄名稱不代表已上傳 NAS。
- `C:/OFDFT/AL_DEFECTS_USB_HANDOFF_20260717/`：歷史原始數據與環境來源。
- `C:/Users/m1331/Documents/Codex/2026-09-24/g/work/`：研究對話的工作目錄；此次只保存已辨識並歸檔的來源，不宣稱整個對話目錄已備份。
- HPC 的 scratch、波函數、密度與大型 checkpoint：依各 workflow manifest 取用。

私人往返信件、登入憑證、暫存與執行中的 scratch 不納入本次提交。已凍結的來源與證據只新增版本或旁附索引，不覆寫原執行。
