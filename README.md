# 鋁體相與空位研究：跨裝置工作入口

## 目前入口：2026-10-21 GitHub 同步版

研究分支為 **`codex/portable-qe-20260923`**。GitHub 預設的 `main` 仍是較早版本；請使用本分支。

- [專案與資料位置索引](docs/REPOSITORY_INDEX.md)：結構生成、計算流程、證據與外部大檔的位置。
- [已驗證流程登錄](docs/VALIDATED_WORKFLOWS.md)與[機讀登錄](docs/validated_workflows.json)：區分程式測試、實際啟動及物理端點驗收；使用前先查此處。
- [收斂基準](docs/CONVERGENCE_BASELINES.md)：保留既有校準與各模型的適用範圍。
- [最新 QE bulk 續算與 k5 紀錄](docs/QE_BULK_REFERENCE_RESUME_20261021.md)：目前使用 v2；啟動證據不等於新結果已驗收。
- [Git push／pull 操作](docs/GIT_SYNC.md)與[保存內容驗證](docs/git_sync_20261021/SHA256SUMS.json)。

本次保留既有程式、輸入、凍結版本、hash 與驗收證據。`archives/research_sync_20261021` 補存原本位於本機資料包的小型來源與結果證據；原始資料包未改動。完整密度、波函數及大型 restart 仍由外部資料索引管理。

```powershell
git clone --branch codex/portable-qe-20260923 https://github.com/DawsonLiu30/nano_tensile_TFvW.git
cd nano_tensile_TFvW
python scripts/verify_research_snapshot.py
```

驗證指令只檢查保存檔案，不啟動 DFT 或提交作業。

---

以下保留 2026-09-25 的工作入口與歷史說明；當時的「最新」以該日期為準。

更新：2026-09-25。這個 repository 保留 `nano_tensile_TFvW` 的歷史名稱；目前重點是 bulk／單空位校準、雙空位構型比較，以及 QE 對照。舊拉伸與奈米線腳本是歷史工作，並非本輪執行入口。

## 先看這裡

- **[Proposal最新執行稿v0.3（9/25晚間）](docs/PROPOSAL_CURRENT_STATUS_20260925_ZH.md)**：已驗證bulk初步結果、兩方法差異、四主體完成/待补矩陣、9/28確認稿優先順序。
- **[最新：2026-09-25 bulk 晶胞鬆弛](docs/AL_BULK_VACANCY_VCRELAX_20260925.md)**：依使用者更正，QE維持PAW/PBE、DFTpy維持LDA/local；共同12.119544 Å初態，八組獨立晶胞鬆弛。DFTpy使用ctest，與舊固定晶胞資料分開。
- **[2026-09-25 QE 條件來源與最新進度](docs/QE_PARAMETER_BASELINES_20260925.md)**：簡報既有 cutoff／k 點測試、單位、不同晶胞／偽勢協定的界限，以及 array `2151591` 的已完成、最佳化停止與排隊狀態。
- **[本輪 bulk 空位鬆弛執行紀錄](docs/AL_BULK_VACANCY_EXECUTION_20260924.md)**：8 組共同理想初態、DFTpy 本機結果、Taiwania3 新 QE `relax` array `2151591`、來源 hashes、輸出位置與只讀監控指令。
- **[2026-09-24 會後 proposal 架構 v0.2](docs/PROPOSAL_HANDOFF_20260924.md)**：bulk Al與晶態Al奈米柱的空缺形成／聚集能量與結構；包含TFvW、QE、線性彈性、計算清單及未寄出的確認信。
- **[在國網 iService／其他電腦執行 QE](docs/CROSS_DEVICE_QE.md)**：下載、環境設定、驗證、執行與結果回傳。
- **[本次 Taiwania3 部署](docs/TAIWANIA3_20260924.md)**：已取得遠端環境紀錄；專用 helper 先檢查，再提交同站 1NN 基準。
- **[前一輪固定幾何 QE SCF 計算包](campaigns/qe_divacancy_20260923/README.md)**：四個精確輸入、同一份偽勢、SHA-256、來源與完成證據；與本輪理想初態原子鬆弛分開保存。
- [科學條件與狀態 manifest](campaigns/qe_divacancy_20260923/manifest.json)。
- [先前 Windows／WSL 操作紀錄](docs/LOCAL_WORKFLOW_20260908.md)：保留原機器與 USB 資料位置。

本次同步分支是 `codex/portable-qe-20260923`。尚未合併到 `main`；另一台電腦要指定這個分支。

```bash
git clone --branch codex/portable-qe-20260923 --single-branch https://github.com/DawsonLiu30/nano_tensile_TFvW.git
cd nano_tensile_TFvW
python3 scripts/qe_portable.py verify
```

`verify` 只檢查輸入，不啟動 QE。執行工具只需要 Python 3.10+ 標準函式庫；計算另需 Linux／WSL 的 QE。提供的最小環境是 `env/qe-linux.yml`（Python3.11、QE7.5）。完整 DFTpy 舊環境規格仍在 `environment-wsl.yml`。

## 前一輪固定幾何 QE SCF：來源快照

| Case | 來源狀態 | 欲回答的問題 |
|---|---|---|
| 1NN [110] | 已收斂，完整輸入／輸出保留 | 與 2NN 形成第一組比較 |
| 2NN [100] | 中斷未完成 | QE 是否支持目前 TFvW 的近鄰偏好？ |
| [310] | 尚未開始 | 教授質疑的低點是否得到 QE 支持？ |
| 中距 [110] | 尚未開始 | 與 [310] 形成第二組比較 |

以上是 2026-09-23 的來源查核狀態，不代表目的地主機已執行。
本輪全部固定各自的 TFvW 鬆弛後幾何，只做 QE SCF。collector 只在原始輸出、輸入與偽勢通過查核後採用能量。沒有兩端點完成的比較保持空白。

目前已完成的 DFTpy 108／256 格點受控比較，存於 campaign 的 `comparison/TFvW_controlled_energies.csv`。它和八月的變晶胞 [110] 結果是不同計算協議。重算支持保留 [310] 低點分析，但還不能宣稱已確認其物理原因或完整準確性。

## GitHub 與大型資料的分工

GitHub 保存程式、精確小型輸入、偽勢、進度與精簡輸出證據。USB 搬來的完整原始資料、已安裝環境與大型波函數仍保存在本機或計算儲存區。這個 clone 不是 5.52 GB USB 快照的完整鏡像。

最新 2NN 中斷 scratch 的完整性不足，本次換機預設從相同科學條件重新開始 SCF。原 scratch 不刪除。新計算必須寫到 repository 外的新 attempt，以免覆寫舊結果；登入資訊也不存入 Git。

國網連線：`ssh dawson666@twnia3.nchc.org.tw`。使用者於 2026-09-24 已登入並取得移轉版 clone；Python、佇列與計畫關聯已有貼回紀錄。QE／MPI 的實際執行與新計算結果尚未確認。請依 [Taiwania3 部署紀錄](docs/TAIWANIA3_20260924.md) 完成現場查核。

## 本機測試

```bash
# 可攜工具測試不執行正式 QE 計算。
python3 -m unittest discover -s tests -p test_qe_portable.py -v
# 完整既有分析測試須在安裝了 numpy/ASE 等套件的環境執行。
python3 -m unittest discover -s tests -v
```

原機器 `C:\OFDFT\AGENTS.md` 的 WSL Sync、執行鎖與歷史資料保留規則仍適用。不要用舊自動上傳／提交腳本代替本次明確分開的 prepare 和 run 流程。
