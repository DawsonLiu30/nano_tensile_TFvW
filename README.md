# 鋁體相與空位研究：跨裝置工作入口

更新：2026-09-24。這個 repository 保留 `nano_tensile_TFvW` 的歷史名稱；目前重點是 bulk／單空位校準、雙空位構型比較，以及 QE 對照。舊拉伸與奈米線腳本是歷史工作，並非本輪執行入口。

## 先看這裡

- **[在國網 iService／其他電腦執行 QE](docs/CROSS_DEVICE_QE.md)**：下載、環境設定、驗證、執行與結果回傳。
- **[本次 Taiwania3 部署](docs/TAIWANIA3_20260924.md)**：已取得遠端環境紀錄；專用 helper 先檢查，再提交同站 1NN 基準。
- **[最新可攜 QE 計算包](campaigns/qe_divacancy_20260923/README.md)**：四個精確輸入、同一份偽勢、SHA-256、來源與完成證據。
- [科學條件與狀態 manifest](campaigns/qe_divacancy_20260923/manifest.json)。
- [先前 Windows／WSL 操作紀錄](docs/LOCAL_WORKFLOW_20260908.md)：保留原機器與 USB 資料位置。

本次同步分支是 `codex/portable-qe-20260923`。尚未合併到 `main`；另一台電腦要指定這個分支。

```bash
git clone --branch codex/portable-qe-20260923 --single-branch https://github.com/DawsonLiu30/nano_tensile_TFvW.git
cd nano_tensile_TFvW
python3 scripts/qe_portable.py verify
```

`verify` 只檢查輸入，不啟動 QE。執行工具只需要 Python 3.10+ 標準函式庫；計算另需 Linux／WSL 的 QE。提供的最小環境是 `env/qe-linux.yml`（Python3.11、QE7.5）。完整 DFTpy 舊環境規格仍在 `environment-wsl.yml`。

## 尚缺的 QE 計算

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
