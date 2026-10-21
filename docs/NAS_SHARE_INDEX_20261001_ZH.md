# Al vacancy 給 Luder 的 NAS 本機資料包

使用者要求將研究數據清楚分類，準備整包自行上傳與 Luder 共用的 NAS。本次只整理本機檔案，沒有查遠端、提交計算或上傳 NAS。

- 主資料夾：`C:/OFDFT/NAS_SHARE/Al_vacancy_Luder_20261001`
- 單一上傳 ZIP：`C:/OFDFT/NAS_SHARE/Al_vacancy_Luder_20261001.zip`
- 中英文入口：資料夾內 `START_HERE.html`、`README_EN.txt`、`README_ZH.txt`
- 回執：`C:/OFDFT/NAS_SHARE/Al_vacancy_Luder_20261001_receipt.json`
- ZIP SHA-256：`f84c755de189ebd09f94ede6b3b23d1d6c5265dbe5b80105d2567c735df51761`

共 1,273 檔；資料夾 1,191,397,653 bytes，ZIP 513,739,609 bytes。資料夾與 ZIP 內容相同，擇一上傳即可。所有包內索引與入口連結使用相對路徑，保留原始計算檔中的舊絕對路徑供追溯。

| 分類 | 內容 |
|---|---|
| 00_INDEX | 16個採用終態、21次原始attempt、20個nano初始結構、19個新提交task，以及逐檔來源與hash索引 |
| 01_REPORTS | 最新9/30修訂PPT、18頁支持報告、曾寄出的PPT及歷史參數測試PPT，版本分開 |
| 02_BULK_RESULTS | 可攜16終態資料包，含canonical CSV/JSON、六幅原圖、圖數據、文獻比較、原始證據子集及核算程式 |
| 03_RAW_HISTORY | 原始bulk、QE continuation、QE 1V repair三批，共746個經原始manifest核驗的檔案，保留失敗歷程 |
| 04_STRUCTURES_TRAJECTORIES | 16個正式終態結構；803幀舊軌跡；可離線使用的1NN/5NN互動檢視器 |
| 05_NANOCOLUMNS | 五種半徑、兩方法、完美／單空缺，共20個初始模型與生成程式、圖 |
| 06_NEW_SUBMISSIONS_NO_NEW_RESULTS | 9/30的提交回執、時間快照、查核與48檔凍結來源；未收錄新計算終態 |
| 07_METHODS_AND_QA | 參數來源、方法與檢查紀錄 |

重要區分：

1. QE配對數據仍待k點等數值驗證，不能把16個optimizer-accepted終態說成所有物理精度皆已確認。
2. 803幀舊軌跡中的QE 1V是修復前失敗歷程；目前合格的1V在正式終態資料夾及repair原始資料中。
3. 新作業2173492、2173494、2173496的最後收錄狀態是9/30台北23:06；本包沒有fresh remote check，也沒有新作業結果。不可因缺新output就重送。
4. 5種半徑均為軸向無限PBC的初始柱；6/8Å已提交，4/10/12Å僅生成結構。原始README中較早的「尚未提交」由包內狀態說明補充，未倒改原檔。
5. 已選擇下載的本機原始證據完整保留；大型遠端QE wavefunction/mixing scratch未在先前下載，NAS包也不含它，不是完整restart備份。歷史QE EOS原始輸出仍未找回。

檢查完成：1,253個複製檔案逐一與來源hash一致；746個原始檔符合歷史manifest；搬移後370項bulk數據核算通過；21個HTML包內連結存在；ZIP全部1,272個manifest payload hash與額外manifest檔自身內容核對通過。

NAS端可執行 `python verify_package.py` 檢查整包，或 `python 02_BULK_RESULTS/verify_data.py` 以NumPy核算原始數據；皆不執行DFT。未來新增結果宜另建有日期的新版本，保留本包及其hash回執。
