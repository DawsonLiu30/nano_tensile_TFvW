# QE 條件來源與進度查核：2026-09-25

**2026-10-11 長期基準入口：** 已逐頁重讀原六頁 PPT、保存原件／原圖／全部顯示數值與找回CSV，見 [CONVERGENCE_BASELINES.md](CONVERGENCE_BASELINES.md)。後續計算與報告先讀該文件；下列9/25進度保留為歷史，不作目前設定或收斂認證。

**後續修訂：** 使用者稍後明確更正「QE維持PAW/PBE、DFTpy維持LDA/local」，並指定晶胞鬆弛。已找回June16實際PAW/PBE、800 eV、3³ k點vc-relax輸入；新協定見 [2026-09-25晶胞鬆弛紀錄](AL_BULK_VACANCY_VCRELAX_20260925.md)。以下保留當時固定晶胞批次的查核，不再代表最新執行設定。

使用者指定以既有收斂測試為依據，不可任意設定 ecut 等條件。本次查閱 `qe_bulk_vacancy.pptx` 全部 6 頁、空白備註、嵌入圖表，以及專案的歷史資料鏈。簡報未修改。

## 最新實際進度

遠端查核時間：2026-09-25 16:47:59（台北）。本輪為 `2151591`，固定共同晶胞的 QE relax。

| Case／array index | 狀態 | 已查證內容 |
|---|---|---|
| 0V／0 | 已完成、可用 | 電子與 BFGS 收斂；最大力 0.000004401 eV/Å；來源與輸出 hashes 通過 |
| 1V／1 | 原子最佳化停止、不可當完成結果 | 18 次 SCF 都收斂；17 個 BFGS 步後 trust radius/history 停滯 |
| 2V 1NN [110]／2 | 已完成、可用 | 電子與 BFGS 收斂；最大力 0.000482081 eV/Å；來源與輸出 hashes 通過 |
| 2V 2NN–6NN／3–7 | PENDING | `QOSGrpJobsLimit`，尚未開始 |

DFTpy 八組均已完成；QE 缺少合格的單空缺參考，目前不可製作完整已收斂的 QE 結合能表。

1V 的最後 SCF accuracy 為 3.8×10⁻¹² Ry。QE 明示 `bfgs failed after 18 scf cycles and 17 bfgs steps, convergence not achieved`。其最大力向量為 0.000960307 eV/Å，雖低於外部 0.001 門檻，最大分量仍為 3.735×10⁻⁵ Ry/bohr，高於該輸入更嚴格的 1.944690×10⁻⁵ Ry/bohr，因此不能標成 QE BFGS 已收斂。

本機 parser 已修正：原先把 BFGS 的 `convergence not achieved` 誤認為電子失敗，現只用獨立 SCF 訊息判斷電子收斂，並記錄 `ionic_optimizer_failed / bfgs_trust_radius_exhausted`。真實停止片段及 CRLF 回歸測試通過，共 14 項測試。原始遠端 runner、attempt/result 與輸出未倒改。這次沒有更改計算參數、取消或重送作業。

## 簡報確實完成過哪些測試

| 項目 | 簡報證據 | 適用範圍 |
|---|---|---|
| Bulk cutoff | 第 2–3 頁：300、400、500、600、800、1000 **eV**，固定 16³ k 點 | 1 atom primitive FCC 的 EOS／B₀ |
| Bulk k 點 | 第 3、6 頁：40³ 與 80³，a₀ 差 0.000040 Å，B₀ 差 0.028 GPa | 簡報認為 40³ 足夠；不能直接把 40³ 套進 108 格點超晶胞 |
| Vacancy cutoff | 第 4 頁：5³ k 點下，400／500／600 eV 的 E_f 為 0.572389／0.572440／0.572529 eV | primitive 4³、64→63；400→600 eV 只差約 0.140 meV |
| Vacancy k 點 | 第 4 頁：600 eV 下測 1³–6³；5³ 約 0.573 eV，6³ 約 0.615 eV | 有測試，但這一段約 42 meV 差值未支持 5³ 已達 k 點極限 |
| 引用的 vacancy 結果 | 第 4、6 頁：600 eV、5³ 的 E_f≈0.5725 eV | 這是投影片採用的結果，不等於所有數值條件都已獨立收斂 |

40³ bulk 結果為 a₀=4.039825 Å、B₀=77.991 GPa；80³ 為 4.039865 Å、77.963 GPa。PPT 的 vacancy cell 向量長約 11.43 Å、體積約 1054.9 Å³，是非正交 primitive 4³，不能視為邊長 11.43 Å 的立方 conventional cell。

單位換算：600 eV = 44.09918661 Ry；60 Ry = 816.34158738 eV。簡報沒有明列 bulk high-k 測試的最終 cutoff，也沒有明列 QE 的偽勢、XC、ecutrho、smearing／width 或原子力門檻。第 5 頁 PBE-GGA、BLPS、fmax<0.02 等屬 DFTpy 欄，不能充作 QE 輸入證據。

## 三組來源不能混用

| 條件 | 歷史 bulk EOS 的準備程式 | 簡報 vacancy 系列 | 本轮實際已提交輸入 |
|---|---|---|---|
| 模型 | primitive FCC 1 atom，a=4.000–4.080 Å | primitive 4³，64→63 | conventional 3³，108／107／106 |
| XC／偽勢 | 準備程式預設 PBE／PAW | PPT 本身未給；需原始該系列輸入核實 | LDA/PZ、NC `Al.pz-vbc.UPF` |
| 波函數 cutoff | 300–1000 eV；high-k 程式預設 1000 eV | 被引用的結果使用 600 eV | 60 Ry，約 816.34 eV |
| ecutrho | 準備程式為 8×ecutwfc | PPT 未給 | 240 Ry，為 4×ecutwfc |
| k 點 | cutoff scan 16³；high-k 20³–80³ | 600 eV 的 1³–6³ | 3³，無位移 |
| smearing | 準備程式為 MP 0.02 Ry | PPT 未給 | MV 0.02 Ry |
| 晶格／目標 | EOS 擬合 a₀、B₀ | 單空缺形成能 | 固定 a_ref=3.954580406 Å，原子鬆弛 |

本輪 60/240 Ry、3³、LDA-NC 的來源是前一輪 `campaigns/qe_divacancy_20260923/manifest.json` 的固定幾何 SCF 協定。它不是任意取自簡報，但上一輪準備時未完整對照這份收斂資料。不能把歷史測試說成從未做過，也不能用不同偽勢／XC／晶胞的測試直接認證本輪的每一項條件。

`prepare_qe_bulk_b_convergence.py` 是可追溯的**準備協定**，不是已找回的實際執行輸入。尤其 high-k 的預設 1000 eV 尚不能寫成已核實的 40³／80³ 實際 cutoff。

## 原始資料追溯

- PPT：`C:\Users\m1331\Downloads\qe_bulk_vacancy.pptx`，SHA-256 `e95c93a0c6404ddb166a791566f7393e65dc8b4c28ffdd11cc8e1dbba96ec73f`。內部 modified 日期為 2026-05-19；下載檔案 mtime 不是新計算日期。
- Bulk EOS 來源鏈：`REPRODUCIBILITY_INDEX.md`、`scripts/prepare_qe_bulk_b_convergence.py`、`scripts/plot_qe_bulk_eos_convergence_figures.py`、`FINAL_PPT_CHINESE_SPEAKER_NOTES_20260527.md`。投影片 40³／80³ 數值與 speaker notes 一致。
- 歷史 EOS raw 路徑記為 `/gpfs-work/dawson666/qe_cases/qe_runs/qe_bulk_b_convergence_20260506`。本次在 Taiwania3 查該路徑及 `/work` 對應位置皆不存在；本機也未找回該 raw EOS／CSV。這表示原始位置待找回，並非沒有做過測試。
- 簡報 vacancy 的整理結果：`outputs/true_vacancy_formation_20260518/vacancy_formation_combined_summary.csv`。600 eV、5³ 的精確值 0.5725285891686487 eV 與投影片相符。
- USB handoff 的 May22 conventional 2×2×4、64→63 是另一批，cell=8.079696×8.079696×16.159392 Å、600 eV／5³ 形成能約 0.601085 eV。其實際 input 為 PBE-PAW、44.09918661／352.79349289 Ry、MP 0.02 Ry，不能冒充簡報 primitive 4³ 的 raw input。
- 上述 May22 PBE PP 與 June30 active QE reference 的 PP hash 為 `450c947cbbf30c33a5a77685bbe9a270947816f8a93837c02c749d0ee4e3ab4a`；本輪 LDA-NC PP 為 `4eab06b63f87f07ede2d5a193e6d993a09107167fd6b8647afa807342501d6e5`。

## 後續設定規則

保留並引用既有測試結果。未找到的欄位標為待核實，不自行補成常見預設；任何新計算都應記錄參數來源及與該來源的差異。k 點需按 primitive／supercell 的倒空間採樣判斷，不能直接搬用相同整數。PBE 與 LDA、PAW 與 NC 的切換屬方法改動，不能視作單純換 cutoff。

本次仍保持已提交的來源快照與科學輸入；這份查核不代表已完成 LDA-NC 新超晶胞的 cutoff／k 點轉移驗證。單空缺最佳化的修復另立可追溯 attempt，不能以放寬力門檻或採用失敗終態來補出結合能。
