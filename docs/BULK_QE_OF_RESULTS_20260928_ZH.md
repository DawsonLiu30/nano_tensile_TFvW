# 9 月 28 日 Al vacancy 結果與 proposal 分析

最新實際遠端查核為2026-09-28 00:07台北。2159026全部終結：1V FAILED，1NN／3NN／5NN COMPLETED且科學验收通过。與首輪合併為DFTpy8/8、QE7/8。本輪僅查核、下載與分析，未提交計算、改變科學門檻或重新開啟監控。

## 交付入口

- 永久備份：`C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_20260928/00_README.md`
- 工作區分類包：`C:/Users/m1331/Documents/Codex/2026-09-24/g/outputs/al_proposal_results_20260928`
- 主報告：分類包內 `05_report/Al_vacancy_proposal_analysis_20260928.docx`，14頁、逐頁排版核查通過。
- 便攜檔：工作區 `outputs/Al_vacancy_proposal_data_20260928.zip`，461668095 bytes，SHA256 `49766c50f85c03f9a1ff6ca8006737e0c51fdef6d9344e8f2a38e9cfb83f1159`。

包含697份原始研究檔的兩個tar.gz、能量/體積/位移/局部應變CSV與JSON、六組圖、15個accepted結構、四主體proposal框架、文獻、獨立QA及重建程式。111.25GB遠端波函數與mixing scratch未下載、未刪，不宣稱完整restart備份。

## 主要科學結論

DFTpy單空缺形成能0.665843 eV，1NN結合能55.481 meV；在此OF模型中1NN最低。QE六個2V的形成能均可報告，同106原子相對排序中5NN最低，1NN比它高59.008 meV。5NN有雙等距週期影像，不能解讀為孤立雙空缺最佳間距；PBE/PAW與LDA/local也不允許把跨方法差異全部歸因TFvW。

QE1V最後力分量0.000549444791 eV/Å，仍比輸入0.0005高9.89%，且無final relaxed-cell SCF。力下降而非已證实震盪；正式1V形成能與全部QE結合能保持null。亞meV OF結合能不以符號直接宣稱吸引或排斥。

1NN最大非仿射位移為OF0.15924 Å、QE0.07833 Å。全15終態使用原始9/25 parent lattice及各終態cell，不以continuation起點重設位移。局部應變為固定FCC第一近鄰圖的非仿射最小平方法擬合，資料保留rank與殘差；非total strain、elastic dipole或無限介質解。原子位移、全局cell strain、體積/等效a與vacancy density均已區分定義。

## Proposal 完整範圍與下一步

報告同時涵蓋bulk／晶態nanocolumn的formation／condensation，能量及結構兩面皆列出可驗證觀察量。TFvW選型、既有cutoff/grid測試、ΔTkin分解、線性彈性關係與2025文獻定位已納入。

最小優先缺项為合格QE1V。Nanocolumn仍只有方法/最小矩陣，沒有本輪新協定下的完整驗證結果；柱軸PBC、橫向5 Å真空、橫向cell固定，生成前列明5 Å幾何定義。不同位置1V需逐一配對才能求2V binding。Elastic dipole、有限尺寸誤差或遷移路徑依欲支持的結論另補，不將靜態最佳化當動力學。所有後續提案均未於此次執行。
