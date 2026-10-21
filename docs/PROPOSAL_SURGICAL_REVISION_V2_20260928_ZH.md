# Proposal 18 頁局部修訂 v2

使用者最新要求保留18頁架構、全部canonical與六圖，只修端點／輸出對應、週期影像診斷、篩查範圍、nano主張與文獻缺口；所有新增DFT均待確認。這項限制優先於更早「修復直到通過」的提交授權。現有bulk已16/16 accepted，不重送。

永久歸檔：`C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_REVISION_V2_20260928`。主報告為 `04_report/Al_vacancy_proposal_revised_v2_20260928.docx`；18頁、六圖原bytes、五則原生Word修訂註解。修改對照與測試紀錄為 `05_revision_comparison_v2.md`、`07_QA/analysis_test_record_v2.md`。64檔已逐檔核對歸檔副本。

- 第12頁12個起步端點只對應C、P(0)、Ωrel預測與位移比較；−P:ε能量驗證需額外兩個受應變1V端點，只涵蓋一個投影。新設定若需pristine重新平衡另列成本。
- 第11頁四態篩查只涵蓋固定幾何Ef1、Eb1NN、Eb2NN、D12；不認證3–6NN或5NN最低。可選5NN兩單點仍待確認。
- 第6頁補六構型與初始／兩方法終態影像距離，完整18列CSV／JSON。5NN以1e-6Å等距容差的重數為初始2、OF1、QE2；OF僅分裂2.525e-5Å，1e-3Å近等距判準三者仍2。這是刪除parent格點的仿射映射，非vacancy migration。
- Nano選擇縮準至位置形成與代表1NN配對；不承諾D12转用。待確認[001]/{100}方柱147格點定義與理想配位12/12/8已做算術核查，未生成生產輸入或提交DFT。真空pilot包含面中央表層1V；OF5/8Å初始網格112×112×78／132×132×78，需保留間距。單點不認證鬆弛結構或配對Eb。
- Qiu只採用bulk晶格與KEDF／尺寸背景，不作TFvW或雙空缺D12的直接認證；補Gupta TableI與本機歷史EOS，沒有扩張間隙／遷移研究。
- 原8tests重跑＋新增6tests通過。16態來源核查及987項算術核查保留舊紀錄，未冒稱本次重跑。原分析與六圖SHA一致。18頁逐頁視檢完成。

報告 SHA256：`80ee05dcd657c438daa6d10a3045e8e55878e93c5d941d780e4c06d986a0d2c2`。
Manifest SHA256：`a9d8509c0a51f0341037e41dd9147ecca18cd26db75e003f6d8d11e6d13cd5be`。
ZIP SHA256：`4d0dbb7e06b1dd4f4fcb7fcd94490cafdabdbc8a39e20dc0c1fd57a98a9b0e1e`。

本次無遠端呼叫、無新DFT、無監控。後續依使用者逐項確認再處理條件式增量，不把proposal設計視為提交授權。
