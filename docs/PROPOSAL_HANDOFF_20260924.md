# 2026-09-24 會後 proposal 工作入口

**最新狀態（9/25晚間）已更新至 [Proposal執行稿v0.3](PROPOSAL_CURRENT_STATUS_20260925_ZH.md) 與 [新版未寄送確認信](LUDER_SCOPE_EMAIL_DRAFT_20260925.md)。** 下文保留9/24會後紀錄；其中「尚未計算」、11.86Å共同起點及舊QE-LDA設定不能代替新協定。

本次對話接續「搬遷 nano_tensile 專案快照」，使用相同活躍原始碼 `C:\OFDFT\nano_tensile_TFvW`。目前沒有另建專案或更換計算協議。

- [研究架構、計算矩陣、公式與文獻 v0.2](PROPOSAL_FRAMEWORK_20260924_ZH.md)
- [Luder 英文範圍確認信草稿](LUDER_SCOPE_EMAIL_DRAFT_20260924.md)

已確認的使用者補充：12 Å是bulk超晶胞邊長，目的是減少週期影像並提供觀察空缺移動的空間。現有108格點胞邊長11.863741 Å符合起點；約12 Å並不代表週期影像已消失，需以256格點等檢查。E、ΔTkin、a₀、Δa₀可能同時涉及缺陷變化與OF–QE比較，兩者分開命名。

最新明確指令（使用者提供導師原話）："you need a full discussion on the energetics and structured details of vacancy formation and vacancy condensation in bulk Al and nanocolumns (crystaline phase)of Al for proposal"。

因此主體是bulk formation、bulk condensation、晶態Al nanocolumn formation、晶態Al nanocolumn condensation四個部分，各含能量與結構。TFvW選型、QE驗證及線性彈性作方法支撐。v0.1的bulk雙空缺主線已被v0.2取代，不能再把奈米柱列可選。

既有奈米柱為FCC起始、軸向週期、橫向真空的模型；舊單空缺／QE資料須查pristine配對、收斂及GGA/LDA／λμ等協議，不直接當新候選模型驗證。bulk12Å不自動決定奈米柱直徑或真空。QE NEB深度、奈米柱採樣／端面邊界、較大clusters新計算量、density含義與差量參考仍待實施確認。不得把草案計畫當成已完成結果或已核准數值設定。

目標於2026-09-28寄出，早於下週二2026-09-29。此輪未寄送、未排定自動寄信、未啟動或重提QE／OF計算。當前遠端工作狀態需要另查；不可將09:41歷史snapshot當成即時狀態。

對外工作副本位於 `C:\Users\m1331\Documents\Codex\2026-09-24\g\outputs`。詳細架構與信件同步保存在本docs目錄；後續修訂應同步兩處或註明新的單一版本入口。
