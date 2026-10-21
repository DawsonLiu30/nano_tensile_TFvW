# Proposal 比較論證修訂版

2026-09-28，依使用者貼上的完整評語完成第一階段。此更新優先於舊版proposal敘述，原計算數據與歸檔不變。

- 最新報告：`C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_REVISION_20260928/04_report/Al_vacancy_proposal_revised_20260928.docx`。18頁、六張實際嵌入圖，逐頁Word／Poppler渲染視檢完成。
- 以兩方法共16個accepted終態為唯一資料基準。主文改為材料問題、候選選型、單空缺比較、雙空缺與文獻、競爭假說、最小驗證與nano；Slurm、hash及驗收過程移附錄。
- 新D12：OF−38.643188meV，QE+25.007536meV；不依賴單空缺參考。1NN的OF−QE Eb差57.808705meV；QE近零符號仍無總物理誤差界。
- 新1NN D_V：OF−0.622215Å³、QE+2.047650Å³；屬結構相關證據，不是成因证明。
- OF 1NN全三維最大位移parent46／60不在原s_z=1/2切面；原切面max0.029185Å。新兩切面同色階、全體CDF、殼層與max atom向量表解決圖文比較範圍不同的問題。
- 六篇原始來源、19筆轉錄值。Bhowmik v4 p5 Results的Eb−64/+9meV和2NN Hb+5meV分開；Radhakrishnan–Gavini作者2016修訂稿p10 WGC +67/+45meV；Ho與Connétable–Maugis全文未取得，只引用核實摘要。Gupta表II雙空缺條件不足以直接標1NN，因此保留ledger而未強行放進1NN圖。
- QE XML獨立對照final geometry、energy、force、stress；DFTpy由ASE traj對照JSON/npy。16態全通過。八份實際PWSCF/XML均7.5，沒有可核實binary build hash；八份DFTpy config／resolved_config權重一致、無跨attempt restart。8項測試通過，舊16列原數值完全保留。
- 未新增DFT、未重啟監控、未改生產參數。Phase2現幾何取樣／smearing篩查與Phase3有選擇的尺寸／機械约束／彈性／nano工作僅為可審查計畫，不是提交回執。不得以此更新推定已執行或自動開全部矩陣。
- 修訂包47檔已複製研究歸檔並逐檔hash核對。DOCX SHA256 `b958c81d8c32c0eb57006d128d3920fc27dd00b8c08330442fc7b27e708cb1f8`。ZIP在本對話outputs，SHA256 `b3a89c557b286a873f4bf2daf0bfcbd67f9965431fbef65bea531e31e15ed1de`。

完整修訂紀錄、文獻／數值表、科學圖、可執行分析／繪圖／DOCX腳本及tests均在同包。原746個raw files與完整結果包仍按原路徑保存，不重新複製大型scratch。
