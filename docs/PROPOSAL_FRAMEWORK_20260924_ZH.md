# 晶態鋁體相與奈米柱的空缺形成及聚集：proposal 架構 v0.2

日期：2026-09-24。v0.2依導師明確原話修正；取代v0.1以bulk雙空缺為主的範圍。bulk與晶態Al nanocolumns均為必要主體，具體計算矩陣仍是實施建議，並非已完成結果。

導師原話（由使用者提供，保留原文）：

> you need a full discussion on the energetics and structured details of vacancy formation and vacancy condensation in bulk Al and nanocolumns (crystaline phase)of Al for proposal

工作解讀：proposal需完整討論**晶態鋁體相與奈米柱中，空缺形成及空缺聚集的能量與結構細節**。原話中的「structured details」按上下文解讀為structural details；「crystaline phase」為晶態要求。

建議題目：**晶態鋁體相與奈米柱中空缺形成及聚集的能量與結構分析：OF-DFT研究與QE驗證**。

建議英文題目：**Energetics and structural details of vacancy formation and condensation in crystalline bulk aluminum and aluminum nanocolumns: an OF-DFT study with QE benchmarks**.

## 1. 本次 proposal 要回答什麼

研究主體是「兩種幾何環境×兩個物理問題×能量及結構兩類證據」。TFvW選型、QE的Kohn–Sham DFT（KS-DFT）對照，以及線性彈性是支撐研究主體的方法。先以bulk建立基準，再討論有限橫截面、自由側表面與晶向如何改變空缺形成和聚集。

| 必須完整討論的部分 | Energetics：能量問題 | Structural details：結構問題 |
|---|---|---|
| Bulk Al：vacancy formation | 單／多空缺形成能、電子動能貢獻、體積與應變依賴 | FCC格點、缺陷位置、原子鬆弛、位移／應變、晶格或體積變化 |
| Bulk Al：vacancy condensation | 結合能、構型相對穩定性、代表性遷移能障；較大團簇的文獻脈絡 | 空缺間距／方向、單空缺到雙空缺的結構變化、聚集與局部重排 |
| 晶態Al nanocolumns：vacancy formation | 相同化學勢定義下的尺寸／位置依賴，core與subsurface／surface差異 | 軸向晶向、截面及表面、軸向週期、缺陷深度、局部配位、表面鬆弛與晶態保持 |
| 晶態Al nanocolumns：vacancy condensation | 不等價格點的結合能、近／遠雙空缺比較、柱內聚集與往表面移動的能量競爭 | 雙空缺在柱內及近表面的構型、局部重排、表面吸收／缺陷辨識、位移與應變 |

核心問題：**bulk與晶態Al奈米柱中的空缺形成及聚集，在能量與原子結構上有何差異？TFvW能否描述這些差異，並獲得QE結果支持？**

Full discussion要求四個部分都有背景、物理問題、文獻、現有證據、計畫及限制；不應只列四個標題，也不能把奈米柱放回可有可無的延伸。這句話沒有要求下週二前完成所有新計算；確認稿需完整呈現工作範圍與已有／待補證據。雙空缺是聚集的最小計算模型，不等於vacancy condensation的全部物理範圍；較大團簇的形成、塌縮及位錯環等需在文獻背景交代，新增大團簇計算量另行界定。

依使用者會議轉述，已知要納入：TFvW 選型、約 12 Å bulk 的雙空缺、計算細節、QE 比較、線性彈性、E／ΔTkin／a₀／Δa₀、vacancy strain field、vacancy density。使用者後續確認：**12 Å 是超晶胞邊長，導師的目的是減少週期影像干擾，並留出空間讓空缺移動及觀察。** E／ΔTkin／a₀／Δa₀ 可能同時指缺陷變化與方法比較，仍未完全確定，因此下文將兩類差量分開定義。

尺寸目的已確認，但12 Å是否足以達到所需精度仍是待測問題。移動部分先規劃一條原子跳躍的最小示範；是否必須有QE能障對照及需多大路徑收斂程度，留在確認稿中明列。

本次確認稿的完成標準是把研究問題、計算矩陣、量的定義、參考文獻與現有證據列清楚。**不需要在寄出前跑完全部計算，也不以宣稱首次發現鋁空缺聚集作為 proposal 成立的前提。**

## 2. 文獻重疊與研究定位

使用者提供的論文是 Bhowmik et al. 的 arXiv:2505.13245v4（2025-10-17），正式出版為 *Physical Review B* **112**, 174105（2025），[DOI: 10.1103/97k3-sfyw](https://doi.org/10.1103/97k3-sfyw)。已讀取全部九頁，核對方法、Table I 及相關圖表。

該文不只研究三、四、七空缺，也計算單空缺、第一近鄰與第二近鄰雙空缺；Table I 明列 1NN 的 (a/2,a/2,0) 與 2NN 的 (a,0,0)。主方法是 SPARC KS-DFT、PBE、ONCV 偽勢；mono 至 quad 使用 864 格點，hepta 使用 2048 格點（pp. 3–4）。無外應變的雙空缺結合能為 1NN −64 meV、2NN +9 meV，正值代表聚集有利；作者估計結合能數值精度約 5 meV。Fig. 3–4 亦包含所選團簇的位移／電子密度表徵，故「研究雙空缺」或「加畫位移」本身均不能當作新穎性宣稱。

依新指令，定位應是**bulk與晶態奈米柱中形成／聚集能量及結構的完整討論**，其中包含TFvW參數依據、跨環境可轉用性與QE驗證。線性彈性提供可檢查的結構響應；本文未見完整elastic dipole tensor比較或TFvW電子動能分解，但不據此主張全領域新穎性。加入奈米柱也不自動代表問題未被研究。

該文的 hydrostatic strain 定義為 ΔV/V，線長倍率為 (1+ΔV/V)^(1/3)，不是各軸都施加同一百分比。不可把其 5%、10% 體積應變直接當成本案線性小應變測試。

## 3. 專案已有什麼，以及這次新增什麼

活躍專案：`C:\OFDFT\nano_tensile_TFvW`。沿用目前 Windows 原始碼與既有資料，不使用舊輸出包中的 source_tree 代替。此輪只整理架構與文獻，沒有啟動或重新提交計算。

| 項目 | 核實的既有狀態 | 對新架構的意義 |
|---|---|---|
| TFvW 定義 | Tₛ[n] = λ T_TF[n] + μ T_vW[n]；λ 乘 TF，μ 乘 vW，無 λ+μ=1 限制 | 選型章須保留這個專案定義，避免沿用不同文獻的 λ 意義 |
| 參數候選 | λ=0.9、μ=0.1；舊 10×10 粗掃描有缺點及未通過項 | 稱工作候選，不稱唯一／全域最佳；說明數值合格與物理準確是兩回事 |
| bulk 尺寸 | 3×3×3 conventional FCC：108 格點，雙空缺 106 原子；a_ref=3.954580406 Å，L=11.863741218 Å | 符合已確認的約12Å邊長起點；仍需尺寸檢查，不能稱已消除週期影像 |
| 較大模型 | 4×4×4：256 格點／254 原子，L≈15.8183 Å，已有受控 OF-DFT 比較 | 可延續尺寸測試，不必將全部工作從頭重做 |
| 晶格常數 | 固定參考 a_ref≈3.95458 Å；較新 TFvW EOS 報告 a_eq≈3.94742 Å、B≈50.19 GPa | 兩者不可混稱同一個已收斂 a₀；改基準尺寸須一致處理四態 |
| 九月 OF-DFT | 有匹配 0V／1V／1NN／2NN 固定胞結果及部分網格／尺寸檢查 | 不把八月變晶胞資料的缺口錯套到所有九月結果；仍不能宣稱無限尺寸收斂 |
| QE 四構型 | portable campaign 為固定 TFvW 幾何上的 LDA SCF；包含 1NN、2NN、[310]、中距 [110] | 可比較同原子數構型總能差，但不能提供 QE 自己的鬆弛位移 |
| QE 最新可讀紀錄 | 2026-09-24 09:41 台北時間紀錄：1NN 已完成並接受，另外三項當時 pending | 這是有時間戳的紀錄，不是目前佇列查詢；續算先查現況，避免重複提交 |
| 幾何分析程式 | 已有 minimum-image 非仿射位移、平均鍵長應變、全晶胞 Green–Lagrange strain | 擴充完整局域 εᵢⱼ 與 elastic dipole；不能把平均 bond strain 當完整張量 |
| ΔTkin 舊程式 | 已有單空缺形成動能差 T(107)−(107/108)T(108) | 保留為優先解讀；QE 電子動能提取與新結果欄位仍需建立 |
| 奈米柱幾何 | 既有FCC builder與歷史幾何審閱支持軸向週期、側向真空的圓形／多邊形截面模型 | 可接回主線；不是有限長顆粒。每個實際使用case仍需核對軸、表面、晶態與邊界 |
| 奈米柱舊能量 | 部分資料只含缺陷E/atom，未有匹配pristine；舊runner亦可能使用GGA及不同a₀／未明列λ、μ | 不能稱已完成新協議形成能或QE驗證；先做輸入與reference配對審閱 |

來源入口：`RESEARCH_PROTOCOL.json`、`campaigns/qe_divacancy_20260923/manifest.json`、其 `comparison/TFvW_controlled_energies.csv`、`docs/TAIWANIA3_20260924.md`、`scripts/plot_vaclm_three_maps_gillan_style.py`、`scripts/analyze_divacancy_geometry_strain.py`；另參照 `C:\OFDFT\RESEARCH_UPDATES\FINAL_PROPOSAL_20260917\audit\proposal.md`、`C:\Users\m1331\Downloads\0924meeting\07_Background_optional\Bulk_EOS\eos_fits.csv` 與 `C:\OFDFT\LOCAL_MACHINE_20260907\TAIWANIA3_DEPLOYMENT_20260924.md`。報告中的舊值只是其原協議的結果，不替代新 QE 對照。

## 4. 方法基準、bulk與nanocolumns的工作包

### A. 保留 TFvW 選型，建立 bulk／單空缺基準

整理已有 λ–μ 掃描、缺失點、收斂門檻與候選理由。以 a₀、E(V) 曲率／B、單空缺形成能作主要校準依據；ΔTkin 與 Δa 作額外诊斷。候選至少要能穩定計算、bulk 平衡合理，不能只因某個形成能接近 QE 就稱最佳。

先查核 0.9／0.1 及與選型有關的鄰近點／缺口；μ=0.1 位於原掃描下界，必要時做更小正 μ 的敏感度檢查。先固定候選判準再評估雙空缺；若用雙空缺重新調參，要改稱新增校準，不能繼續將同一資料稱獨立驗證。

QE 與 OF-DFT 各自建立 bulk EOS。為線性彈性取得 cubic C₁₁、C₁₂、C₄₄，並檢查 B=(C₁₁+2C₁₂)/3 與 EOS 相容。完整彈性常數只需對選定候選與 QE 計算，無須對全部 100 組參數展開。

### B. 約 12 Å bulk 的雙空缺與 QE 直接比較

最小四態：0V（108 原子）、1V（107）、2V-1NN [110]（106）、2V-2NN [100]（106）。1NN 初始距離 a/√2，2NN 初始距離 a；[110] 的第二個距離點不是 FCC 第二近鄰。

兩層對照各有用途：

1. **同一晶胞及同一原子幾何的 SCF**：比较方法對該幾何的能量差、力、應力與密度。現有四構型 QE campaign 屬此層。
2. **同一固定晶胞中，各方法各自鬆弛原子**：比較平衡構型、形成／結合能、位移、局域應變及殘餘應力。QE 需要 `relax`，現有 SCF 不能替代。

bulk a₀ 的方法差異另外記錄；以共同晶胞對照時扣除各自 pristine 的殘餘應力。再對代表性模型做零外壓變晶胞計算，取得 ΔV／Δa，與固定胞結果分表呈現。[QE 官方 relaxation 說明](https://www.quantum-espresso.org/Doc/pw_user_guide/node11.html)

[310] 與中距 [110] 保留為會議已關注的診斷構型；既有計算收集完成後可列附表，不將所有方向、距離、應變做全排列。它們的比較同時改變方向與距離，不能單獨歸因為方向效應。

### C. 由原子位移連接線性彈性與 vacancy condensation

對 1V、1NN、2NN 輸出原子位移向量 u、局部 εᵢⱼ、全胞平均應變、殘餘應力、elastic dipole Pᵢⱼ 與 relaxation volume。比較 OF-DFT 與 QE 的空間分布、張量分量及殼層平均，而不只比較漂亮的箭頭圖。

vacancy condensation的首階段計算為**兩個單空缺參考與雙空缺終態的能量及鬆弛響應比較**，例如E_b、P₂−2P₁、Ω₂−2Ω₁。這能描述最小聚合單元的驅動力和彈性變化，不能直接給出真實時間、速率或大團簇成長；proposal的文獻討論仍須涵蓋較完整的聚集物理。

依「留出空間讓空缺移動並觀察」的補充，**最小計畫另納一條既有 2NN→1NN 原子跳躍示範**：保留同一106原子、晶胞與原子對應，顯示原子填入空格點後空缺位置如何改變，並用NEB估計路徑能障。安排QE端點鬆弛；若要宣稱遷移能障也得到QE驗證，必須追加同一路徑的QE NEB，只有端點SCF不足。將是否必須在本proposal核心完成QE路徑列為明確確認項。

這條NEB是候選最小能量路徑，不是直接觀察到的時間演化，也不能證明自發凝聚。空間足夠不代表一般靜態鬆弛會自行跨越能障。有限溫度MD、速率及多路徑網路維持延伸，幾何優化步數不作物理時間。

### D. 晶態Al nanocolumns：形成與聚集均列核心

從既有模型選一個有完整來源、穩定晶態核心的晶向／截面，先比較兩個直徑；具體選哪個須看pristine、1V、QE資料的完整性，不能僅依檔名或builder預設選擇。原有[100]、[110]、[111]家族可作候選，不要求三晶向×所有尺寸×所有位置同時全掃。

**Formation：** 每種柱體建立匹配的pristine，選core與subsurface兩種局部環境，必要時加topmost surface。報告移除格點位置、徑向座標／距表面深度、鄰居配位、形成能、ΔTkin、原子位移及重排。表面原子移除可能對應表面缺陷或形貌改變，須據弛豫後結構命名，不一律當成保留在晶內的空缺。

**Condensation：** 在同一柱體內選近接與較遠的雙空缺、並比較core與近表面代表構型。各位置的單空缺能量可能不同，使用位置配對的結合能，不能直接用2倍中心單空缺能量。結構圖要顯示空格點、原子填補、局部配位及是否仍為晶態；若空缺移向表面或被表面吸收，要與柱內聚集分別標示。

**方法對照：** 代表性pristine／1V／2V均規劃QE同幾何及鬆弛比較。TFvW在bulk校準後延伸至表面附近，是另外一項可轉用性檢查；不能因bulk a₀接近就預設低配位表面也正確。奈米柱部分的能量和結構對照不是附錄選項。

**尺寸與晶態：** 約12Å要求針對bulk cell，不自動成為奈米柱直徑、真空厚度或軸長規格。奈米柱要分別檢查直徑、軸向重複長度、橫向真空及軸向k點；弛豫前後檢查FCC核心／局部配位與表面重構。表面配位降低本來就可能存在，不因表面原子不被分類為FCC就把整根柱說成非晶；如果整個核心重構成另一相，也不能仍當原FCC晶態樣本。

**線性彈性：** bulk的週期彈性偶極／均勻體積公式不能直接當作奈米柱自由表面的完整解。奈米柱先報位移、局部應變、軸向尺寸／力響應；需要continuum比較時使用對應自由側表面及軸向邊界。含真空整盒體積歸一化的stress及density不能直接當成材料內部值。

## 4.1 Proposal的章節安排

1. 研究背景與問題：formation、condensation及bulk／nanocolumn的物理差異。
2. 理論與能量定義：DFT、TFvW、化學勢、形成／結合能、結構表徵與線性彈性。
3. 計算方法及TFvW選型：已有參數掃描、QE對照、幾何與收斂規則。
4. Bulk Al的vacancy formation：文獻、已有結果、能量與結構、待補計算。
5. Bulk Al的vacancy condensation：雙空缺最小模型、較大聚集體文獻、構型／路徑與彈性。
6. 晶態Al nanocolumns的vacancy formation：尺寸、晶向、core／subsurface／surface及QE比較。
7. 晶態Al nanocolumns的vacancy condensation：位置配對、柱內聚集與表面作用、能量及結構。
8. Bulk與nanocolumns的綜合比較、可重現性、限制及研究進度。

第4–7章每章皆要明確回答：物理問題是什麼、文獻已知什麼、我們已有什麼證據、還要算什麼。這樣才能符合「full discussion」，而非只把模型或程式展示一遍。

## 5. 計算矩陣與預期交付

下表是 proposal 計畫；不是承諾 9/29 前全部完成。所有正式主結論均規劃 QE 配對；僅 OF-DFT 的大尺寸探索須明列為未有 QE 同尺寸驗證。

| ID／優先順序 | 模型與工作 | OF-DFT／QE 要比較的量 | 交付與完成判準 |
|---|---|---|---|
| A1 必做 | 既有 TFvW scan 整理與候選查核 | a₀、B、E_f¹、ΔTkin_f¹、收斂狀態 | 參數表／熱圖，列選取與排除理由，不把缺值當零 |
| A2 必做 | pristine bulk EOS，建議 7–9 個平衡附近體積點 | a₀、B、相對 E(V) | QE／OF 各自數值收斂；記錄 XC、偽勢及擬合範圍 |
| A3 必做 | bulk 小應變，零點及正負幅度；建議先 ±0.25%、±0.5% 線性應變 | C₁₁、C₁₂、C₄₄、B、需要時方向性 Young's modulus | 應力斜率／能量曲率相符；剪應變定義一致；幅度為待測起點 |
| B1 必做 | 108-site 四態：共同晶胞，SCF＋各自 fixed-cell relax | E_f¹、E_f²、E_b、E(1NN)−E(2NN)、u、σ、ρ_e | 匹配的四態基準；同幾何與各自鬆弛結果分列 |
| B2 必做 | 108→256 sites 重複關鍵四態，保持局部構型與相近網格／k 點密度 | E_b、構型能差、P、位移／應變對尺寸的敏感性 | 至少雙尺寸；若弱結合符號仍在誤差內，報告未定；必要時 OF 延至500 sites，QE再按資源追加 |
| B3 既有診斷 | [310] 與中距 [110] 已排程 SCF | 同原子數相對能量、必要時力 | 收集原已提交工作，核對當前狀態；不新增全排列 |
| C1 必做 | B1／B2 鬆弛結構及 stress 後處理 | u、εᵢⱼ、Pᵢⱼ、Ω_rel，OF–QE 差異 | 固定配對、PBC、去整體平移；局域應變擬合殘差及核心區標記 |
| C2 必做 | 零外壓代表性 0V／1V／1NN／2NN，或先從最小代表子集驗證 | ΔV、Δa_eff、cell strain、Ω_rel | 與 fixed-cell dipole 推估相互核對；不可用固定胞 Δa=0 宣稱無缺陷應變 |
| C3 必做診斷 | 四態電子動能提取；電子密度差的代表性切面 | ΔTkin_f、ΔTkin_b；ρ_e、Δρ_e | 清楚定義單位、電子數、參考密度及可比性；若未完成不假填 |
| C4 小型驗證 | 選定構型／四態於零點及正負小體積應變，視需要加一種方向應變 | E_b 的應變斜率與 P₂−2P₁ | 驗證線性區域；能量差需隨數值加嚴穩定，不沿用文獻大應變當線性假設 |
| D1 最小移動示範 | 既有2NN→1NN單次跳躍，審閱／收斂NEB影像及端點 | 原子／空缺位置變化、路徑能量與能障；QE端點，QE路徑程度待确认 | 區分局部鬆弛與空缺換格點；單一路徑不宣稱速率；僅OF能障標未經QE驗證 |
| N0 必做 | 審閱舊奈米柱幾何、XC／偽勢／TFvW參數、pristine配對與QE收斂 | 已有資料能否進入同一比較協議 | 只保留可核實的匹配結果；不把缺陷E/atom當形成能；舊狀態不視為即時狀態 |
| N1 必做 | 一種晶向／截面、兩種直徑；pristine＋core／subsurface的1V | 位置／尺寸相關E_f、ΔTkin、位移、配位、軸向變形；代表性QE配對 | 模型維持晶態核心、軸長／真空／k點收斂；surface case視需要補 |
| N2 必做 | 同柱內近／遠2V、core與近表面代表構型，加匹配位置的1V | 位置配對E_b、相對能量、空缺位置／重排、局域應變；QE比較 | 近遠比較不直接當路徑；判別聚集、移表面或表面形貌變化 |
| N3 討論必含，路徑計算程度待定 | 柱內聚集與向表面移動的競爭；選一條有穩定端點的奈米柱路徑作候選 | 端點能量、需要時NEB能障與QE驗證 | 未有能障不宣稱速率或必定被表面吸收；不足之處明列待算 |
| X 延伸 | 多路徑NEB網路、有限溫度、大團簇、更多KEDF | 速率／模型廣泛比較 | 待範圍確認與主要工作完成後再決定 |

建議主圖：①λ–μ選型與候選表；②OF／QE bulk E(V)；③bulk四態形成／結合能及尺寸敏感性；④bulk與nanocolumn幾何、空缺位置及相同定義的位移／局域應變；⑤bulk P／Ω_rel與奈米柱軸向／局域結構響應分開呈現；⑥兩種環境的ΔTkin與代表性密度切面；⑦原子跳躍示意；⑧奈米柱core／subsurface形成能及位置配對結合能。a₀與B提供bulk彈性基底，奈米柱尺寸／軸長變化另有定義。

## 6. 量與公式：寄出前要固定的定義

### 6.1 能量與動能

以下6.1–6.3中的均勻體相公式主要用於bulk；奈米柱的化學勢、位置與真空歸一化另見6.5。N是pristine的格點數，q是移除原子數，E_q是同一方法／協議下N−q原子的總能；E₀是N原子pristine總能。

\[
E_f^{(q)}=E_q-\frac{N-q}{N}E_0,\qquad
E_b=2E_f^{(1)}-E_f^{(2)}=2E_1-E_2-E_0.
\]

採 E_b>0 表示相對兩個單空缺結合有利。不同尺寸、固定胞／變胞及不同外壓各自配對。非零外壓若問題是熱力學聚合，須比較適當的形成／結合焓 H=E+pV，不直接混合兩種邊界條件。

\[
\Delta T_{\mathrm{kin},f}^{(q)}=T_q-\frac{N-q}{N}T_0,\qquad
\Delta T_{\mathrm{kin},b}=2T_1-T_2-T_0.
\]

前者接續既有 Gillan-style 單空缺表，後者是 binding-energy 的動能貢獻；它们不是溫度變化，也不是完整結合能。跨方法差異另寫 δ_Q=Q_OF−Q_QE，例如 δ(E_b)、δ(ΔTkin_f)。E_tot、E_f、E_b、Young's modulus E_Y 分開命名，避免全部叫 E。

QE 的 one-electron contribution 包含動能以外的項，不能直接當 Tₛ。計畫檢查 QE 7.5 的電子動能提取／`pp.x plot_num=22` 路徑及積分單位，再與能量分解核對。[QE pp.x 官方文件](https://www.quantum-espresso.org/Doc/INPUT_PP.html)

目前 OF 的局域 `al.lda.recpot` 與 QE 的非局域 `Al.pz-vbc.UPF` 不同。原始總能零點不能直接相減評分；同方法先組成形成／結合／相對能量，再作比較。即使同為 LDA，電子動能與密度差也包含偽勢效應，不能全部解釋為 TFvW 近似誤差。

金屬QE計算另須固定展寬能量約定：記錄採用的total-energy欄位與entropy-like展寬項，檢查degauss及k點收斂，再與0 K OF結果比較；目前MV展寬不是實際有限溫度熱力學模型。不能把未檢查的有限展寬總能自動稱為0 K基態能。[QE開發者對展寬能量的說明](https://lists.quantum-espresso.org/pipermail/users/2009-May/012884.html)

### 6.2 晶格、位移、局部應變

bulk 平衡 conventional lattice constant 記為 a₀；方法誤差 δa₀=a₀_OF−a₀_QE。缺陷造成的零外壓變化另記 Δa_def=a_eff,q−a₀，a_eff=(4V/N_sites)^(1/3)。分母用原始格點數，不能用剩餘原子數，否則會把空缺本身混入晶格縮放。

缺陷晶胞若變成非立方，a_eff 只是體積等效量，另報完整 cell strain。Δa/a₀≈ΔV/(3V₀) 只適用小且等向的變化。固定晶胞下整體 Δa 為零，局部位移／應變仍可非零。

u_i 為充分鬆弛後原子相對其匹配參考格點的位移，處理 PBC 並去掉整體平移；變胞時另拆出均勻仿射變形。不同空缺構型的直接位移差須先建立原子／格點對應，不能逐列相減不同編號的原子。

\[
\varepsilon_{ij}(\mathbf r)=\tfrac12(\partial_j u_i+\partial_i u_j),\quad
\sigma_{ij}=C_{ijkl}\varepsilon_{kl},\quad
\varepsilon_{ij}=S_{ijkl}\sigma_{kl},\quad S=C^{-1}.
\]

離散原子的 εᵢⱼ 以固定鄰居與最小平方局部變形梯度估計，報告鄰居定義與擬合殘差。空缺位置没有原子，不應憑空賦予未定義的「該原子的應變」。缺陷核心變形大／非仿射時，不強迫線性彈性成立。既有平均鍵長應變可作輔助，不等於上述張量。

口述「epsilon=E*S」較可能與 Hooke 定律有關；一般式為 ε=S:σ。若 E 是 Young's modulus，單軸應力下為 σ=E_Y ε。這是可能的解讀，不是聲稱導師實際說了哪個公式。

### 6.3 Elastic dipole 與鬆弛體積

以拉伸為正的應力慣例，在相同固定晶胞、各態原子充分鬆弛後：

\[
P_{ij}^{(q)}=-V[\sigma_{ij}^{(q)}-\sigma_{ij}^{(0)}],\qquad
\Omega_{ij}^{\rm rel}=S_{ijkl}P_{kl},\qquad
\Omega_{\rm rel}=\operatorname{tr}\Omega^{\rm rel}.
\]

先用 bulk 小應變的能量導數核對 QE／DFTpy 應力符號與單位。P 的單位為能量，Ω 為體積；GPa、eV/Å³ 的轉換要明列。此線性、稀缺陷模型及由 stress 提取的方法見 [Varvenne & Clouet 2017](https://doi.org/10.1103/PhysRevB.96.224103)，理論綜述見 [Clouet et al. 2018](https://doi.org/10.1016/j.commatsci.2018.01.053)。

以各方法自己的零外應力bulk為基準，S:P是線性、小變形下的鬆弛體積張量推估；零外應力cell relaxation可獨立量得Ω_rel=V_q−V₀。formation volume為Ω_f=V_q−(N−q)V₀/N=Ω_rel+qΩ₀，其中Ω₀=V₀/N，兩者不能互換。小尺寸下cell relaxation與dipole推估的差異本身需測試。

共同參考晶胞若有物理預應力，得到的是該狀態的P；不得直接搭配零應力S，宣稱精確預測零外壓V_q−V₀。須明確處理預應變與對應彈性響應，或另在各方法零外應力附近計算供核對。

為分析聚合，定義 ΔP_cond=P₂−2P₁、ΔΩ_cond=Ω₂−2Ω₁。在四態共用同一晶胞應變、遵守上述應力定義時：

\[
E_b(\varepsilon)\simeq E_b(0)+(P_2-2P_1):\varepsilon.
\]

此式適合當小應變驗證。若對 formation energy 使用 E_q−(N−q)E₀/N，且 pristine 本身有非零物理應力，其應變導數還含 bulk reservoir 項；不能無條件把 −∂E_f/∂ε 等同已扣 pristine stress 的 P。優先在各方法 bulk 零應力附近建立彈性常數與定義，再單獨處理共同參考晶胞的應力差。

更完整的遠場可由 anisotropic elastic Green function 與 P 預測，並與原子位移比較。12 Å cell 未必有足夠「核心外又遠離週期影像」的區域，先以 residual stress 提取 P，遠場擬合列進階檢查。週期 DFT 須用含週期影像的彈性場；有限尺寸修正只處理長程彈性，不能取代網格、k 點與尺寸檢查。[Varvenne et al. 2013](https://doi.org/10.1103/PhysRevB.88.134102)

### 6.4 Vacancy density 與電子密度

| 名稱 | 定義 | 本案中的解釋 |
|---|---|---|
| 空缺比例 c_v | N_v/N_sites | 108 sites 的雙空缺為 2/108≈1.852%；256 sites 為0.781% |
| 空缺數密度 n_v | N_v/V | 單位 Å⁻³；同 N_v、同晶胞中改變間距不改變此量 |
| 團簇數密度 n_d | N_cluster/V | 全為雙空缺時 n_v=2n_d |
| 電子密度 ρ_e(r) | 價電子的空間分布 | 與 vacancy density 不同；3價電子Al之四態電子數為324、321、318、318 |
| 電子密度差 Δρ_e | 必須明列兩個參考狀態 | ρ_q−ρ₀ 積分為−3q；ρ_q−(N−q)ρ₀/N積分為0，但代表另一種定義 |

為研究聚合導致的平均變形，可在稀濃度、線性、已知取向分布與零外應力條件下，用ε̄ᵢⱼ≈n_d〈Ωᵢⱼ_rel〉；其跡為tr ε̄≈n_d〈Ω_rel〉，等向時Δa/a≈n_dΩ_rel/3。這是缺陷彈性模型的推論，不是已確認的導師要求，也不直接外推到高濃度。

固定每胞兩空缺而改變 N 時，同時改變濃度與週期影像距離；先稱尺寸／濃度敏感性測試，不能聲稱已把兩種效應分離。最低計畫不需新增一整套多空缺濃度掃描。

### 6.5 奈米柱專用的能量參考、位置及尺寸定義

對N原子的完整奈米柱和移除q個原子的相同柱體，先明列原子移往哪個reservoir：

\[
E_{f,\mathrm{col}}^{(q)}=E_{\mathrm{col},N-q}-E_{\mathrm{col},N}+q\mu_{\mathrm{Al}}.
\]

為比較bulk與nanocolumns，建議使用各方法一致計算的bulk Al每原子化學勢μ_Al，並說明外應變／壓力與reservoir條件。既有same-column scaled量E_def−(N−q)E_col/N相當於選μ_Al=E_col/N，包含奈米柱表面能等有限尺寸貢獻；可保留並明確標名，但不能與μ_bulk的結果混成同一條曲線。**兩者是不同參考定義，不是單憑數值正負就能判誰算錯。** 本輪只修正討論定義，不改寫或重算歷史數據。

奈米柱位置s、t通常不等價，配對binding定義為：

\[
E_b(s,t)=E_f^{1V}(s)+E_f^{1V}(t)-E_f^{2V}(s,t)
=E_{1V(s)}+E_{1V(t)}-E_{2V(s,t)}-E_{0V}.
\]

在相容幾何／邊界和同一reservoir下μ相消。不能任意用2E_f(core)替代兩個不同位置的1V參考。該量仍是有限週期模型的相互作用，需軸向尺寸檢查。較遠與較近2V的E_far−E_near則為相同原子數的構型差，不必然等於上述binding，也不自動是一條遷移反應路徑。

若選bulk reservoir，電子動能分解亦需一致：ΔTkin_f=T_col,N−q−T_col,N+q t_bulk，t_bulk為同方法／協議的bulk每原子電子動能；這是能量分解中的參考項，不是獨立的熱力學化學勢。binding動能貢獻為T_1V(s)+T_1V(t)−T_2V(s,t)−T_0V。舊scaled-column動能差不可當作相同定義。

位置偏好可定義ΔE_seg(s)=E_f(s)−E_f(core)，負值表示該條件下位置較有利；沒有路徑能障時不推論移動速率或必然到達表面。某些subsurface形成能可高於bulk，不能先假設由core向surface單調下降。[Gupta et al. 2016](https://doi.org/10.1103/PhysRevB.93.085432)

幾何與數值約定：

- 既有builder為三方向PBC，沿柱軸連續、橫向以真空分隔，模擬軸向無限週期柱；L_z是重複長度，沒有端面。有限長有端面的pillar是另一種邊界，不能混用名稱而不說明。
- q/N_sites先記作移除原子的名目比例；只有q個晶內空缺仍可辨識時才能同時稱其為晶內vacancy fraction。被表面吸收後另記N_v,interior及表面形貌，不能把q一直當實際晶內空缺數。數密度須用明確定義的材料體積（例如選定幾何截面A_eff×L_z），不可除以含真空的整盒體積；A_eff取法固定。
- 整盒平均stress會隨橫向真空而變。需要軸向力或有效截面平均軸向stress時，以能量對L_z的導數或明列A_eff轉換；先核對符號和單位。這個截面平均量包含表面影響，不能當局域bulk stress或直接用原整盒GPa作比較。
- 奈米柱a₀優先指建立模型使用的bulk參考；結構結果另報ΔL_z/L_z、局部鍵長、徑向／表面位移及必要的有效直徑。不能把含真空cell邊長叫成奈米柱的晶格常數。
- 軸向力學條件固定：同一L_z下各態只鬆弛原子，與各態放鬆L_z到零軸向力的結果分列；後者的ΔL_z以同一直徑／截面pristine柱的零力L_z為基準。表面stress本來就可能使完整柱軸長偏離由bulk a₀建立的初始值。橫向真空不跟著材料一起作不受限vc-relax。
- 主題限定晶態Al，弛豫後對晶態核心與表面重構分別描述；不把非晶Al、氧化鋁或AlN算作同一材料體系。

可核實的舊資料入口：`app/ase_nanocrystal.py`（FCC、晶向、截面及PBC）、`outputs/nanocolumn_nanocrystal_vacancy_audit_20260518.md`（歷史95例幾何及1V資料）、`REPRODUCIBILITY_INDEX.md`第355行附近（faceted表缺pristine配對的限制）、`scripts/prepare_vacancy_periodic_wire.py`（inner/middle/outer選點、雙參考能量及濃度欄位）。其中旧runner預設GGA勢與不同a₀，未明確接通本輪λ／μ，不能直接拿來稱LDA 0.9／0.1新計算。

## 7. 計算細節與結果可信度

- 每個 case 保存方法、版本、XC、偽勢與雜湊、價電子數、晶胞、移除格點索引／方向／MIC 距離、原子數、電子數、邊界條件、數值門檻、輸入／輸出及狀態。
- OF：報告實際 FFT 點數與間距、密度最小化、力／stress 收斂。QE：報告 ecutwfc／ecutrho、k 點與 offset、展寬／degauss、空帶、SCF 與 ionic 門檻。現有60/240 Ry、3³k點、0.02 Ry、1e−9 Ry是 campaign 起始條件，不等於已完成全部收斂證明。
- 以 E_b、構型能差、P 和位移的穩定程度驗證；不只看總能／原子收斂。對數meV弱結合，可把能量差變化約1 meV作為理想工作目標，實際誤差由加嚴測試決定；若成本不足以判斷符號，就報未定。
- 密度圖固定切面、座標、色階及參考；方法間插值須檢查積分電子數。收斂、缺失、執行中與失敗分開，不把沒有結果當零。
- 目前局域與非局域偽勢不同，所以這首先是完整方法的比較；若要把誤差只歸因到 KEDF，需另設可比 Hamiltonian 的控制實驗。
- 奈米柱另記晶向的實際Cartesian對應、表面／截面、軸向PBC與真空、core／subsurface格點、弛豫前後晶態及原子移除後是否形成不同形貌。只有缺陷E/atom的舊表不能代替formation／binding配對。

## 8. 優先閱讀清單

| 優先 | 文獻與核實連結 | 本計畫要讀出的內容 |
|---|---|---|
| 1 | Bhowmik et al. (2025), *Ab initio study of strain-driven vacancy clustering in aluminum*, PRB 112, 174105. [DOI](https://doi.org/10.1103/97k3-sfyw)；[預印本](https://arxiv.org/abs/2505.13245) | 已有範圍、1NN/2NN基準、能量／焓定義、體積應變與數值細節；不照搬其巨大計算矩陣 |
| 2 | Clouet, Varvenne & Jourdan (2018), *Elastic modeling of point-defects and their interaction*, Computational Materials Science 147, 49–63. [DOI](https://doi.org/10.1016/j.commatsci.2018.01.053)；[開放稿](https://arxiv.org/abs/1802.04062) | 線性彈性、P、Green function、缺陷與外應變的關係；屬理論綜述，非Al數值基準 |
| 3 | Varvenne & Clouet (2017), *Elastic dipoles of point defects from atomistic simulations*, PRB 96, 224103. [DOI](https://doi.org/10.1103/PhysRevB.96.224103) | residual stress、Kanzaki force、位移擬合三種提取法；本案優先stress法 |
| 4 | Varvenne, Bruneval, Marinica & Clouet (2013), *Point defect modeling in materials: Coupling ab initio and elasticity approaches*, PRB 88, 134102. [DOI](https://doi.org/10.1103/PhysRevB.88.134102)；[開放稿](https://arxiv.org/abs/1310.5799) | 固定胞／零stress邊界與週期影像修正；不是直接套用Al數值 |
| 5 | Qiu et al. (2017), *Energetics of intrinsic point defects in aluminium via orbital-free density functional theory*, Philosophical Magazine 97, 2164–2181. [DOI](https://doi.org/10.1080/14786435.2017.1328139)；[開放稿](https://arxiv.org/abs/1611.07631) | OF／KS、KEDF、bulk與點缺陷、晶胞尺寸；避免只看a₀選參 |
| 6 | Radhakrishnan & Gavini (2016), *Orbital-free density functional theory study of the energetics of vacancy clustering and prismatic dislocation loop nucleation in aluminium*, Philosophical Magazine 96, 2468–2487. [DOI](https://doi.org/10.1080/14786435.2016.1205232) | 既有OF空缺聚集與大尺度效應；2015預印本與2016正式出版勿當兩篇 |
| 7 | Puchala, Falk & Garikipati (2008), *Elastic effects on relaxation volume tensor calculations*, PRB 77, 174116. [DOI](https://doi.org/10.1103/PhysRevB.77.174116)；[開放稿](https://arxiv.org/abs/0802.1300) | relaxation／formation volume、stress與cell strain、尺寸影響 |
| 8 | Shao et al. (2021), *DFTpy: An efficient and object-oriented platform for orbital-free DFT simulations*, WIREs Computational Molecular Science 11, e1482. [DOI](https://doi.org/10.1002/wcms.1482) | OF計算平台及方法描述；版本和本機λ/μ實作另查專案 |

補充閱讀，分別對應尺寸、早期雙空缺、應變與移動：

- Radhakrishnan & Gavini (2010), *Effect of cell size on the energetics of vacancies in aluminum studied via orbital-free density functional theory*, PRB 82, 094117，[DOI](https://doi.org/10.1103/PhysRevB.82.094117)。說明尺寸甚至可能影響結合符號；其特定模型與邊界不可直接作本案硬門檻。
- Uesugi, Kohyama & Higashi (2003), *Ab initio study on divacancy binding energies in aluminum and magnesium*, PRB 68, 184103，[DOI](https://doi.org/10.1103/PhysRevB.68.184103)。1NN／2NN與k點、尺寸的重要歷史基準。
- Gavini (2008), *Role of Macroscopic Deformations in Energetics of Vacancies in Aluminum*, PRL 101, 205503，[DOI](https://doi.org/10.1103/PhysRevLett.101.205503)。體積及剪切變形對空缺能量已有直接前例。
- Ho et al. (2007), *Energetics and kinetics of vacancy diffusion and aggregation in shocked aluminium via orbital-free density functional theory*, PCCP 9, 4951–4966，[DOI](https://doi.org/10.1039/B705455F)。遷移／應變下動力學的先例，支援移動示範的背景。

### 8.1 晶態奈米柱與Al表面：本輪新增的閱讀範圍

| 文獻 | 對本proposal的用途 | 不可越界的解讀 |
|---|---|---|
| Gupta, van Huis, Dijkstra & Sluiter (2016), *Depth dependence of vacancy formation energy at (100), (110), and (111) Al surfaces: A first-principles study*, PRB93,085432. [DOI](https://doi.org/10.1103/PhysRevB.93.085432) | core／subsurface／surface必須分辨；空缺形成與原子層鬆弛、表面取向相關 | slab DFT，不是奈米柱凝聚的直接結果；不以終態能量推斷速率 |
| Li et al. (2013), *Surface energy and surface self-diffusion of Al calculated by embedded atom method*, Physica B422,51–55. [DOI](https://doi.org/10.1016/j.physb.2013.03.043) | 表面空缺形成及遷移的晶面依賴 | EAM，不是DFT，不能替代QE基準 |
| Stumpf & Scheffler (1996), *Ab initio calculations of energies and self-diffusion on flat and stepped surfaces of Al and their implications on crystal growth*, PRB53,4958. [DOI](https://doi.org/10.1103/PhysRevB.53.4958) | 表面／台階、缺陷形成與局部環境的DFT背景 | adatom diffusion與vacancy migration須分開，不能挪用能障 |
| Ho & Carter (2009), *Mechanical response of aluminum nanowires via orbital-free density functional theory*, J. Comput. Theor. Nanosci.6,1236–1246. [DOI](https://doi.org/10.1166/jctn.2009.1172) | 晶態Al wire的OF-DFT建模、尺寸／晶向與結構響應背景 | 不是vacancy formation／condensation論文，只支援模型與方法背景 |

本輪搜尋尚未確認完全對應「純晶態Al奈米柱中雙空缺凝聚」的原始論文。這是目前文獻蒐集的缺口，不構成首次研究的證明。正式literature review需繼續沿上述文獻引用鏈查找；表面slab、力學nanowire及bulk clustering應清楚分層，不當成同一種證據。

## 9. 9/29 前的交付安排與待確認事項

以今天 2026-09-24（四）計，「下週二」為 **2026-09-29**。依「週二前寄出」安排，目標是 **2026-09-28（一）寄出範圍確認稿**。

| 日期 | 工作與交付 |
|---|---|
| 9/24（四） | 依導師原話完成v0.2：四個必討論主題、bulk／nano現有資料對照、新增表面文獻與信件修訂 |
| 9/25（五） | 對齊會議用詞；優先讀2025論文與缺陷彈性綜述；固定Δ量定義與移動示範範圍 |
| 9/26（六） | 固定bulk及奈米柱的最小計算清單；標已完成／待算／可選，核对奈米柱配對及既有QE工作現況 |
| 9/27（日） | 整理短版英文架構及計算細節表；檢查參考與公式 |
| 9/28（一） | 校閱後寄給Luder確認範圍；未知結果保留pending，不等待全部QE結束 |

bulk與晶態nanocolumns兩個分支，以及formation／condensation和能量／結構，均已由原話確定，不再詢問是否納入。需要確認的是實施細節：

1. 已確認12Å指bulk超晶胞邊長；計畫以現有11.86Å作起點、256格點作尺寸檢查，是否符合預期的精度與工作範圍？
2. E、ΔTkin、a₀、Δa₀分別指哪些參考差量？建議同時分列形成／聚合引起的變化與OF–QE差異。
3. vacancy density指N_v/V／N_v/N_sites，還是vacancy附近電子密度？本架構可涵蓋兩者，但工作量不同。
4. 計畫加入一條2NN→1NN跳躍示範，是否需要在核心階段完成QE NEB能障對照，或先以QE端點＋OF路徑示範呈現可行性？
5. 奈米柱擬沿用軸向週期、側向真空的晶態模型；首輪選一晶向／截面、兩直徑、core與subsurface的1V／2V，是否符合預期的計算規模？若導師要求有限長有端面的柱，需要另行建模並分開報告。
6. 較大vacancy clusters將納入完整文獻討論；本次新計算先以1V／2V為起點，是否需要指定更大的團簇作核心案例？

本草案已可作下一輪討論基底。上述未確認內容保留為問題，不寫成導師原話或已批准要求。
