# Proposal 執行稿 v0.3：2026-09-25 晚間

**9/28 09:59 最新更新（優先於下方歷史狀態）：** QE1V修復2162428已於03:13:58完成，frozen inspect與獨立raw QA均通過，現在DFTpy8/8、QE8/8。QE1VEf=0.644963eV，六組binding已補齊；1NN約−2.328meV的物理符號仍需數值／尺寸誤差界。新49原始檔已核hash下載、三批合計746檔；完整bulk分類包為本對話 `outputs/al_proposal_results_complete_20260928`。最新結論見 [完整bulk結果索引](BULK_QE_OF_RESULTS_COMPLETE_20260928_ZH.md)。本次未新增計算或啟用持續監控；nano、有限尺寸／數值誤差及所需遷移路徑仍待補。

**9/28 00:44–00:52 修復更新：** 依使用者「分析解決並重送直到通過」指示，QE1V修復job **2162428**已在ct56執行；新trust_radius_min=1e-6、SCF加嚴1e-10，其餘門檻與物理設定保留。新結果尚未驗收；7/8合格數目暫不改。00:52既有SSH共用連線失效，需使用者重新登入才能取得輸出；Slurm批次獨立繼續。回執和完整修復來源在本對話outputs/qe_1v_repair_submission_20260928，永久副本C:/OFDFT/RESEARCH_UPDATES/AL_QE_1V_REPAIR_20260928/submission。不得重複提交2162428；先查最新attempt和queue。

**9/28 00:07 查核／歸檔更新（優先於下文歷史狀態）：** 本輪全部作業已終結。DFTpy8/8、QE7/8可用；續跑1NN/3NN/5NN完成，1V再次BFGS停止，QE1V形成能及全部binding留缺。全部697份原始研究檔已下載核hash，14頁完整分析報告與101檔分類資料包保存於 `C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_20260928`，入口為00_README.md。結果與後續proposal缺項見 [9/28結果索引](BULK_QE_OF_RESULTS_20260928_ZH.md)。此次沒有新提交、放寬門檻或定時監控；下方running/pending與heartbeat敘述均為歷史快照。

**9/26 18:28接續更新：** 使用者已要求繼續QE並拉長時限，正式array **2159026** 的1V、1NN、3NN、5NN四組均RUNNING（48h Slurm／47h QE，四組並行）。新起點均為已核實完整SCF幾何、新BFGS及電子初態；新manifest/runner來源鏈已驗證，科學門檻未放寬。原4組usable保留，正式QE binding仍待1V及各case通過inspect。DFTpy8組無需重跑，近低點切optimizer已完成歷史實作查核，混合FIRE仍待獨立驗證。回執與設計記錄見本對話outputs/QE_continuation_DFTpy_optimizer_20260926_zh.md；永久副本位於 C:/OFDFT/RESEARCH_UPDATES/AL_BULK_QE_CONTINUE_20260926。定時automation保持關閉。

**9/26 17:22最新結果：** 本輪已全數停止，DFTpy8/8可用，QE4/8可用（0V、2NN、4NN、6NN）；1V為BFGS停止，1NN／3NN／5NN為11h上限。534原始檔已下載核hash，六頁報告與分析見 [傍晚結果索引](BULK_QE_OF_RESULTS_20260926_ZH.md)。後續本機預設已改48h／47h，尚未提交接續作業。下文較早running/pending描述均為歷史快照。

**9/26 02:09更新：** 依使用者要求，10分鐘heartbeat已刪除，今晚不再定時查詢。QE array2154594已改為最多8組同時執行，剩餘1V與六種2V全部RUNNING；目前收斂門檻保留，明早再依耗時、文獻與精度評估。下文9/25排隊與監控描述保留為歷史快照。英文scope/calculation plan v0.4和nano準備盤點已另存本對話outputs。

此稿更新當前決策與結果，優先於9/24架構中已過時的起始晶胞、QE方法和「尚未計算」敘述。完整理論公式與既有文獻仍見 `PROPOSAL_FRAMEWORK_20260924_ZH.md`，數值來源见本輪manifest與分析報告。導師要求的四個部分均保留：bulk／晶態Al nanocolumn各自的vacancy formation及condensation，每部分都要有energetics與structural details。

## 可直接寫入 proposal 的方法

- 活躍專案仍為 `C:/OFDFT/nano_tensile_TFvW`，沒有另建研究分支或覆寫舊結果。
- Bulk為3³ conventional FCC，108父格點、三方向PBC。共同初始邊長12.119544 Å；0V、1V與1NN–6NN雙空缺合計8組。兩種方法均独立零壓鬆弛原子與全晶胞。
- QE保留PBE／PAW，800 eV（58.798915481 Ry）、8倍density cutoff、3³ k點、MP 0.02 Ry、SCF 1e−8 Ry。來源是June16實際vc-relax輸入與PPT已測cutoff，不自訂新cutoff。
- DFTpy保留LDA／local recpot，TFvW λ=.9、μ=.1、固定78³密度網格。9/15–17同方法網格與EOS測試支持此選擇，完整a(t)、h(t)與實際spacing隨過程保存。
- 力向量驗收≤.001 eV/Å；DFTpy另驗cell filter廣義力及全stress≤.05 GPa。QE另驗final relaxed-cell SCF及全stress≤.5 kbar。作業成功退出與科學收斂分開判定。
- Nanocolumn沿柱軸無限延伸、橫向5 Å真空。柱軸可鬆弛、橫向盒固定；實際生成前明確列出5 Å是總間隙或單側留白。不能套用bulk全自由cell縮掉真空。
- QE與DFTpy刻意採不同XC／偽勢。比較同方法形成能、結合能、結構與所定義的動能差；不直接相減兩套原始總能，也不能把全部差異歸因於TFvW。

## 當前證據與必須補齊的內容

| Proposal主體 | 已有證據 | 下一個有意義的產出 |
|---|---|---|
| Bulk formation | DFTpy 0V、1V零壓結果已收斂；QE 0V已重新inspect可用、1V執行中 | OF/QE各自的E_f、a₀、ΔV、非仿射位移並排；QE完成前保留pending |
| Bulk condensation | DFTpy六種2V全部收斂；QE六組排隊；原子軌跡未見換到另一父格點 | E_b、ΔTkin、位移/應變/密度圖；代表性路徑另列，不將BFGS軌跡稱作擴散動力學 |
| Nanocolumn formation | 軸向PBC、側向真空及可生成結構的程式已確認；歷史資料仍需按PP/XC/權重/0V配對篩選 | 選一晶向/截面、兩直徑、core/subsurface格點；同方法0V/1V與QE比較的最小表 |
| Nanocolumn condensation | 既有結構工具可選雙空缺；本輪尚無新協定下已驗證nano 2V結果 | 近接/較遠2V加各位置匹配1V，使用E1(s)+E1(t)−E2(s,t)−E0；保留表面吸收與晶內聚集的區別 |

截至22:50台北查核，DFTpy job2154593三組均COMPLETED、8/8 accepted；QE job2154594為1/8可用（0V）、1/8執行（1V）、6/8排隊。這是時間戳快照，後續由此對話10分鐘追蹤更新，不把舊狀態當即時狀態。

## 初步數值與可說的結論

DFTpy的單空缺形成能為0.665843 eV。1NN至6NN結合能依序為55.481、16.838、2.796、0.459、6.949、−0.175 meV（正值表示在此定義下配對有利）。1V relaxation volume為−4.943749 Å³，formation volume為+10.434166 Å³，不能混稱。1NN最大去仿射、去整體平移的原子位移約0.15924 Å。

完整DFTpy bulk體積等效a₀為3.947473 Å；QE 0V的a_eq為4.041725 Å（由V=1782.641084 Å³計算）。兩方法約2.33%的線長差只是這套方法與有限採樣下的結果；不能當成KEDF單一誤差，也不能把PPT primitive高k EOS與本108-site vc-relax當同一數值協定。

1NN與2NN的正結合能支持本OF模型偏好近鄰配對，但尚待QE和尺寸/相關數值檢查。5NN [310]有兩個等距週期影像，保留為影像診斷。4NN與6NN接近零，尚不依亞meV符號主張吸引/排斥。此批静態鬆弛沒有建立凝聚速率、遷移能障或自發聚集。

ΔTkin_f與ΔTkin_b採同形成/結合線性組合，報weighted TF/vW項。1NN binding動能分量約−137.6 meV，總binding為+55.5 meV，說明其他能量分量共同決定穩定性；不可單看kinetic項取代總能判斷。

## 交付順序：以9/28確認稿為目標

1. 先完成「研究問題、四主體、方法與參數來源、能量/結構量定義、文獻定位」的可審查稿。既有TFvW選型及PPT收斂證據納入方法章。
2. 把已驗證bulk OF結果作preliminary results，QE已完成與pending逐項標示；繼續計算時同步整理，無須等待最後一組才寫proposal。
3. 完成nano最小計算矩陣與可重用歷史配對清單；以一晶向/截面、兩尺寸和少數代表位置為擬議首輪，避免展開所有晶向×尺寸×位置全掃。
4. 將線性彈性與migration示範定義清楚：零壓終態殘餘stress不能直接代替固定胞elastic dipole；路徑和NEB比較屬獨立計算內容。
5. 9/27完成英文範圍稿、參數/計算表和圖，9/28人工校閱後寄給Luder；待算項保留pending，不為趕稿放寬門檻或假稱已完成。寄信尚未獲明確代寄指令，本次只備稿。

## 操作可靠性

已區分current checkout、immutable snapshots、結果和OLD歷史布局；舊脚本中過時路径與不同QE/conda版本不可直接沿用。本輪實際sbatch及環境來源已記於iService架構報告。

每10分鐘追蹤既有Slurm結果、保留有意義的進度與分析更新；不自動重送或更改科學條件。網頁閒置、互動Bash的TMOUT=900與SSH共用連線為不同層；查詢SSH不保證網頁不登出。Slurm批次計算不依赖提交端網頁持續開著。

官方/work為工作空間，28天未使用可清理且不備份。已將本輪重要completed檔案與hash同步到本機，包含逐次電子事件、final density、結構和QE 0V原始輸出；不修改timestamp規避清理，不把大型scratch全部搬回。
