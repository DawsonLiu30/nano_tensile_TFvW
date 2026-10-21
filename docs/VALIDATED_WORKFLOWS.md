## 2026-10-21 21:33 QE重啟與迭代已實測確認

21:33 台灣時間啟動查核完成：三個v2續算已實際讀到保存密度與波函數，進入第16次SCF；原子初猜警告及讀檔回退均未出現。0V/1V k5均進入第1次SCF。五個計算RUNNING；三個後續k5等待各自k4通過，沒有新完成端點。三個v1正常停止、舊依賴取消。已停止遠端查詢；沒有持續監控。

## 2026-10-21 續算電子初始態修正 v2

以 `al_bulk_reference_resume_k5_20261021_v2` 為目前入口。v1三組因電子初猜可能破壞重啟而正常停止。v2明訂file/file並保留原物理與驗收；27項程式測試通過，實際讀檔與新物理端點分開驗證。詳見 `QE_BULK_REFERENCE_RESUME_20261021.md`。

## 2026-10-21 QE bulk續算與k5診斷

詳見 `QE_BULK_REFERENCE_RESUME_20261021.md` 與 registry。永久 `workflows/al_bulk_reference_resume_k5_20261021_v1` 保留原科學核心與新case-bound adapter；23項程式測試通過，新的物理端點待驗收。不得直接重跑舊fresh-only attempt_001。

<!-- BEGIN R13_ACCEPTED_RESULTS_20261019 -->
## 2026-10-19 R13 續算驗收完成（最新）

- 2255218於10/18 22:21:11台灣時間正常完成（4h01m12s）；本輪10/19單次SSH查核佇列空。46個完整離子步、185次電子原生成功、無HEG重試；Fmax=.000976534、filter=.000976680eV/Å、σzz=.000040852GPa，596/596保存資料檢查通過。維持.001力與.05GPa軸應力標準，R13_0V正式升級為本模型下數值合格端點。
- 已實際成功的收尾為z-only FrechetCellFilter+BFGSLineSearch（試探增量.02）配合電子TN/econv1e−11/ncheck3；不是FIRE。從2254539 step12 coherent bundle續算，fresh Hessian，新optimizer_state與step46結構／密度bundle一致。永久凍結入口workflows/al_nano_R13_axial_continue_20261018_v2，manifest0873dc84e7e796fea4f743ab37e1f2dcc1a34dd86bee2cf9fef70cc7e3970bbf；原source不改，後驗收證據docs/validated_workflow_evidence/R13_axial_continue_2255218_20261019.json。成功範圍限此R13案例，不把所有柱或全域最低能保證化。
- 此續算能量下降.950994eV，終態Lz=13.739937Å；原子有明顯整體重排。R13的0V/1V相對各自ideal RMS約.75887/.76305Å，但兩個終態去仿射配對RMS=.029942、最大=.105318Å，缺格parent_site_id33仍保留。應先討論重排後結構與空缺造成的增量反應，結晶相分類仍待專門確認。
- 五個OF尺寸已有10/10數值合格端點，R13同方法μAl=-62.943857545eV，Ef=.616366216eV。五組依R11/R13/R15/R17/R20為.424814871/.616366216/.598150394/.628738683/.631826061eV，並非隨尺寸嚴格單調；R11亦有廣泛空缺誘發重排。R13_0V新TN較嚴與原1V LBFGS電子精度差異有列來源；網格／真空／取樣及QE一致性仍另驗證。
- 新raw42檔與遠端SHA一致，資料、結構、逐步分析、五配對能量分項／位移、圖與腳本在C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261019。10/17、10/18舊包與舊canonical表原封保留，大密度／PP／父bundle按hash引用。不新增DFT、不定時監控、未上傳NAS或寄信。QE k9仍為固定幾何診斷（力未過），六個新尺寸未放行。
<!-- END R13_ACCEPTED_RESULTS_20261019 -->

<!-- BEGIN R13_AXIAL_CONTINUATION_SUBMITTED_20261018 -->
## 2026-10-18 18:19 R13 有界續算已提交（最新作業）

- 正式作業**2255218**，ct56/6h/24GiB，回執唯一快照PENDING；不另輪詢。腳本請求1 task/1CPU，scontrol記NumCPUs=2、CPUs/Task=2（當時AllocTRES空），勿把請求值稱為實際已配置1CPU。
- 從2254539完整step12一致bundle續跑，SHA d30d37f9c33e7ef7be05a5ac653232f348774304636eb6691163f5b508f5297f；新attempt_005_axial_linesearch_continue。保留z-only BFGSLineSearch/TN1e−11/LDA-local/TF.9-vW.1/140²×96及force/filter .001、|σzz|.05驗收。線搜尋試探增量.005→.02Å；不是整個離子步位移上限。內部21000s/120ionic/360SCF、循環停止，不自動再續算。
- 前一試跑沒有完整Hessian，這次明確fresh inverse Hessian；新checkpoint後保存optimizer_state與其step/bundle hash/geometry對位，沒有未審核的自動舊state載入。只有原生CONVERGENCE可接受，WARNING不放行。本機11項/遠端11項、API smoke、真實checkpoint preflight通過；尚無新accepted physical endpoint。
- v1僅sbatch --test-only因CRLF拒絕，沒有新DFT；v2獨立snapshot只改LF與封裝名稱，科學來源相同。manifest SHA 0873dc84e7e796fea4f743ab37e1f2dcc1a34dd86bee2cf9fef70cc7e3970bbf。永久入口workflows/al_nano_R13_axial_continue_20261018_v2，完整來源／兩版證據／實際回執在C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261018/RECOVERY_R13_AXIAL_CONTINUATION。
- 已核查的2254539仍未過.001；四組OF配對／九端點保留，R13 Ef空，QE無新作業。不定時監控、未上傳NAS或寄信。
<!-- END R13_AXIAL_CONTINUATION_SUBMITTED_20261018 -->

<!-- BEGIN R13_AXIAL_TRIAL_RESULTS_20261018 -->
## 2026-10-18 2254539 短試結果已核查（最新結果）

- R13_0V為76原子完整柱。2254539於15:06:10達內部6600秒預算退出，非OOM／空轉；111個完成電子呼叫原生成功，第112個被預算中止。Slurm峰值約18.70GiB/24GiB。
- 完成12個離子步（13幀），每步能量均下降，合計3.140396meV；柱長13.431287→13.497657Å（+.494142%）。Fmax .003625856→.013176242，filter .013240575，仍未過.001；σzz由−.022252319→+.008534206GPa。不能稱已接近收斂或新增R13形成能。
- 35個原始檔remote SHA一致、158/158資料檢查；獨立重建12步Wolfe條件皆通過，梯度積分與能量差最大.000846meV。支持有界同法續算的準備，尚非accepted endpoint或全面force/stress認證。後期線搜尋昂貴；原試跑未保存完整BFGS狀態，續跑須明列如何重建或重置。
- 最新完整續跑來源為`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261018/R13_AXIAL_TRIAL_RESULTS/R13_0V/checkpoint_bundle.npz`，step12、SHA d30d37f9c33e7ef7be05a5ac653232f348774304636eb6691163f5b508f5297f。後面未完成搜尋的trial點不能取代此bundle；舊POST_STOP step483是前一輪歷史。
- 分析、逐步表、原始log、唯讀腳本與獨立review同增量包。原四組OF配對／九端點保持，R13 Ef留空；QE無新結果，六個新尺寸仍未放行。新續算是否提交，以另一筆正式回執與CURRENT_STATUS為準；此段只記已核查的2254539結果。
<!-- END R13_AXIAL_TRIAL_RESULTS_20261018 -->

<!-- BEGIN R13_AXIAL_LINESEARCH_TRIAL_20261018 -->
## 2026-10-18 R13 軸向 line-search 短試已提交（最新）

- 實際作業 **2254539**，ctest／24GiB／2h，僅R13_0V一組；回執時間 `2026-10-18T05:15:56.446020+00:00`。執行至多60個離子步、120次logical電子呼叫及6600秒，沒有自動長續算或定時監控。
- 使用停止後step483的一致checkpoint，bundle SHA `11b4681bcdf6edbe1502c96015edd43fbbdeab87d6e0ccd05cad28ab6bb3f799`。恢復原子的移動與z軸晶胞自由度，以ASE BFGSLineSearch配合z-only FrechetCellFilter做有限短試；電子仍TN／1e−11Ha/atom，LDA/local、TF=.9/vW=.1與140×140×96網格保留。原子力與filter力均≤.001eV/Å、|σzz|≤.05GPa不變。
- 這是新的ionic候選：本機7項及遠端7項程式檢查、現役ASE API smoke與真實checkpoint preflight通過，尚無新物理端點。原科學核心沿用；新revision manifest SHA `0c4cd1072d87fe1fbe3aff60bc0c45d72a10e29f5dc0096dca8278b3fd9ab01e`。不可宣稱已解決收斂或新增R13形成能。
- 永久流程 `C:/OFDFT/nano_tensile_TFvW/workflows/al_nano_R13_axial_linesearch_20261018_v1`；來源、輸入引用、QA與正式回執在 `C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261018/RECOVERY_R13_AXIAL_LINESEARCH`。舊2253354主動停止紀錄與四組OF配對／九端點維持。QE k9資料與六個未放行尺寸狀態保持；本次沒有另送QE、上傳NAS或寄信。
<!-- END R13_AXIAL_LINESEARCH_TRIAL_20261018 -->

## 2026-10-18 13:00 大查核與主動停止（最新）

- QE2253295_0/1已完成k9 SCF，原runner重驗與82項獨立raw/XML檢查通過。k7→k9配對能差−1.215521meV，但最大逐原子力變化.018339/.005629eV/Å，k9 Fmax .046689/.033824；不是relaxed endpoints。六個新尺寸仍未放行，最小後續為兩個k11同幾何診斷，再選代表柱重新鬆弛網格；本次未提交。
- R13 gate2253353成功，抽查force FD最大誤差8.82e−7eV/Å、獨立初猜力差1.82e−5。TN起點驗證可保留。2253354後續電子連續成功，但離子LBFGS形成四步循環、步步maxstep=.01Å，2h42m同相位只降.0171meV。固定Lz同時仍驗cell filter造成額外收尾問題，不能再用相同fixedcell LBFGS耗完整天，不能放寬.001。
- root在已保存一致checkpoint與完整history後，依用戶先前避免無效耗費／排查未通過的授權，一次SIGTERM正常停止2253354，12:59:58結束。runner=interrupted，Slurm FAILED/1:0是主動停止，非OOM、時間牆或新電子失敗。最後完成step483，step484被中止；485次電子呼叫中最後一次因TERM中止。最後bundle SHA11b4681bcdf6edbe1502c96015edd43fbbdeab87d6e0ccd05cad28ab6bb3f799，Ne228，Fmax .003624287/filter .010078514，R13仍未驗收。下一步需恢復z自由度、可控收尾與震盪檢查，保留TN/1e−11及原模型／驗收。沒有宣稱新ionic方法已驗證。
- 新歸檔C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261018；初次91檔與停止後40檔SHA一致；POST_STOP/R13_0V是最新續跑來源，DFTPY/R13_RUNNING_SNAPSHOT僅歷史快照。原四組OF配對／九端點與canonical表未改。無新作業、定時監控、NAS上傳或寄信。

## 2026-10-18 未通過項目定向修復（最新狀態）

- 使用者明確要求找出問題、修正並重送。QE R15 k9兩個SCF已提交2253295_0/1，ct56/29ranks/256GiB/72h，提交快照PENDING；僅改k取樣，原PAW/PBE/800eV與幾何保留。k5→k7配對能差10.049meV仍未過5meV，故尚未放行六個新尺寸。
- R13已提交驗證2253353（ctest2h）與續算2253354（ct56 24h），afterok:2253353且程式重驗gate。
- R13是電子line-search失敗，非OOM/TIMEOUT；從已驗hash的step100 coherent checkpoint開始。電子改TN並以1e-11Ha/atom精度驗證；離子仍固定cell atomic LBFGS maxstep=.01Å，非FIRE。TN warm原生flag1可一次獨立HEG重試，失敗留錄，native0才計算力。最多8個logical／14個physical診斷SCF，核對獨立初猜與兩個有限差分步長。Fmax/filter≤.001eV/Å、|sigma_zz|≤.05GPa保持。
- 原runner核心與舊版逐位元相同，新修訂manifest79009e0248878103f7b2ac86b390e03843100324866678008dc0e69881d8c223；本機與遠端各25項測試及16項獨立程式審查通過，實際物理結果待跑。來源、輸入、hash、預檢與回執分別保存於workflows/al_nano_R13_electronic_20261017_v1及NAS包RECOVERY_R13_ELECTRONIC；QE同包QE_R15_K9_REVIEW。不得將prepared submission_metadata.json當實際回執。
- 原四組OF完整配對、九個端點維持；R13形成能仍缺。不定時監控、不放寬門檻、未上傳NAS／寄信。最新入口C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/RECOVERY_STATUS_20261018.json。


<!-- END RECOVERY 20261018 -->

# Al 計算流程索引

更新：2026-10-17。先查本索引，再準備計算。保留已成功使用的程式、輸入、版本、hash、驗收紀錄及代表結果；不因新一輪工作重新另寫整套計算程式。凍結的舊版本不得覆寫。

「程式測試通過」、「作業正常結束」與「物理端點通過驗收」分開記錄。新的最佳化修訂要有自己的版本與父版本來源；實際結果通過後，才能標為已驗證流程。

## 目前可追溯的流程

| 流程 | 保存位置 | 實際使用／驗收證據 | 使用範圍 |
|---|---|---|---|
| QE bulk k4 全晶胞 vc-relax | `C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/EXECUTION_SOURCE_FROZEN_20261015/QE_BULK` | `RAW/QE_BULK/0V`、`1V`；array 2245215_0/1；獨立 XML/stdout 驗收通過 | 保留 800 eV、PAW/PBE、原 PP/MP；k4 幾何通過，更密 k 點的能量驗證另做 |
| QE nano 代表柱取樣檢查與依賴控制 | 同包 `EXECUTION_SOURCE_FROZEN_20261015/QE_NANO` | 2245288 兩個 k5 SCF 正常；2245289 正確拒絕；六個後續端點未啟動 | 檢查程式的功能有效；不能將 gate 失敗當成取樣已通過 |
| DFTpy 修正應力的軸向鬆弛 | 同包 `EXECUTION_SOURCE_FROZEN_20261015/DFTPY_NANO`；完整父 snapshot 見 10/15 執行包 | 2245302 的 R15 總 zz 應力／抽查力／暖重啟通過；十端點九個符合門檻 | LDA/local、TF=.9/vW=.1、固定橫向、zz cell filter；R13_0V 的 ionic LBFGS 有震盪問題 |

完整的 10/15 輸入、結構、提交回執、偽勢與 manifest：`C:/OFDFT/NAS_SHARE/LUDER_NAS_UPDATE_20261015/EXECUTION_20261015`。
10/17 新結果與分析：`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017`。兩者應一併保存。

## 離子最佳化切換規則

使用者曾要求在 LBFGS 接近最低點後，以 FIRE 或其他穩定方法收尾，2026-10-17 再次明確重申。後續不得將已知低力區震盪的個案，再以相同 LBFGS 單階段跑到大量步數上限。

離子最佳化與電子密度最佳化是兩個層次。此次處理的是原子／軸長的離子最佳化；電子 LBFGS、物理模型與驗收門檻保持原設定。切換須保留同一晶胞 filter、完整配對的結構與密度 checkpoint，建立新的 optimizer 狀態，記錄實際切換步數、原因及參數。

R13_0V 本次耗時 18.76 h、1000 ionic steps，電子 1001 次均通過，末步 Fmax=.005827418 eV/Å。原 checkpoint bundle 與 final 完全對位，已下載保存。最後 100 步能量擺動 17.306 meV，適合針對離子收尾方法處理；保留 Fmax≤.001 eV/Å、filter force≤.001、|sigma_zz|≤.05 GPa 的要求。

使用者隨後特別要求核對「上次並非 FIRE」。本次以已用過的小步長原子 LBFGS 收尾為優先，FIRE候選未提交。R13使用既有coherent checkpoint、固定當前晶胞，原子maxstep=.01 Å，仍以原z-only filter計算最終廣義力與軸向應力；任一驗收失敗便保持未通過。這是歷史收尾方法在當前奈米柱模型的續算驗證，不宣稱已驗收。

已找回六月實際成功的歷史流程：hydrostatic FrechetCellFilter+BFGS（.02 eV/Å）→固定晶胞原子 LBFGS（maxstep .01 Å、當時目標 .002 eV/Å）。TF=.9/vW=.1 的 107 原子例在539步達 .0016483 eV/Å，符合當時要求，但尚未達目前 .001 的要求。該方法更換了晶胞自由度，不能直接套用取代本輪 z-only 軸向鬆弛。程式／log／hash詳見 `C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/ANALYSIS/DFTPY/historical_ionic_optimizer_audit.json`，原始檔保存在 `C:/OFDFT/AL_DEFECTS_USB_HANDOFF_20260717`。原 FIRE 試跑沒有達標；MDMin 試驗亦有 slow/bouncy 紀錄，全部保留作為避免重複試錯的依據。

## 保留原則

### 本次 R13 續跑已提交

2026-10-17 14:35:49（台灣）提交 **2252172**，提交後快照為PENDING；ct56、24h、24GiB、1task／1執行緒，Slurm配置詳見回執。只補R13_0V，FIRE未提交。新attempt為原runroot中的`R13_0V/attempt_002_smallstep`，沿用同一case lock，父attempt保持原樣。

固定保存的程式入口：`C:/OFDFT/nano_tensile_TFvW/workflows/al_nano_smallstep_20261017_v1/recover_nano.py`。此入口以R13及父checkpoint身份鎖定，其他case應複用科學核心與收尾模組並審核新case manifest，不重新寫核心。機讀索引為`docs/validated_workflows.json`。

完整程式、三處最小diff、原始方法證據、16項本機／16項遠端程式測試、真實checkpoint驗證及提交回執：`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/RECOVERY_R13_SMALLSTEP`。Revision manifest SHA256為`fff9270f92d11dbdd820150675196eb14ed75dc9c6eb5890febeb8ec0e05c0e1`。程式與執行前檢查通過；**實際R13收斂驗證尚待結果**。沒有定時監控。

## 保存與複用要求

1. 原始輸入、stdout、結果、結構、checkpoint、版本與 hashes 不刪除；大型重複檔可共用一份，索引必須指得回去。
2. 成功與未通過的案例都留結果與理由，避免重複踩同一問題。
3. 改動集中在必要模組，記錄父版本與差異；後續從已登錄入口複用。
4. 收斂力門檻不能為了消除失敗標記而放寬。已收斂的九個 OF 端點不因 R13 續跑而全部重算。


## 2026-10-17 R15 k7 診斷已提交

**2252206_8/9**，14:45:28台灣時間提交，當時PENDING；只執行R15 0V/1V固定幾何k7 SCF。重用10/15凍結k7輸入與原inspector、52檔hash不變；ct56/29ranks/256GiB/72h，兩組可並行。12項本機與12項遠端程式檢查通過，尚非物理結果。

新入口`workflows/al_nano_R15_k7_review_20261017_v1/run_reviewed_k7.py`只允許index8/9，原guard未更動。由k5相對k3的36.291322meV變化出發，先做k7，再據結果決定代表柱重新鬆弛及其他尺寸。六組新QE尺寸仍未啟動，原取樣gate不改為通過；原.001力門檻保持。不得把k7單點成功當成幾何或取樣收斂，不重複提交、不定時監控。

回執與審查在`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/QE_R15_K7_REVIEW`。四組OF成功配對及QE k4 bulk能量／結構分析在同包`ANALYSIS/ACCEPTED_RESULTS`。R13續算2252172仍待驗收。

## 2026-10-17 23:44 續算結果已核查（優先最新狀態）

- 既有SSH單次查核佇列空。2252172 R13小步長續算FAILED（29m19s），第102次電子計算line-search flag1，非TIMEOUT/OOM；最後完整checkpoint是ionic step100，Fmax=.003903133eV/Å、sigma_zz=-.006843516GPa。曾於step14到.001865808，仍未過.001；不填R13形成能，不把第101個提出步當完成。原九個OF合格端點／四完整配對保持。
- QE2252206_8/9兩個k7 SCF正常完成並通過診斷檢查（44m41s/47m53s），但k5→k7配對能差仍約+10.049meV（容許5meV），原k3幾何在k7 Fmax=.034336576/.035994055，sigma_zz=-.012108042/-.058926311GPa。單點不是重新鬆弛；六個新QE尺寸仍不放行。
- 新55檔（16,471,402bytes）remote SHA一致，含R13一致checkpoint、QE XML/stdout/結構，存`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/FOLLOWUP_RESULTS`；最新狀態、診斷與來源均在此。此次沒有新DFT提交／取消、NAS上傳、寄信或定時監控。後續先做代表柱更密取樣及R13同一checkpoint電子／力一致性診斷，勿無差別重送整批或放寬.001。

