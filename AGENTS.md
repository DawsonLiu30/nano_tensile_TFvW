## 2026-10-21 GitHub 保存與本機連線

- 本機 Windows Git Credential Manager 已以 DawsonLiu30 完成登入，fetch 與 push dry-run 通過；研究分支仍為 `codex/portable-qe-20260923`。同步狀態以實際 Git commit／遠端 ref 為準，登入不等於所有本機資料已上傳。
- 最新導覽見 `docs/REPOSITORY_INDEX.md`、`docs/GIT_SYNC.md`；發佈快照驗證用 `python scripts/verify_research_snapshot.py`。`archives/research_sync_20261021` 保存既有來源及小型驗收證據的逐位元副本，原本機資料包保持不變。大檔按 manifest 引用；不把 GitHub clone 稱為完整 restart 備份。
- campaigns、workflows、archives 與校準原件採 Git `-text` 保存，避免 Windows／Linux 換行轉換破壞 hashes。凍結版本仍不可原地改寫；普通文件後續更新應另提交，不倒改歷史 manifest。
- 私人信件草稿留在本機。沒有自動同步或 DFT／scheduler 動作；此輪 Git 工作不代表科學驗收狀態改變。

## 2026-10-21 21:33 QE重啟與迭代已實測確認

21:33 台灣時間啟動查核完成：三個v2續算已實際讀到保存密度與波函數，進入第16次SCF；原子初猜警告及讀檔回退均未出現。0V/1V k5均進入第1次SCF。五個計算RUNNING；三個後續k5等待各自k4通過，沒有新完成端點。三個v1正常停止、舊依賴取消。已停止遠端查詢；沒有持續監控。

<!-- BEGIN QE_BULK_RESTART_FILE_FIX_20261021 -->
## 2026-10-21 QE bulk重啟修正 v2（優先於下方v1）

v1 2285948啟動檢查發現startingwfc atomic+random may spoil restart；依使用者續算授權正常停止三組。不是物理參數或力門檻改動。舊待跑2285950/51/52/53取消；0V/1V k5 2285949保留。v2 2286031_2/3/4從原10/15 checkpoint續跑，restart_mode=restart、startingpot=file、startingwfc=file；2286032_2、2286033_3、2286034_4為各自afterok k5，2286038一次性分析。以完整job IDs與最新回執為準，不重送。

QE7.5對錯誤startingwfc只警告，不會自動改file；舊已成功nano wrapper的操作選項不能未查直接套第一SCF中斷。以密度讀檔、Starting wfcs from file、沒有fallback及實際迭代前進驗證。新入口workflows/al_bulk_reference_resume_k5_20261021_v2，manifest d42f9fff5f3542f91873f3a52426864f3e8bfd8a9f0e7831d477fc1c57a6f482；27項程式檢查通過。原物理／.001力／fullstress與72h不變。尚無新accepted endpoint，不持續監控。
<!-- END QE_BULK_RESTART_FILE_FIX_20261021 -->

<!-- BEGIN QE_BULK_RESUMED_20261021 -->
## 2026-10-21 QE bulk 雙空缺續算與五態 k5 已提交（最新）

使用者明確授權後，已查fresh queue空及357父scratch hashes；三組第一SCF第15次迭代正常停止的checkpoint完整。array2285948_2/3/4分別1NN/2NN/5NN原k4 vc-relax續算；2285949_0/1補已接受0V/1V的k5；2285950_2、2285951_3、2285952_4各自afterok對應k4，驗收通過後才做k5；2285953一次性分析。未提交其他DFT矩陣或自動監控。

72h，續算29ranks/64GiB，k5 29ranks/128GiB且disk_io=medium；PAW/PBE800eV/MP.02/原PP不變。Fmax≤.001、component≤.0005eV/Å、全stress≤.05GPa/finalSCF1e−10與原BFGS驗收保留。k5僅固定幾何診斷，形成／結合能每個網格各用自身E_ref_Al；5meV判讀解析度不等於已完全取樣收斂。

原已驗證科學核心不改，小型續算／診斷adapter經23項程式測試；新physical endpoint待跑。永久入口workflows/al_bulk_reference_resume_k5_20261021_v1；本機執行歸檔C:/OFDFT/NAS_SHARE/QE_REFERENCE_EXECUTION_20261021；遠端AL_BULK_REFERENCE_RESUME_K5_20261021。細節docs/QE_BULK_REFERENCE_RESUME_20261021.md及registry。已完成workitem包不混入此次未完成結果。後續查fresh狀態與回執，不重複提交。
<!-- END QE_BULK_RESUMED_20261021 -->

<!-- BEGIN QE_REFERENCE_FOCUS_20261021 -->
## 2026-10-21 最新研究優先順序：QE bulk vacancy/divacancy reference

Lüder要求先聚焦QE單／雙空缺參考，完成資料單獨放workitem，補vacancy-site分析並替換mu(Al)記號。詳見 `docs/LUDER_QE_REFERENCE_FOCUS_20261021.md`。本機分享包 `C:/OFDFT/NAS_SHARE/workitem_QE_bulk_vacancy_reference_20261021` 只含完成k4鬆弛的0V/1V及其分析；k取樣與有限尺寸驗證仍待補，k4三組2V先前正常停止未完成。新寫作統一 `E_Al^ref=E_0V/108`（每原子參考能），原raw不改。NAS仍由使用者手動上傳。本輪沒有新DFT或遠端查詢；後續先複用registry中的QE流程與查fresh狀態，不據舊回執重送。
<!-- END QE_REFERENCE_FOCUS_20261021 -->

<!-- BEGIN STRUCTURE_SUPPLEMENT_20261019 -->
## 2026-10-19 五尺寸結構能量補件

現有資料增修在 `C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261019_SUPPLEMENT`，詳見 `docs/AL_STRUCTURE_ENERGY_SUPPLEMENT_20261019.md`。五點非單調；R11/R13大位移經限制z軸旋轉診斷幾乎不變，近鄰與q4/q6只作幾何描述，未認證晶相。新主圖的12NN採映射後鬆弛0V參考，非ideal原始12NN。原raw部分結構header遺留初始標籤，狀態以result.json為準；另有僅修metadata的可視化副本。原10/19封存包與成功DFT流程不改。

本輪無新DFT／監控／寄信。使用者指定NAS自行手動上傳；只整理本機分類。QE代表柱密網格重新鬆弛與R13電子精度配對等列條件性計畫，未提交，也未完成所有物理驗證。
<!-- END STRUCTURE_SUPPLEMENT_20261019 -->

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

## 2026-10-17 23:44 續算結果已核查（優先最新狀態）

- 既有SSH單次查核佇列空。2252172 R13小步長續算FAILED（29m19s），第102次電子計算line-search flag1，非TIMEOUT/OOM；最後完整checkpoint是ionic step100，Fmax=.003903133eV/Å、sigma_zz=-.006843516GPa。曾於step14到.001865808，仍未過.001；不填R13形成能，不把第101個提出步當完成。原九個OF合格端點／四完整配對保持。
- QE2252206_8/9兩個k7 SCF正常完成並通過診斷檢查（44m41s/47m53s），但k5→k7配對能差仍約+10.049meV（容許5meV），原k3幾何在k7 Fmax=.034336576/.035994055，sigma_zz=-.012108042/-.058926311GPa。單點不是重新鬆弛；六個新QE尺寸仍不放行。
- 新55檔（16,471,402bytes）remote SHA一致，含R13一致checkpoint、QE XML/stdout/結構，存`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/FOLLOWUP_RESULTS`；最新狀態、診斷與來源均在此。此次沒有新DFT提交／取消、NAS上傳、寄信或定時監控。後續先做代表柱更密取樣及R13同一checkpoint電子／力一致性診斷，勿無差別重送整批或放寬.001。



## 2026-10-17 R15 k7 診斷已提交

**2252206_8/9**，14:45:28台灣時間提交，當時PENDING；只執行R15 0V/1V固定幾何k7 SCF。重用10/15凍結k7輸入與原inspector、52檔hash不變；ct56/29ranks/256GiB/72h，兩組可並行。12項本機與12項遠端程式檢查通過，尚非物理結果。

新入口`workflows/al_nano_R15_k7_review_20261017_v1/run_reviewed_k7.py`只允許index8/9，原guard未更動。由k5相對k3的36.291322meV變化出發，先做k7，再據結果決定代表柱重新鬆弛及其他尺寸。六組新QE尺寸仍未啟動，原取樣gate不改為通過；原.001力門檻保持。不得把k7單點成功當成幾何或取樣收斂，不重複提交、不定時監控。

回執與審查在`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/QE_R15_K7_REVIEW`。四組OF成功配對及QE k4 bulk能量／結構分析在同包`ANALYSIS/ACCEPTED_RESULTS`。R13續算2252172仍待驗收。

# 2026-10-17 必須複用已驗證流程與最佳化收尾

- 使用者隨後提醒「上次並非FIRE」；歷史確實成功的是cell BFGS→fixed-cell atomic LBFGS maxstep=.01Å（當時force .002），而非FIRE。已優先沿用此收尾家族，保留目前.001門檻及原z-filter/σzz驗收。
- **2252172已於10/17 14:35:49台灣提交**，唯一R13_0V固定晶胞小步長LBFGS續跑；ct56/24h/24GiB/1task，提交後PENDING。新`attempt_002_smallstep`沿父runroot case lock；電子LBFGS、LDA/local/.9/.1、140×140×96與stress候選不變，FIRE未提交。16項本機與16項遠端程式檢查、真實bundle/hash/config通過，不代表已物理收斂。禁止重複送算，不開監控。
- 永久本機入口`workflows/al_nano_smallstep_20261017_v1/recover_nano.py`、registry `docs/validated_workflows.json`。遠端snapshot `/work/dawson666/src/campaign_snapshots/al_nano_R13_smallstep_20261017_v1`；runroot `/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015`。回執與版本在`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017/RECOVERY_R13_SMALLSTEP`。提交回覆餘額1322.6235是當時值，舊191不是目前值。
- 使用者再次明確要求：計算已成功使用並驗收的腳本、輸入、版本、hash與成功結果證據必須保存；後續先讀 `docs/VALIDATED_WORKFLOWS.md`，從已登錄流程複用，不每輪重寫。新修訂與已實際驗收的版本分開，舊凍結snapshot不覆寫。
- 已知 ionic LBFGS 低力區震盪時，採用LBFGS→FIRE或其他經驗證的分段收尾；不得再用同一LBFGS反覆耗到步數上限，也不得放寬Fmax≤.001 eV/Å。電子LBFGS與離子optimizer不同，切換需保留同一cell filter及coherent geometry+density checkpoint，明列真實optimizer與切換原因。
- 10/17 14:15單次查核queue空；2245215的bulk0V/1V通過，k4 Ef=.677935897 eV（舊k3 .644962822，差32.973075meV），k收斂仍待更密網格。R15k5單點完成但gate拒絕：Δ(E1−E0)=36.291322meV、fmax=.059967544/.049514736；六組新QE尺寸實際未啟動。
- OF gate通過，10端點9通過：R11/R15/R17/R20完整配對Ef=.424814871/.598150394/.628738683/.631826061eV；R13_0V ionic LBFGS1000步未過（Fmax=.005827418），不是TIMEOUT/OOM，1001電子呼叫皆通過。R11有廣泛結構鬆弛，218幀缺格仍原位置，勿稱空缺遷移或凝聚。
- 新原始資料380檔remote hashes一致，資料與分析在`C:/OFDFT/NAS_SHARE/LUDER_RESULTS_20261017`；未覆寫舊canonical值。OF127項、QE262項獨立資料檢查通過；並非額外物理計算數量。後續R13分段續算另留新版本／提交回執，不混入這份查核快照。

# 2026-10-15 圓柱生成與 QE 原子數限制

- 使用者補充老師要求QE結構最多250顆原子，作為本專案的運算規模上限。後續QE輸入、尺寸增加或軸週期加倍，提交前必查實際nat≤250；超過須先縮減或另行確認，不能以nbnd當原子數。現有五柱0V/1V為52/51、76/75、96/95、128/127、176/175，全部通過。
- 本輪使用ASE3.29檢查20個實際提交/現有R20參照結構，226項查核通過。20組最短週期平移≥13.397763Å，橫向原子中心投影間隙≥10.310586Å，無重複/過近格點，且均有跨z邊界近鄰。查核副本、三張圖、CSV與腳本在`C:/OFDFT/NAS_SHARE/LUDER_NAS_UPDATE_20261015/STRUCTURE_PP_AUDIT_20261015`。圖中R15為本輪暖重啟/單點起點，QE R20為已驗收參照，勿全部稱為新鬆弛結果。
- 圓柱為FCC晶格確定性圓形裁切，`app/ase_nanocrystal.py`採x²+y²≤R²；[111]沿z，三向PBC，橫向Lx=Ly=2R+10Å。新增三尺寸採固定R20母晶格/原點的巢狀裁切，來源已核對。面積直接πR²。目前工作樹及可用本機歷史未找到Monte Carlo切割或體積程式；唯一Voronoi字樣是最近父格點佔據分析。不得宣稱已確認老師所說MC就是體積估算。
- 使用者提供EACcodes/local-pseudopotentials來源：目前四份OF `al.lda.recpot`均與commit `7a0dc7a1236a1d0a86f8a71c89999079bd14edec`的`BLPS/LDA/reci/al.lda.recpot`位元相同，SHA256 `c2fd4300aa39aa2466149a315e1c7cd87aac6615032db5ded8f237e9c40df803`；README明列PZ-LDA，現行DFTpy為lda_x/lda_c_pz。QE可用PZ-LDA，但PZ-PAW與同一local BLPS是兩種不同控制方案。QE7.5支援local-only UPF，該庫沒有現成UPF，recpot不可直接改副檔名；同BLPS的可信轉換與驗證尚未做。DFTpy讀PAW UPF只取local部分，不能假稱兩程式同用PAW。完整官方來源與最小對照方案在該查核包`evidence/pp_review.txt`；此輪不更換生產偽勢。
- 此輪僅結構/來源分析，未修改或新增作業。追加遠端唯讀hash查核時SSH已失效；以先前提交成功時的遠端hash回執與本機凍結輸入查核，未宣稱再次讀到遠端。

# 2026-10-15 20:53 五半徑 QE／DFTpy 已提交（目前優先狀態）

- 使用者重新登入後，QE完整回執已下載並核對Slurm依賴；2245288兩個R15 k5 task在20:53已實際進入第9／8次電子迭代。2245289 afterany檢查、2245290/91 afterok等待條件正確。不得重複提交。
- DFTpy已真正提交：**2245302** ctest、1CPU、24GiB、2h，做R15 0V/1V的總軸應力有限形變、重新自洽、力有限差分與修正stress暖重啟驗證；**2245303** ct56、array0-7%8、每組1CPU/24GiB/24h，afterok:2245302且kill-on-invalid-dep=yes。包含R11/R13/R17六個新端點，加R20 0V/1V暖重啟重驗。沒有修改已安裝DFTpy，使用process-local候選；尚不能稱總stress驗證已完成。
- OF遠端snapshot `/work/dawson666/src/campaign_snapshots/al_nano_five_radii_of_20261015_20261015T124450Z`，runroot `/work/dawson666/dftpy_project/relax/AL_NANO_FIVE_RADII_CORRECTED_20261015`。41檔遠端hash與dry通過；四個密度由原遠端結果驗hash後複製。完整manifest SHA256 `e2a78d9c360f5fbef8ec267a0565e2f74d17542e5641262071cf3bbb5fbe1674`。test-only2245300/01不是生產。
- OF本機41項、QE18項程式／數值檢查通過；本機未進行新的自洽DFT。20:53 OF驗證執行中，R15_0V暖重啟已輸出converged，成對gate尚未確認通過。沒有定時監控。
- 本機交付增量：`C:/OFDFT/NAS_SHARE/LUDER_NAS_UPDATE_20261015/EXECUTION_20261015`。7頁`20261015_Al_五尺寸執行對照_提交確認版.pptx`依使用者指定kickoff模板製作，PowerPoint逐頁QA通過；含原信逐項處理、真實提交回執、輸入／結構／程式與驗證。大密度引用原有Al_QE_DFTpy_DATA，未重複保存。NAS仍無入口，**未上傳、未寄信**。最新執行優先順序取代先前五個bulk並跑，但原始科學數據保留。
- 詳見`docs/LUDER_EXECUTION_20261015.md`。新五尺寸完成前保留兩尺寸觀察的範圍；smearing/真空/網格/週期、完整QE PAW動能/Vei及有限LDA對照仍有待做，不能把提交當作完成。

# 2026-10-15 20:39 依老師要求重排作業（歷史快照）

- 使用者明確要求把五半徑nano放主線、重審bulk成本。20:28對2245215_2/3/4（1NN/2NN/5NN）用QE prefix.EXIT正常停止；20:31確認pw.x exit0、wrapper因未達科學終態為FAILED，此為主動延後，不是新故障。原始資料及357個scratch檔保留；2245215_0/1（bulk0V/1V）繼續。
- QE五尺寸所缺的六個端點已真正提交，非僅備結構：**2245288** R15_0V/1V k5單點array6-7%2；**2245289** gate afterany:2245288；**2245290** R11/R13四組128GiB，**2245291** R17兩組192GiB，均afterok:2245289、kill-on-invalid-dep=yes。初始20:39快照PENDING；29ranks、72h/71h；原PAW/PBE800eV、force norm .001/component .0005、z-only vc-relax保持。Gate須k5力/SCF/zz應力達標且固定幾何(E1−E0)相對k3改變≤5meV才放行。5meV為操作容許，非全面收斂證明；新半徑、smearing/真空/週期敏感度仍有檢查範圍。
- QE snapshot `/work/dawson666/src/campaign_snapshots/al_nano_five_radii_qe_20261015_20261015T123901Z`，runroot `/work/dawson666/qe_cases/qe_runs/AL_NANO_FIVE_RADII_QE_20261015`。本對話`work/luder_execution_20261015/qe_submission_confirmed_summary.json`保存成功回覆；52檔遠端hash與verify通過。18個runner/gate測試通過。預檢2245284–2245287不是生產。不得因隨後SSH失效而重複提交。
- 20:40 SSH失效，已向使用者請重新登入；DFTpy驗證/新尺寸仍準備中，尚未提交。沒有監控。NAS無可用掛載或登入分頁，已詢問可用入口；未上傳。

# 2026-10-15 Lüder 回饋處理與 bulk k4 補算

- 使用者再次強調老師要求約五種不同半徑的奈米柱，這是本輪必做主線。規劃R11/R13/R15/R17/R20，每半徑含0V與1V，QE與DFTpy都算：共20個端點。已有R15/R20的8個計算端點（各有尚待完成的取樣／OF應力驗證），新增R11/R13/R17需要12個端點，目前只有結構，尚未提交。bulk五缺陷態array2245215不能計入此五尺寸要求；生成五種結構也不等於完成五個尺寸的能量趨勢。
- 讀 `docs/LUDER_REVIEW_RESPONSE_20261015.md`。使用者寄出的 Desktop `20261013_Al_vacancy_progress.pptx` 實際13頁，頁腳仍為1,2,4,5,6,7,8,9,10,11,13,14,15；老師依頁腳評論，不能把 slide4 誤認為 bulk 診斷。
- 原始力重新核對35列：目前24個鬆弛終態全部 Fmax norm≤0.001 eV/Å（QE bulk8、OF bulk8、QE nano4、OF nano4）。另外11列為固定幾何診斷，其中7個k4缺陷點力0.029504–0.045689 eV/Å。SCF診斷通過不等於幾何通過；已撤下其平衡配對能量比較。此24列結論不涵蓋早期其他門檻的校準。
- 已真正提交 **2245215**：0V、1V、1NN、2NN、5NN，ct56、29 ranks、64GiB、72hSlurm/71hQE、array0-4%5。2026-10-15 19:13:44台灣唯一提交後快照為PENDING/QOSGrpJobsLimit；無監控。維持PAW/PBE800eV、原PP/MP.02Ry，k4重新全cell vc-relax、norm≤.001/分量≤.0005 eV/Å、6應力分量≤.05GPa、finalSCF≤1e-10Ry。k4是驗證而非已收斂設定，後續需matched denser-mesh檢查。test-only2245212不是正式作業，禁止重複提交。
- Snapshot `/work/dawson666/src/campaign_snapshots/al_bulk_k4_validation_20261015_20261015T111240Z`；runroot `/work/dawson666/qe_cases/qe_runs/AL_BULK_K4_VALIDATION_20261015`。輸入/程式/回執在本對話 `work/luder_response_20261015/bulk_k4_validation_20261015`。部署34檔local/remote hashes均通過。
- 四個nano1V最短影像距離：QE R15/R20=13.623481/13.825627Å，OF=13.397763/13.471846Å，均≥10Å；是幾何要求通過，並非週期能量收斂認證。空缺近軸父格點CN12，R15的12近鄰有2個原本CN11的表面格點，R20近鄰均CN12。QE R20全體umax較小但第一近鄰平均內移較大，避免把umax當作鬆弛能代理。
- 生成同母晶格5尺寸R/a0=1.1/1.3/1.5/1.7/2.0（0V原子數52/76/96/128/176）；新增R11/R13/R17只有結構，未提交nano DFT。保持[111]軸週期與橫向每側5Å padding。應先完成代表柱取樣與OF總stress重驗，再擴尺寸。
- OF能量分項已按同方法bulk參考形成能分解；QE one-electron不可命名為純Ts，完整PAW Ts/Vei尚待正確後處理。DFTpy是PZ-LDA；QE-LDA控制方案已查官方PP，未提交。現行力達標不替代nano取樣或OF stress驗證。
- 修訂13頁簡報與數據/程式/回執增量包：`C:/OFDFT/NAS_SHARE/LUDER_NAS_UPDATE_20261015`，尚未上傳NAS或寄信。原始canonical數據與10/14 Word稿保持不變；新版PPT作為對此次回饋的回應。

# 2026-10-14 正式 proposal 與 R20 已完成（優先於下列歷史狀態）

- 最新可編輯正式稿：`C:/OFDFT/PROPOSAL/鋁空缺與奈米柱研究計畫_正式版_20261014.docx`。維持中文正文、中英摘要、標楷體／Times New Roman、每式獨立解釋。新增 Sholl–Steckel 2009 理論依據、前期 KEDF 比較、100 組 TFvW 選參及已驗收 R20；12 圖、22 表、26 原生公式、20 篇文獻。詳細來源與驗證見 `docs/PROPOSAL_FORMAL_REVISION_20261014.md`。
- 10/14 單次查核：作業 **2234373** 已於 10/13 21:17:37 台灣時間結束，凍結 runner inspect **accepted=true**，查核時 queue 空。QE R20 1V 正式 Ef=**0.47675189365281767 eV**，較最後 optimizer 暫定值高 **0.8071027696 meV**；fmax=.000463456509 eV/Å，sigma_zz=.000113131210 GPa，SCF error=5.06519716e-11 Ry。正式值不再為 null。
- 原始輸出/XML/結構已 hash 歸檔 `C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/QE/08_Nanocolumn_R20_final_SCF_ACCEPTED_20261014`；四組 nano 形成能與 540 個共同格點位移在 `REPORTS/nano_progress_20261014`，238 項資料查核通過。QE R20−R15 Ef=−270.366081 meV，OF=+33.675589 meV。Nano k／真空／軸週期與 OF 總應力驗證仍待做；舊 bulk k3 排序與 k4 固定幾何診斷的範圍維持。
- 權重選擇來源已找回：六月 LDA/local、h=.20 Å、108/107 原子 100 組掃描，77 組通過當時力／應力／晶格檢查。原始排序採 **QE 64/63 參考 0.601167 eV**；TF=.9/vW=.1 的 Ef=.665082516 eV、a0=3.950912158 Å，在合格候選中最接近該形成能。原 Excel λ=TF、μ=vW，兩者獨立掃描；四月稿 λ 曾指 vW，正式稿統一 α=TF、β=vW。不要以後來 collector 預設 .638912226 eV 倒改原選參依據。此為模型校準，不是獨立 vacancy 驗證或現行 78³ 網格測試。
- 舊稿圖 1 的 KEDF 指標是奈米柱每原子超額能量相對 QE 的 MAE：LMGPA .28、SM .61、WT 1.32、TFvW 1.77、vW 4.42 eV/atom（原圖精度）。TFvW 的選用理由是形式簡單、權重可調與結構鬆弛工作流程；不能寫成能量排名第一，亦不能由歷史選型認證目前 surface stress。
- 新增證據集中 `C:/OFDFT/PROPOSAL/SOURCES_20261014`；132 檔 hash 驗證，39 個既有 canonical 來源未變。本輪沒有提交 DFT、定時監控、上傳 NAS 或寄送文件。10/13 PPT 保持原本暫定快照，未同步覆寫。

# 2026-10-13 Lüder 英文進度更新

- 最新寄送用簡報：`C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/REPORTS/Luder_update_20261013/20261013_Al_vacancy_progress.pptx`，16 頁，沿用 9/29 附件風格，先校準、再 bulk QE 查驗、再 R15/R20 nano 能量與結構。數據／程式／來源索引附於同目錄，詳見 `docs/LUDER_PROGRESS_UPDATE_20261013.md`。本輪未上傳、發信、新送 DFT 或監控；R20 暫定值、OF pilot 和 k4 固定幾何範圍皆保留。

# 2026-10-13 19:59 R20 結束查核與終態 SCF 補完（最新作業）

- 使用者隨後明確要求：既然幾何力門檻已達標，先以現有結果分析。可使用清楚標為「暫定／最後最佳化步」的 QE R20 Ef=.4759447909eV 與成對結構結果，不必等final SCF才討論。原正式accepted欄保持null，避免把資料驗收狀態和可進行的初步分析混在一起。最新比較為`REPORTS/nano_progress_20261013/PROVISIONAL_ANALYSIS_ZH.txt`及`provisional_size_comparison.csv/json/png`。
- 已量化：末5個optimizer步能量僅變.003347meV；未完成final SCF當前能量相比末optimizer高.802165meV，對應Ef=.4767469562eV（仍非converged）。這支持以約.476eV先討論，但.8meV不是剩餘誤差上限。R20−R15：QE −271.173meV、OF +33.676meV；同方法相同bulk mu在此差值中精確消去，差異來自四個柱體能量。Nano取樣及OF stress驗證仍待做。
- 空缺誘發umax：QE R15/R20=.134335/.085865Å，OF=.145032/.071852Å；兩方法都較細柱局部位移大。空缺引起軸向變化QE +.07591%/+.16587%，OF −.14701%/−.10172%；這兩種觀察應分開討論，位移較大本身不決定形成能高低。R20小於bulk、R15大於bulk的QE暫定值尚不支持普遍單調尺寸規律。
- 父作業 2221957_1 於 10/13 10:47:03 結束，Slurm FAILED 是 wrapper 未驗收，pw.x 本身 exit0/JOB DONE。實際原因為 final relaxed-cell SCF 達 QE 71h max_seconds。已完成 BFGS：77 SCF cycles／76 BFGS steps；本次續跑有51個完整步（前次26），電子迭代標记1719，非空跑。
- 最後完整最佳化步：fmax=.000464134255、最大分量=.000416173063 eV/Å、sigma_zz=.000122419486 GPa、Lz=13.825627084719253Å，均達幾何門檻。Final SCF 第57次電子迭代誤差1.282993e-9 Ry，尚高於1e-10；XML output無forces/stress且opt_conv=false。幾何完成依據是stdout BFGS、最後完整step與output幾何一致，不可把不完整output能量配成正式端點。R20正式Ef仍null。
- 沿用使用者「未通過要解決並重送」及10/10加值後續跑授權，只補此終態SCF：已提交 **2234373**，19:59台灣PENDING/QOSGrpJobsLimit。ct56/mst114175、29 ranks、256GiB、72hSlurm/71hQE；沒有定時監控。test-only 2234372不是正式作業；提交時錢包192.2371（歷史值）。
- 新campaign `al_nano_R20_final_scf_20261013`，遠端snapshot `/work/dawson666/src/campaign_snapshots/al_nano_R20_final_scf_20261013`，run root `/work/dawson666/qe_cases/qe_runs/AL_NANO_R20_FINAL_SCF_20261013`。case `nano_R20_1V_final_scf/attempt_001`。查新作業，勿只用舊雙case status helper。使用 `scripts/run_nano_final_scf_20261013.py inspect --campaign <snapshot>/campaigns/al_nano_R20_final_scf_20261013 --attempt <runroot>/nano_R20_1V_final_scf/attempt_001`。
- 此為已完成axial vc-relax後的final SCF：calculation=scf/nstep1，固定父final-cell XML幾何，移除IONS/CELL，restart_mode仍restart，明设startingpot/wfc=file。其餘cutoff/kmesh/PP/XC/MP/電子門檻未變。父BFGS history在完成後已無.bfgs，不能套舊runner的必需.bfgs規則；120個density/wfc/mixing/restart檔共8.88GB預檢存在。作業起跑才複製逐檔驗hash；19:59尚未起跑，不能宣稱已完成scratch複製或實際電子迭代。
- 本機與遠端各5組程式測試通過，包括rehashed物理參數異動拒絕、幾何異動拒絕與force/stress驗收。新inspect需同時验证父BFGS、固定幾何、電子、fmax<=.001、分量<=.0005 eV/Å、|sigma_zz|<=.05GPa。即使SCF完成也須重驗力，不可僅用SCF退出碼。
- 新29原始檔hash下載、467項資料/步序查核。歸檔 `C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/QE/06_Nanocolumn_R20_geometry_converged_20261013`；新輸入與回執`QE/07_Nanocolumn_R20_final_SCF_20261013`；0V沿用`QE/05_...`accepted資料不重複複製。分析`REPORTS/nano_progress_20261013/REPORT_ZH.txt`，狀態`current_status_20261013.json`，新manifest72檔。未上傳NAS。
- R20暫定0V→1V結構比較：ΔLz=+.022894939Å／+.1658725%；移除均勻晶胞變形及平移後umax=.08586535Å、uRMS=.02561605Å。最後optimizer能量的形成能診斷.47594479eV不等於final-SCF正式值，僅存provisional診斷，不填正式比較表。Proposal仍保留10/12資料截止，不擅自把本輪未完成端點寫成完成。

# 2026-10-13 正式稿格式與寫作偏好（優先遵循）

- 使用者要求中文標楷體（DFKai-SB／標楷體）、英文 Times New Roman；公式參照四月 `劉兆祥proposal(comments).pdf`，每式獨立置中、編號固定靠右、間距一致，每式下方解釋符號與用途。數學結構使用可編輯 Word 公式，保留原稿的數學字體；不可把多個獨立公式擠在同一行。
- 中文正文應簡潔清楚，讓沒有本領域背景的人也能理解，程度符合碩士研究生。先解釋問題、方法和數據代表的意思；刪去說教、反覆否定與內部稽核式敘述。研究限制保留必要事實，以直接正面句型交代。
- 既有 QE／DFTpy 校準是研究成果的開頭，報告須展示其數據、圖表、測試目的與選用條件，再介紹 bulk／nanocolumn 結果；不得只用一段話略過或以缺少 metadata 為正文主軸。歷史 PBE／BLPS 與目前 LDA／TFvW 的資料來源仍各自標清。
- 最新可編輯稿：`C:/OFDFT/PROPOSAL/鋁空缺與奈米柱研究計畫_中文修訂稿_20261013.docx`。41 頁、11 圖（其中 5 幅校準）、20 表（其中 6 張校準）、22 個逐式解釋的原生公式、19 篇已核實文獻。依原 PDF 量測版心與公式位置重排，中文正文 12pt／21pt 行距；全頁 Word 渲染檢查通過。
- 建置與 QA：`C:/Users/m1331/Documents/Codex/2026-09-24/g/work/proposal_revision_20261013`。25 項文件檢查通過；39 個 canonical 來源 hash 未變、原 108 項數值查核保留，新增校準 14 個原始點的 70 項檢查通過。資料截止仍為 2026-10-12 19:08，沒有新增 DFT 或遠端監控；未完成結果維持待完成。下列 10/12 初稿為歷史版本。

# 2026-10-12 中文 proposal 初稿（撰稿成果，非新計算）

- 使用者指定中文正文、中英摘要，參照四月 `劉兆祥proposal(comments).pdf` 格式。已完成可編輯 Word：`C:/OFDFT/PROPOSAL/鋁空缺與奈米柱研究計畫_中文初稿_20261012.docx`，39 頁、7 圖、15 表、11 個原生公式、19 篇核實文獻；資料截止仍為下列 19:08 快照，沒有新增 DFT 或監控。
- 七章涵蓋研究目的、文獻、理論、既有校準與方法、bulk/nano 結果、階段結論、11/30 前工作。明列舊 QE k3 排序未收斂、k4 尚為舊幾何單點診斷、R20 QE 1V 缺合格端點及 OF nano stress production 驗證缺口。單空缺曾參與 TFvW 候選篩選，不是完全獨立驗證；不可宣稱表面一定降低形成能。
- Word 建置、39 個來源 hashes、108 項數值查核、文獻核實記錄與逐頁 QA：`C:/Users/m1331/Documents/Codex/2026-09-24/g/work/proposal_draft_20261012`。正式交付為上述 DOCX；QA PDF/頁圖僅供檢查，不是另一版研究報告。新端點完成後應更新第五章與附錄來源，避免倒改歷史快照或將規劃當已完成結果。

# 2026-10-12 19:08 最新 QE R20 狀態（單次查核）

- array2221957：0V於10/12 07:14:12 COMPLETED且frozen runner inspect accepted；1V在19:08仍RUNNING，有電子迭代（18:58至19:08計數1338→1342），29 ranks完整，job memory peak177.96GiB/256GiB、OOM/failcnt=0。Slurm時間牆10/13 11:45:27，QE內部71h會稍早停止；未改參數、未重送、無定時監控。
- R20 0V最終SCF E=-94548.01223174046eV、Lz=13.80273214568Å、fmax=.00045886987eV/Å、max component=.00041487636、sigma_zz=.00012337748GPa。最終SCF比最後optimizer步高.920791meV，後續形成能使用final SCF，不選較低前一步。
- R20 1V快照最後完整步fmax=.00108565444、component=.00100864948eV/Å（門檻.001/.0005），未accepted；正式R20 Ef保持null。不要用running XML output的新幾何與先前步能量拼終態；使用已對位的XML step與stdout total-force block。
- 新28原始檔已hash下載，432項資料/步序/終態/結構檢查通過，非新的DFT收斂實驗。本機 `C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA/QE/05_Nanocolumn_R20_restart_20261010`；解說/數值表/圖於 `REPORTS/nano_progress_20261012`；根目錄RUN_STATUS_20261012與current_status_20261012.json作最新狀態覆蓋。原case_index/689檔manifest保留歷史，不倒改QE/04旧attempt。
- QE 7.5 XML step.scf_error已是Ry，final output.scf_error為Ha；分析換算需分開。已核對官方add_qexsd_step.f90/pw_restart_new.f90及stdout。原production runner讀final output的×2換算正確，不因step的特殊單位誤報runner bug。
- QE完整柱R15/R20相對各自初始柱軸向收縮2.7698%/1.4157%，是0V表面鬆弛，不能稱vacancy-induced strain。nano k/vacuum/period轉用驗證、OF stress、bulk QE pair k敏感問題均仍未解決。下列既有校準規則繼續有效。

# 2026-10-11 優先規則：既有收斂校準必須先讀、先用、先介紹

- 本日01:07–01:15台灣時間已完成iService與35組主資料全面核對，讀 `docs/ISERVICE_CALIBRATION_AUDIT_20261011.md`。兩個2221957 R20續跑仍RUNNING、有迭代且各29ranks完整；此為單次快照，不開定時監控。不要把舊R20未收斂result倒改成新終態。
- 此次另發現restart輸入仍帶startingwfc='atomic+random'，stdout有警告與atomic wfcs初始化。QE7.5對startingpot會自動改file，startingwfc只警告；SCF/BFGS計數確有接續，但不能宣稱所有wfc歷史正確讀回。現役不改動；後續restart應明設startingpot/startingwfc='file'，另建可追溯版本及測試，不能繞過原runner「僅兩欄不同」檢查。詳細證據與官方來源見查核文件。
- 準備任何 Al QE/DFTpy 計算、提交新作業、寫報告或 PPT 前，先讀 `docs/CONVERGENCE_BASELINES.md` 與 `docs/convergence_baselines_20261011.json`。這是使用者明確要求的長期記憶，優先於下方歷史快照與舊腳本預設。
- 附件 `qe_bulk_vacancy.pptx` 已完整重讀與保存：原始單原子 primitive FCC 的 40³/80³，a₀=4.039825/4.039865 Å，B₀=77.991/77.963 GPa，差0.000040 Å/0.028 GPa。**該 bulk EOS 以40³為已有測試支持的基準，80³為驗證；不可再說使用者沒做收斂測試。**
- QE cutoff 測300/400/500/600/800/1000 eV（固定16³）；目前PAW/PBE維持800 eV=58.7989154810 Ry、ecutrho8倍，不能任意改常見預設。PPT未明列的high-k cutoff/PP/smearing不能補造。
- 空缺primitive4³、64→63在600eV下k1–k6的Ef為−1.113/0.735/0.945/0.648/0.573/0.615eV（PPT顯示精度），k5→k6仍約42meV。5³ cutoff400/500/600eV的Ef=.572389/.572440/.572529eV，cutoff穩定不代表k點已收斂。
- 108格點舊k3來自June16輸入，並非40³原胞校準的已驗證轉用；k4目前只是診斷／準備，不能作已收斂正式預設。任何新生產提交前，必須按原／新晶胞、倒空間取樣、PP/XC與目標量完成基準對照。不能盲目把40³塞入所有超晶胞，也不能未對照就用3³或4³。五態k4提交前先重審此點。
- DFTpy原PPT為PBE/BLPS，spacing .20/.18Å的Ef約2.9371eV；目前LDA/local、TF=.9/vW=.1之78³依九月78³/96³等獨立校準。兩套方法和能量不可混同；nano1×1×3/真空/stress尚無本PPT的收斂認證。
- 後續報告先交代既有QE cutoff/k點與DFTpy網格校準，再談空缺／奈米柱，並清楚列轉用驗證的缺口。保留既有原始數據，不把修訂說明當新DFT或倒改歷史輸入。

# QE 計算條件的來源

2026-09-25 使用者明確要求：ecut 等條件已有測試，不可任意設定或用常見預設覆蓋。

- 準備或修改 QE 計算前，讀 `docs/QE_PARAMETER_BASELINES_20260925.md`，追溯對應的既有測試與實際輸入。
- 同時核對 cutoff 單位、ecutrho、XC、偽勢 hash、晶胞定義、k 點、smearing、觀察量與鬆弛方式。PPT 的 eV 與 QE 輸入的 Ry 不可混用。
- 區分「已有測試」、「測試支持的精度／適用範圍」與「目前未找回原始輸入」。未找回文件不代表使用者從未測試；另一套偽勢／XC 的結果也不自動認證新協定。
- 保留已提交來源快照、原始輸出與歷史回執。條件修訂應有可追溯的新版本，不可倒改原始紀錄或默默重送既有作業。
- 本輪 array 為 `2151591`；查最新 Slurm 與輸出狀態後再處理，不依旧狀態重複提交。

## 2026-09-25 使用者最新更正（優先於較早的 LDA QE 選項）

- QE 維持歷史 PAW／PBE；DFTpy 維持 LDA／local recpot。兩套偽勢與 XC 本來就不同，不為了形式一致更換方法。
- 新 bulk 使用 `vc-relax`／全晶胞鬆弛、三方向 PBC。新 campaign 為 `al_bulk_vacancy_vcrelax_20260925`，與固定晶胞結果分開。
- QE 800 eV、8 倍密度 cutoff、3³ k 點來自已查到的 June16 實際 108-site vc-relax 輸入；DFTpy 78³ 來自 LDA/.9/.1 的 September15–17 網格測試。兩種 cutoff/grid 的物理意義不同，不能以名目數字相同宣稱相同精度。
- Nanocolumn 柱軸無限延伸，橫向保留 5 Å 真空；橫向晶胞不可在一般全自由 cell filter 中一起縮掉。實際生成前明記 5 Å 的幾何定義。
- DFTpy 可提交 `/work` 的 ctest（2 小時）；按使用者要求最多兩個執行、一個等待。
- 舊固定晶胞 array `2151591` 的 pending 3–7 已用 state-filter cancel；已產生資料保留。新作業仍須先查最新 queue 與提交回執，禁止重複提交。

## 2026-09-25 晚間：proposal 最高優先

- 使用者要求所有工作以完成proposal為最高優先，並定期報告正在做什麼、計算進度與分析。既定主體為bulk及晶態Al nanocolumns各自的formation/condensation，能量與結構均必須涵蓋。
- 當前變晶胞作業：DFTpy array2154593八構型已全部accepted；QE array2154594的0V已完成且inspect可用，1V執行中、六種2V排隊。此為2026-09-25約22:50台北快照，後續先查fresh status。
- 不因排隊等待而停下proposal：先整理已驗證方法、形成/結合/動能差、非仿射位移、體積和文獻/待算矩陣；9/28範圍確認稿保留pending，不能偽造完成或為趕時間放寬門檻。
- 已建立此對話10分鐘heartbeat `al-proposal`，使用目前SSH共用連線做實際狀態查詢和結果整理。使用者說15分鐘斷線指iService網頁/網頁終端；SSH查詢不是網頁保活的保證。遠端公開/etc/profile設TMOUT=900且readonly；不可偷偷更改登入或安全設定。
- 原始輸入、完成output與hash應同步本機；官方/work規則為28天未使用可清除、不備份，不以touch/改時間戳規避清理。只搬必要研究證據，不遍歷或下載全部大型scratch。

## 2026-09-26 凌晨：取消定時監控、提高 QE 並行

- 使用者明確取消每10分鐘追蹤，避免Codex用量。`al-proposal` automation 已刪除並驗證設定檔不存在；不得自動重建、改頻率後重建或繼續定時輪詢。
- 使用者要求ct56不要限制一次一組。既有QE array2154594已原地更新ArrayTaskThrottle=8；02:09台北最後查核，1V及六種2V共7組全部RUNNING，0V先前已完成。沒有重新提交或新增計算；各case仍29 ranks、64GiB、12h，immutable科學輸入未修改。
- 使用者要求先讓計算繼續，明早再查。收斂門檻本晚保留；使用者授權日後依文獻、耗時與觀察量精度判斷是否放寬，不需為例行門檻判斷重複請示，但修訂須另留可追溯協定，不能倒改已完成結果。不得以任何0.05不帶單位數字直接替換力/應力/能量門檻。
- 已核實Bhowmik2025力門檻1e-4 Ha/bohr約0.005142 eV/Å；本輪目前0.001 eV/Å較嚴。文獻短表保存於本對話outputs；0.05 eV/Å對本案幾meV結合能的適用性尚未證明。
- 本輪英文scope/calculation plan v0.4與nano程式盤點已保存於本對話outputs。nano舊prep帶PBE/GGA與舊a0預設且會執行relax，不能直接冒充LDA/.9/.1純結構生成。

## 2026-09-26 傍晚歸檔與48小時預算

- 17:22台北實際查核：兩批全部終結。DFTpy8/8accepted；QE只有0V、2NN、4NN、6NN四組usable。1V為BFGS trust-radius/history failure；1NN、3NN、5NN為QE內部11h上限。沒有新的生產提交。
- 已下載並逐檔SHA驗證534個原始研究檔（663.3 MB），334.8 MB壓縮包及分析／報告副本保存於 `C:/OFDFT/RESEARCH_UPDATES/AL_BULK_VCRELAX_20260926`。原始輸出、density、XML、source snapshot及失敗case均保留；71.70 GB波函數／mixing scratch留遠端，未刪除，省略清單可追溯。
- 正式QE雙空缺形成能：2NN1.267246、4NN1.273023、6NN1.266611 eV，採既定MP smearing functional。1V未usable，QE1V形成能及所有binding仍缺；不能以failed中間值拼出正式表。
- timeout的最後完整SCF cell/positions/energy/force/stress已按raw line匹配，見archive analysis/qe_completed_step_alignment.json。尾端accuracy及geometry末幀已屬下一步，勿從舊inspect各欄不加辨識拼成同一finalstate。
- 使用者要求ct56兩三天預算（4天較難排）。本機active bulk runner與sbatch已改48hSlurm/47hQE（169200秒），17tests+8caseverify通過；未改cutoff/XC/PP/kmesh/force或原manifest。舊snapshot仍12h/11h；新部署後才生效。嵌套protocol.max_seconds保留歷史來源，實際預算讀新attempt頂層與pw.in。
- 使用者本次主請求為查詢、歸檔與分析報告；三個timeout可準備接續輸入，但本次尚未部署或提交。原子門檻是否放寬另依已核實文獻和所需精度判斷，不能倒改失敗紀錄。

## 2026-09-26 18:28：使用者授權接續，QE已重新執行

- 最新使用者要求繼續送QE、拉長時間；已於17:53:58正式提交array **2159026**，四case順序1V/1NN/3NN/5NN，`0-3%4`、ct56、各29ranks/64GiB、Slurm48h/QE47h。18:28四組均RUNNING，elapsed34:43；三組2V首SCF完成，1V首SCF仍在迭代。2159025為test-only估算，不是production。此紀錄優先於前段「尚未提交」。
- 新campaign `al_bulk_vacancy_vcrelax_continue_20260926`；immutable snapshot suffix `20260926T095326Z`，44檔hash與本機/遠端6tests通過。新run root `/work/dawson666/qe_cases/qe_runs/AL_BULK_VACANCY_VCRELAX_CONTINUE_20260926`。查fresh狀態與submission receipt再操作，不重複提交。
- 使用最後完整SCF的cell/positions、新BFGS history與新電子初態；cutoff/XC/PP/kmesh/SCF/force/stress/nstep均保留。1V先fresh BFGS，若仍停滯再診斷，不能預先標為收斂。
- 新作業必須用 `scripts/run_bulk_vacancy_qe_continue_20260926.py inspect` 配對新manifest，原八case runner直接入口不適用；共同物理協定與幾何來源分開比對，勿強求新舊含初始幾何的science hash相同。
- DFTpy八組已收斂（每組2.4–11.9min），未重跑。使用者希望近低點切換optimizer，已查舊BFGS→固定cell原子LBFGS歷史；FIRE仍為待驗證候選。現行是ASE ionic/cell LBFGS與獨立electronic LBFGS，兩者不可混淆。若做混合需保持同一cell filter、另建optimizer狀態、保留力/stress/電子驗收。
- 定時automation仍關閉。WSL helper現可在使用者已開多個同帳號SSH master時選活躍連線，ProxyCommand=false防止multiplex失效後另開登入；未更改登入、安全或TMOUT設定。

## 2026-09-28 00:07 查核與最終歸檔

- 使用者要求確認、下載、分類及 proposal 標準報告；array2159026已全數終結、queue為空。續跑1NN/3NN/5NN通過frozen runner inspect與獨立原始輸出核查；1V再次BFGS trust-radius/history failure。合併為DFTpy8/8、QE7/8可用，沒有自動重送／修改參數／重啟監控。
- QE1V最後max force component=0.000549444791 eV/Å，比輸入0.0005高9.89%；四個完整SCF的力分量持續下降，不能稱已證實震盪或force noise。缺final relaxed-cell SCF，1V正式形成能與全部QE binding均保留null。
- 六個QE2V形成能依1NN–6NN為1.29225330547、1.2672457694、1.23977941246、1.2730231549、1.23324514228、1.2666105196 eV，採原MP smearing functional。有限週期模型QE5NN最低、OF1NN最低；5NN雙等距影像限制必須明列，不能當孤立最佳距離或凝聚動力學。
- 所有15個accepted終態從原始9/25 ideal parent lattice計算位移與局部非仿射應變，不從continuation起點計算。1595原子最近parent assignment無改變；局部kinematic應變不等於total strain或elastic dipole。
- 新續跑下載163原始檔，連同原534檔共697檔逐檔核hash；111.25GB WFC/mixing scratch未下載且未刪除，這是研究證據備份而非完整wavefunction restart備份。
- 最終101檔分類包與14頁DOCX已保存於 `C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_20260928`；工作區 `outputs/al_proposal_results_20260928` 有00_README、原始包、分析CSV/JSON、六組圖、15結構、報告、文獻及QA與重建來源。便攜ZIP `outputs/Al_vacancy_proposal_data_20260928.zip` SHA256=49766c50f85c03f9a1ff6ca8006737e0c51fdef6d9344e8f2a38e9cfb83f1159。314項合併算術核查通過。
- Proposal仍需合格QE1V、nanocolumn formation/condensation新協定結果，以及依所需結論補尺寸／kmesh／smearing誤差、elastic dipole或路徑；本報告已列具體矩陣，不能把設計當完成結果。詳見docs/BULK_QE_OF_RESULTS_20260928_ZH.md。

## 2026-09-28 00:44：使用者授權修復直到通過，已提交1V

- 使用者明確要求查原因及資料、修正並重新提交直到科學通過。正式job **2162428** 已00:42:37提交、00:43:08在cpn3032開始RUNNING；不可重複提交。新root `/work/dawson666/qe_cases/qe_runs/AL_BULK_VACANCY_1V_REPAIR_20260928`，snapshot `al_bulk_vacancy_1v_repair_20260928_20260927T164201Z`。
- 已核實停止源為很小的下降步觸發consecutive minimum trust-radius/history reset；實際遠端QE7.5鄰接source default為1e-4 bohr，文件寫1e-3，不能混稱。此次明設trust_radius_min=1e-6；另conv_thr從1e-8加嚴為1e-10 Ry。其餘物理/force/stress/energy停止條件完全保留，並非放鬆驗收。
- 只有1V，ct56，29ranks/64GiB，48hSlurm/47hQE。使用最後完整SCF幾何，fresh BFGS/electrons；不存在17組重跑或限制其他計算並行。source18檔hash、本機/遠端16tests與campaign verify通过。
- 專用 `scripts/run_bulk_vacancy_qe_1v_repair_20260928.py inspect` 包含原驗收與最終SCF force component≤0.0005eV/Å；不得用舊runner略過repair來源鏈或最終force gate。正式結果仍待新attempt完成；只有通過後才補1V形成能與QE binding。
- 提交回執、診斷文獻、實際輸入與snapshot存本對話 `outputs/qe_1v_repair_submission_20260928`，永久副本 `C:/OFDFT/RESEARCH_UPDATES/AL_QE_1V_REPAIR_20260928/submission`。定時automation仍停用；本次工作內只等待此作業完成，不重建週期Codex監控。
- 00:52使用者的共用SSH master失效；長等候程式隨連線結束，沒有取消Slurm。最後已驗證狀態00:44 RUNNING且第一SCF未完成。已用async提示請使用者重新登入；不要自動開新認證、不重送同一作業，也不能把失聯當計算失敗。
- 恢復後先執行本對話work/remote_qe_1v_repair_status_20260928.sh經remote_exec.py，finished才用已準備archive downloader取回。獨立QA為work/audit_qe_1v_repair_final_20260928.py；合併後處理在work/repair_analysis_20260928/README.md。新8/8報告builder已準備但有all16usable保護，尚未產出新報告，不能把準備文字當研究結論。
- 9/28 01:20使用者重新登入並明確要求「不要一直掛著監控，確認好就好」。本次僅單次查核：2162428 RUNNING、elapsed37:28，第一SCF第19iteration，最近accuracy約1e-7Ry、尚未達新1e-10門檻，未有BFGS failure事件。停止本回合查詢，不啟動長等候、不建立automation；後續等使用者再要求查詢。修復直到合格的授權仍在，但不得用它推定持續消耗Codex監控的授權。

## 2026-09-28：本機鬆弛軌跡檢視

- 使用者要求看 bulk vacancy relaxation／可能的塌陷過程；僅處理既有本機原始與continuation archives，未查遠端、未提交或啟動監控。新1V repair不在此資料包內。
- 本對話 `outputs/al_bulk_trajectory_viewer_20260928` 含16組803幀（787完整計算幀＋16理想參考）、OVITO multiframe extxyz（真實cell/PBC、標準id/disp）、完整108格點Wigner-Seitz reference、JSON來源hash、獨立QA及README。全幀roundtrip驗證，15個accepted末態符合既有分析；舊QE1V保留未accepted。
- 所有已記錄幀parent-site assignment未變，無多重佔據；證據支持局部鬆弛與晶胞變形，未觀察到空位合併／跨格點跳躍。不能把optimizer步數當物理時間，或以均勻晶胞收縮宣稱vacancy collapse。1NN終態max nonaffine u：QE0.07833Å，DFTpy0.15924Å。
- 對話預覽含QE/DFTpy的1NN/5NN完整287幀，支援實際尺度與去除均勻變形後局部位移×10；×10不是計算結構。未找到可啟動的OVITO，已用ASE驗證並提供匯入檔，不宣稱OVITO GUI已測試。

## 2026-09-28 09:59：QE 1V 修復已驗收

- 使用者要求查昨夜作業，2162428實際03:13:58 COMPLETED，耗時02:30:50；frozen repair inspect與獨立raw QA均通過，現DFTpy8/8、QE8/8。禁止再送重複repair。此次只查狀態與後處理，未開監控／新計算。
- 1V finalSCF accuracy8e-13Ry，max component0.000463055718eV/Å≤0.0005，norm0.000463055860≤0.001，fullstress合格；3個optimizerSCF／2BFGS步加1次finalSCF。QE1VEf=0.644962821749eV；1NN–6NN Eb meV=−2.327662,22.679874,50.146231,16.902489,56.680501,23.315124。1NN近零符號未有物理誤差界；5NN雙影像限制保留。
- 新原始archive `outputs/al_bulk_qe_1v_repair_archive_20260928`49檔hash核驗，三批合746原始檔；新完整bulk分析包 `outputs/al_proposal_results_complete_20260928`，140原數值欄位不變、987獨立核查通過；舊報告不覆寫。詳見docs/BULK_QE_OF_RESULTS_COMPLETE_20260928_ZH.md。
- 803幀舊trajectory包尚不含新增repair，仍標記舊1V失敗歷程；不可把新結果狀態倒改進舊軌跡版本。既有1NN/5NN預覽不受新1V影響。Proposal還缺nano新協定結果、有限尺寸／數值誤差和所需彈性／路徑證據。
- 新14頁報告已逐頁PNG視檢通過，完整分類包122檔已複製至 `C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_COMPLETE_20260928` 並核對全部hash；主報告為05_report/Al_vacancy_proposal_analysis_complete_20260928.docx，00_README與06scope均已更新16/16與803舊幀版本邊界。歸檔回執 `outputs/Al_vacancy_proposal_complete_20260928.receipt.json`。

## 2026-09-28 Proposal 重大修訂完成

- 使用者要求保留現有數據、重寫比較與論證，第一階段不得新增DFT。已完成18頁修訂報告、六圖、文獻／數值ledger、D12／DV／位移分布與修改紀錄；47檔歸檔 `C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_REVISION_20260928`。詳讀 `docs/PROPOSAL_REVISION_COMPARISON_20260928_ZH.md`。
- 16個合格終態及原始資料完全保留，新增QE XML／DFTpy ASE終態交叉核查全通過、8tests通過。這不等於近零Eb的總誤差已收斂。D12為OF−38.643188／QE+25.007536meV；OF最大位移parent46／60不在舊z切面，已修圖說並補第二切面。
- Phase2只提出四幾何的取樣／smearing單點篩查與資源／停止規則，Phase3按結果選擇尺寸／機械約束／彈性與代表nano。尚未提交，不把計畫當完成結果或自動啟動完整矩陣。現行無監控、無新DFT作業。
- Ho及Connétable–Maugis全文取得受限，僅用核實原出版摘要；不填未知數字。Bhowmik v4結合能來自Results p5不是TableI，Eb/Hb分開；WGC文獻不能推廣為所有OF的符號。

## 2026-09-28：18頁局部修訂 v2，後續DFT須確認

- 使用者明確要求保留18頁架構、canonical及六圖、不全面重寫、不提交新DFT。v2已完成，詳 `docs/PROPOSAL_SURGICAL_REVISION_V2_20260928_ZH.md`；永久包 `C:/OFDFT/RESEARCH_UPDATES/AL_VACANCY_PROPOSAL_REVISION_V2_20260928`。舊版原樣保留。
- 第12頁端點／輸出已對齊；第11頁四態單點篩查不能認證六構型排序或5NN最低。初始及兩方法終態18列週期影像診斷已補；5NN OF只有2.525e-5Å微小分裂，不能因此稱影像已隔離。
- Nano選擇縮準為位置形成與代表1NN配對，不承諾D12轉用。147格點方柱、近表面代表及真空網格是待確認設計，算術核查不等於已完成DFT或真空收斂。Qiu採用／排除及bulk EOS比較已補。
- 原分析與六圖逐檔hash不變；原8tests與新6tests全通過，18頁逐頁視檢。此次沒有遠端查詢／作業／監控；任何新DFT只列條件式增量，等待使用者確認，勿沿用早期修復授權自動啟動。

## 2026-09-30：Luder 回信後的資料與主張查核

- 先讀 `docs/LUDER_FEEDBACK_ACTIONS_20260930_ZH.md`。Luder要求提供數據、從實際PPT第5頁驗證可疑QE、縮準主張；主線推進不同尺寸nanocolumn的單空缺能量與結構。B0對λ/μ及bulk 1V的B0為bonus；nano雙空缺待單空缺／表面分析清楚後再選。
- 新查核：原始QE XML的1NN/3NN/5NN最長最短晶胞向量差為4.4016%/4.5065%/4.3500%，前版只列等體積a_eq隱藏此形變。這是原始計算特徵，物理或數值原因未確定；不能以先前runner驗收取代k點/smearing/尺寸驗證，也不能斷言是續跑bug。局部位移已扣掉均勻晶胞形變，不能據此說QE全部結構變化較小。
- QE smearing項代入binding差式為約31–49meV，僅作敏感性診斷，不是誤差條／零溫修正。原16態canonical、六圖、18頁DOCX保留；實際寄出17頁PPT另修成20頁，永久包 `C:/OFDFT/RESEARCH_UPDATES/AL_LUDER_FEEDBACK_20260930`。可分享ZIP約16.6MB含原始證據及可攜核算。370項核算與重跑既有14tests通過，不能混稱新物理精度已驗證。
- `proposed_QE_checks`備妥1NN/3NN/5NN各3³/4³/5³固定幾何SCF，先建議六個3³/4³，5³為條件式延伸。保留PAW/PBE、800eV、8倍密度cutoff、MP0.02Ry，統一conv_thr1e-10Ry。只檢查三態能量差、力與全應力；沒有匹配0V/1V時不產新Ef/Eb，不認證完整六配對排序。尚未提交，需使用者確認，不以早期repair授權自動提交。
- 使用者回憶nano建模有Monte Carlo/Voronoi，未接受新方柱半寬4/6/8Å提議。已找到本機及GitHub `app/ase_nanocrystal.py`的radius clipping與Wulff-style週期稜柱，未建立Monte Carlo／Voronoi實作證據。兩者可能是生成或有效體積估算，勿猜作既定方法。舊builder vacuum是每側留白，參數5意味名目總間隙10Å；與先前候選總gap5Å不同。正式nano計算前需統一幾何／vacuum及尺寸定義。
- 本次SSH共用連線查核不可用，已提示一次重連；未自動登入、未開監控、未提交新DFT。網頁登入不等於共享SSH可用。不要為未收到回覆而反覆輪詢。


## 2026-09-30 23:06：新授權後已提交 QE 驗證與 nano 1V

- 使用者新授權先查可疑 QE 再提交可推進計算，並要求多半徑、柱軸無限延伸；優先於上段「未提交／等待確認」。詳讀 docs/LUDER_VALIDATION_NANO_SUBMISSION_20260930_ZH.md。
- 現場重新核對八組 accepted QE 的24檔，遠端／本機hash一致；4.4%方向形變仍需驗證，未確認是物理或數值原因。原16态canonical與六圖保留。
- **QE array2173492** 已提交ct56，48h/47h，0-14%15：全八態k4固定幾何SCF＋1NN/3NN/5NN的k3對照，共11；另外四組為[111]圓切奈米柱96/176格點、0V/內部1V的z-only vc-relax。800eV、PAW/PBE、8xrho、MP.02未改。此非完整物理收斂矩陣。
- **DFTpy array2173494** 為96格點0V/1V，ctest2h、兩組並行；**2173496** 為176格點0V/1V，ct56 24h。LDA/local/.9/.1，<=.15Å起始間距的固定FFT點數，只放zz自由度。17個ct56 task待QOSGrpJobsLimit；23:06两個ctest已電子迭代。後續查fresh狀態，不依此紀錄重送。
- 快照al_luder_validation_nano_20260930_20260930T150143Z；QE root /work/dawson666/qe_cases/qe_runs/AL_LUDER_VALIDATION_NANO_20260930；OF root /work/dawson666/dftpy_project/relax/AL_NANO_SINGLE_VACANCY_20260930。新runner run_luder_qe_20260930.py（inspect按manifest index）與run_nano_dftpy_pilot_20260930.py，不適用舊bulk runner入口。
- 已另生成半徑約4/6/8/10/12Å五尺寸，QE父格點38/96/176/278/380；campaigns/al_nanocolumn_radius_gallery_20260930有兩方法0V/1V共20結構及圖。4/10/12Å尚未送DFT。舊builder build_periodic_prism已核實；Monte Carlo／Voronoi用途仍未確認。
- 三方向計算PBC但柱軸[111]連續、橫向真空；vacuum=5明定每側padding5Å、名目影像gap10Å，橫向盒子不得鬆弛。兩方法同無因次parent、各自a0。6/8Å的內部空缺離數學軸约.825Å、原CN12，不稱恰在軸心。真空/軸長/k/grid仍待收斂；nano應力是含真空盒歸一化。
- 本機／遠端各7tests、20結構讀回及PBC拓撲核查通過；不代表新DFT結果完成。無定時監控、沒有自動更改登入逾時；收尾後等使用者要求再查。

## 2026-10-01：供使用者自行上傳 NAS 的完整本機分類包

- 詳讀 `docs/NAS_SHARE_INDEX_20261001_ZH.md`。主資料夾 `C:/OFDFT/NAS_SHARE/Al_vacancy_Luder_20261001`；同名ZIP約514MB，資料夾約1.19GB。入口START_HERE.html有中英文說明及相對連結。沒有自動上傳NAS或發信。
- 共1273檔，1253複製來源hash一致；746原始檔與歷史manifest一致；搬移後370核算、21相對連結及ZIP全部payload檢查通過。ZIP SHA256=f84c755de189ebd09f94ede6b3b23d1d6c5265dbe5b80105d2567c735df51761。原canonical及六图與原始檔未改。
- 00_INDEX索引分清16個正式終態、21次成功／失敗／續跑attempt、20個初始nano模型、19個新提交task。報告、bulk、完整本機raw history、終態與legacy軌跡、nano初始結構、新提交、方法QA各自分類。
- 本次僅本機整理，沒有遠端查詢。新作業仍只含9/30 23:06舊狀態與輸入回執，未收錄新結果；不要把資料包狀態當fresh queue。舊803幀QE1V為修復前失敗，正式1V另索引。未下載WFC/mixing未含在包內。

## 2026-10-01 晚間：完成結果查核，發現取樣／應力／資源問題

- 先讀 docs/LUDER_RESULTS_AUDIT_20261001_ZH.md。21時左右實查：QE2173492前11個SCF全部accepted diagnostic；DFTpy2173494/2173496四組結束。4個QE nano仍RUNNING但約17h無輸出，29ranks實剩28/25/18/27、cgroup峰值近64GiB，QE估72–156GB。缺失rank確切死因未取到，OOMkill counter0，不得虛稱直接OOM證據。本次已問是否取消四組，寫入時未收到回答、尚未取消或重送。後續先查新訊息與fresh queue。
- QE k3對照重現舊能量<0.000053meV，但同舊幾何k4的Ef1從.644963到.679875eV、5NN Eb從+56.681到−181.997meV，D12從+25.008到+287.686meV。全部新差式用匹配k4參考。缺陷力.030–.046eV/Å、部分stress>1GPa；不能當k4 relaxed端點或收斂排序。舊5NN偏好與用舊QE證明OF機制的主張暫緩，原canonical不改。
- OF nano的兩半徑5.921/7.895Å、96/176父格點，bulk reservoir Ef暫.598151/.631826eV；終態無parent換位。原σyy高達−178613GPa拒用。保存rho後處理定位vW的∇ρ∇ρ/ρ真空數值放大，與其離散能量strain導數不一致；sqrt(rho)診斷可重現能量導數，沒有新SCF。診斷修正對zz影響<.00033GPa但非全force/stress驗證；energy/structure僅pilot，vacuum/grid/period及QE comparator仍待驗證，不倒改原result.json。
- 新309檔NAS增量 `C:/OFDFT/NAS_SHARE/Al_vacancy_Luder_update_20261001_evening`，同名ZIP92.42MB；永久 `C:/OFDFT/RESEARCH_UPDATES/AL_LUDER_RESULTS_20261001`。START_HERE.html中文、REPORT_EN.html英文；279raw hashes、348分析核查、27連結與ZIP逐檔通過。ZIP SHA256 fb008e36f984e86f48455261890bc0d4a66682b29efe86644abe558de12653bf。舊514MB包與canonical/六圖保留；此次無新DFT、無監控、無NAS上傳或發信。

## 2026-10-01 22:03：明確授權取消空轉並重送，已執行

- 先讀 docs/QE_NANO_MEMORY_RECOVERY_20261001_ZH.md。使用者授權確認空转後砍掉重來；20秒實測CPU~99.6%卻全I/O與輸出不變、缺MPI ranks，job-root failcnt大量觸限，21:53取消舊2173492_11–14並保留全部輸出。此優先於上段待取消確認。
- 新array **2180070** index0–1（R15）128GiB、**2180073** index2–3（R20）256GiB，均ct56/29ranks/1node/48h（QE47h），各%2可同時四組。22:03仍PENDING/Priority，不等排隊、不監控、不重送。
- 新root `/work/dawson666/qe_cases/qe_runs/AL_NANO_QE_MEMORY_RECOVERY_20261001`；snapshot `al_nano_qe_memory_recovery_20261001_20261001T135638Z`。用新runner run_nano_qe_memory_recovery_20261001.py與新campaign inspect，不混原array index。四份pw.in原bytes/hash完全不變，新增actual-cgroup/Slurm-memory guard及srun --kill-on-bad-exit=1 --oom-kill-step=1 --wait=60。正式本機/遠端各5test通過；沒有改cutoff/PP/XC/收斂/初始結構。
- 前100行available RAM屬節點，不是job額度；QE估72/156GB總量實在L367–531。現場ct56有>=384564MiB節點，支持新allocation。記憶體限制不足已證實，未取得每個缺rank直接kernel kill證據，勿虛稱所有OOMkill已確證。
- OF vW stress候選只留本對話outputs/qe_nano_recovery_20261001/proposed_fix，未接production。2 synthetic+4真實rho共36tensor finite-strain checks通過(max1.242e-9GPa)，沒有SCF；full-total-stress/force、grid/vacuum仍待驗證。這次只重送原4個QE nano，沒增加bulk/OF矩陣。
- 本輪105檔處置包 `C:/OFDFT/NAS_SHARE/Al_vacancy_recovery_20261001`，REPORT_ZH.html詳細中文、SUMMARY_EN.txt英文、sources.json十份官方／原始來源；同名ZIP4.80MB，SHA256=092f0b86ae893ba0130eb86ad7c03200229a841bbb685200d53644647ac8514b。永久副本 `C:/OFDFT/RESEARCH_UPDATES/AL_QE_NANO_RECOVERY_20261001`。檔案hash/ZIP及13相對連結通過；22:03最後狀態仍排隊。先前92MB／514MB包不覆寫。

## 2026-10-02 13:04：四組 QE nano 單次執行查核

- 使用者要求確認是否空轉。透過既有SSH單次有界查核；未新增監控／DFT，未取消或改參數。2180070_0/1及2180073_2/3均RUNNING，10/2 00:05:47開始，已跑約13小時。
- 25秒節點快照各29/29 pw.x且PID集合不變；各job與step memory.failcnt=0。歷史記憶體峰值依R15_0V、R15_1V、R20_0V、R20_1V為80.45/79.90/162.95/162.20GiB，上限128/128/256/256GiB。目前未觸限，不保證後續最大需求。
- 13:01至13:04電子iteration累計895→899、1114→1120、384→384、269→271。R20_0V同一次SCF新增降低本徵值門檻後重新對角化輸出，29 ranks I/O均改變；四組有實際計算進展，無上次缺rank空轉跡象。
- 完整收斂SCF／力區塊40/48/10/5；最近原子最大力norm為0.005879/0.004985/0.096024/0.246572 eV/Å，均仍高於0.001。未有正式終態；SCF區塊數不是BFGS步數或完成百分比。
- 證據在本對話outputs/qe_nano_livecheck_20261002的REPORT_ZH.txt、summary.json、log_evidence.json、probe.json及hash清單。初次即時力解析混入verbose分項力，已移除錯誤衍生欄位，改取Forces acting on atoms下首個連續nat列總力，驗證ID為1..nat；正式result/canonical/production runner未修改。後續取summary.json與log_evidence.json的力。

## 2026-10-03 16:40：小柱QE完成，綜合Luder疑點檢討

- 使用者要求分析完成結果、綜整全部數據、找出suspicious。已單次查核、下載及本機分析，未新增DFT／監控、未取消或改現役参数。兩個R15已完成，兩個R20仍RUNNING；不要用舊「四個都跑」状态。
- R15_0V／1V於10/2 19:10:42／19:43:22台灣時間完成，耗19:04:55／19:37:35；frozen inspect及獨立XML／文字驗證通過。fmax=0.000479706／0.000541242 eV/Å，force component=0.000420891／0.000485204；最後晶胞SCF、電子誤差及zz應力均通過。這是數值鬆弛驗收，nano k/vacuum/period/smearing尚未收斂認證。
- R15 QE Ef=0.747117974 eV，用同方法舊k3 bulk μ=−537.458611320 eV/atom；僅換k4 bulk μ得到0.744864054（−2.254meV），只作原子庫敏感性，不認證柱子k點。OF R15=0.598150643、R20=0.631826232 eV仍pilot。QE R20尚未有終態，禁止用中間能量產正式Ef。
- R15 1V相對0V軸長QE+0.010333307Å（+0.0759068%），OF−0.019724634Å（−0.1470069%）。已新增逐parent ID配對、映射0V至1V晶胞再扣均勻平移的局部位移：QE R15 max0.1343347/rms0.0414741Å，OF R15 max0.1450316/rms0.0525246Å，OF R20 max0.0718520/rms0.0197611Å。不能混同各自理想初態到終態最大位移；後者QE R15 0V/1V為0.332857/0.390013Å，含表面鬆弛。
- 16:40 R20_0V/1V已完成34/21次SCF與力計算，fmax0.0070818/0.0304303 eV/Å，29 MPI完整、memory failcnt0、峰值162.99/162.26GiB於256GiB配额。Slurm EndTime=10/4 00:05:47；QE max_seconds169200按已印CPU約剩6.43h，非精確牆鐘預報。不因仍執行或即將到時就當空轉；本輪保留繼續跑，沒有延長或新續算提交。
- bulk k3→k4的疑點仍未修復；新nano不能恢復舊5NN偏好或OF因果說法。Luder沒點名數字，9項檢討屬我方查核發現。兩方法PP/XC不同，差異不能只歸因TFvW。
- 新NAS包 C:/OFDFT/NAS_SHARE/Al_vacancy_review_20261003，中文REPORT_ZH.html、SUMMARY_EN.html英文、35筆總表（16舊bulk+11診斷+4OF nano+4QE nano，其中2running）、四新圖、六終態與三配對位移。原canonical／六圖不改。87新raw hashes、308來源/解析/結構核查、80舊raw選取副本、22連結檢查通過；非精度收斂測試。完整歷史及OF density另在先前NAS包。未下載約7.09GB QE scratch，不是完整restart備份。
- 工作 outputs/luder_review_20261003；永久 C:/OFDFT/RESEARCH_UPDATES/AL_LUDER_REVIEW_20261003。初版ZIP300檔27,834,233bytes，SHA623b1d13d13ccea973d947e536ce07d10a02b8a1ea74385fd20dd7324eb8ad66；若後續加QA需以delivery_receipt最新為準。沒有NAS上傳或發信。
- 同回合最後完成獨立目錄可攜重建：308核查再通過，23個表／圖／結構hash完全一致。QA記錄已加入NAS包；最終ZIP301檔、27,835,011bytes，SHA256=353365407e9f482719ea8523b69477a925bbf3119e61989ba144334b749eede4（優先於上段初版hash）。

## 2026-10-03：核對使用者貼文，準備五態 k4（尚未提交）

- 已核對 a442416c 附件意見；採用「理想0V→鬆弛0V／鬆弛0V→鬆弛1V／形成能」三層比較，不能當已證實因果鏈。配對位移扣除完美柱基準但保留空缺引起的表面反應。R15最大位移只差7.96%、RMS差26.64%、p95差27.54%，不推翻另一系統bulk1NN的原最大位移數字；不能以bulk最大值直接解釋錯誤結合。
- 本次沒有遠端查詢、提交、取消、監控或實際DFT。現役R20狀態仍只有上段16:40快照，後續需要使用者要求時再做單次查核。原canonical、六圖、10/3分析及NAS包不改。
- 準備 `campaigns/al_bulk_k4_vcrelax_review_20261003` 五態0V/1V/1NN/2NN/5NN，所有起始幾何直接沿用已歸檔k4固定幾何SCF輸入，且逐原子對上獨立解碼的k3終態。三方向PBC，vc-relax全晶胞；PAW/PBE、800eV、8倍rho cutoff、MP0.02Ry、k4無位移、conv_thr1e-10、原force/energy/stress门檻保留。
- 五態明設trust_radius_min=1e-6（成功1Vrepair的optimizer選項），fresh BFGS/electrons，不能稱相對所有舊k3協定只改k。資源提案為ct56每組29ranks/64GiB、Slurm72h/QE255600s，五組並行；現有k4印出>36.58–37.04GB下界，64GiB不是最大需求保證。尚無sbatch及新runner；提交前需適配campaign/provenance與全應力驗收，不能直接用nano axial-only inspector或舊campaign-ID runner。
- 五態僅支持指定Ef/Eb/D12及5NN−1NN比較，不認證六構型排序；k5必須由新的合格k4終態生成，目前沒有k4新終態或k5假輸入。3NN、nano convergence、stress production validation仍為有條件增量。
- 本機117項來源/參數/ASE結構核查通過，NAS副本獨立重驗117通過；這是準備查核，不是科學收斂驗證。58檔新包 `C:/OFDFT/NAS_SHARE/Al_bulk_k4_preparation_20261003`，入口REVIEW_ZH.html，完整README_ZH.txt，英文LUDER_UPDATE_DRAFT_EN.txt，inputs與protocol/manifest。永久副本 `C:/OFDFT/RESEARCH_UPDATES/AL_BULK_K4_PREPARATION_20261003`；本對話outputs/bulk_k4_preparation_20261003。
- 新ZIP3,537,816bytes，SHA256=36722f6f0794b261900803172e0dd26de31e81700a69f7c57976fb91091623e2；13相對連結、複本及ZIP內hash通過。沒有NAS上傳或發信。逐檔回執outputs/bulk_k4_preparation_20261003_delivery_receipt.json。

## 2026-10-04 00:19：R20 QE 為內部時間上限正常停止，尚未合格

- 使用者重新登入並要求確認計算停止是否撞時間牆。透過既有 SSH 做一次狀態查核及資料下載，queue 空。沒有新 DFT、取消、參數部署或監控；此前五態 bulk k4 仍是 review-only。
- R15 2180070_0/1 仍 accepted，能量/力與之前一致。R20 2180073_2/3 在10/3 23:06:19／23:06:03台灣時間停，耗47:00:32／47:00:16；兩份pw.out明示 Maximum CPU time exceeded、max_seconds=169200、JOB DONE。QE step exit0/COMPLETED、stderr空；外層runner因未通過驗收exit1，故job FAILED。沒有被48h Slurm TIMEOUT硬切，也沒有新增OOM/MPI故障證據。
- R20_0V/1V完整SCF/力步數39／26（上次34／21）。最後完整步fmax0.00481501357／0.01937995077eV/Å，分量0.00362163837／0.01824582292，σzzQE +0.00042656150／−0.00439994719GPa；仍無合格final relaxed-cell SCF。兩組各多完成5個SCF，並非空轉，不能把末幾步非單調當已證實低點震盪。
- XML output已屬下一個未完成SCF，SCF/optimizer convergence_achieved均false；能量/力/幾何不可跨步拼接。已從step區塊取最後完整幾何，交叉核對所有65步stdout能量/連續總力及ASE幾何；首步stdout axes精度有限，幾何容差1e-5Å，其餘1e-7Å，XML為正式幾何來源。R20正式Ef與半徑趨勢維持缺值。
- 遠端每R20有29份wfc、29mix以及density/BFGS/update/restart_scf/XML，僅檢查存在與大小，未實際讀回restart。依QE官方node20正常max_seconds退出可走restart；本機04_continuation_review兩份輸入只改restart_mode=restart與max_seconds=255600，提案同29ranks/1node/256GiB、Slurm72h。未部署或提交、無sbatch；後續需另建attempt複製完整遠端scratch及相同PP/binary，避免覆寫歷史。完整restart成功與否仍需實際執行確認。
- 213原始檔逐hash下載，原始tar16,950,659bytes；約24.27GB省略大檔仍在遠端（含四組nano），未刪。本機652項來源/步數/幾何/輸入核查通過，不等於科學收斂。
- 新235檔包 `C:/OFDFT/NAS_SHARE/Al_nano_stopcheck_20261004`，入口REPORT_ZH.html或REPORT_ZH.txt；永久 `C:/OFDFT/RESEARCH_UPDATES/AL_NANO_STOPCHECK_20261004`，本對話outputs/qe_nano_stopcheck_20261004。ZIP33,254,704bytes、SHA256=69e159cb89013c9213b311371cc5db38d1de75dcc63f15fb4b92668c342dccab；副本與ZIP逐檔核驗、四個相對連結通過。無NAS上傳。Windows偏好已改為default app/Explorer，勿再自動開Codex側欄。

## 2026-10-04：續跑因額度不足被拒，完成全盤分析與使用者授權清理

- 使用者明確要求立即續跑R20並刪除舊報告／重複結果、完整分類。本輪已完成新campaign `al_nano_qe_restart_20261004`、專用runner/sbatch及本機／遠端各5項測試；只有restart_mode與時間預算改變，保留電子/BFGS歷史、29ranks、256GiB、兩組並行、Slurm72h/QE71h。
- snapshot `/work/dawson666/src/campaign_snapshots/al_nano_qe_restart_20261004_20261003T164119Z`；root `/work/dawson666/qe_cases/qe_runs/AL_NANO_QE_RESTART_20261004`。test-only因mst114175錢包餘額−42.1093被拒，尚未呼叫production sbatch，無新job ID。使用者已答覆先向Luder申請經費、目前先整理分析；不得重試提交或建監控。恢復額度後先驗證既有部署/回執/queue再提交，不重跑會衝突的部署腳本。
- 唯一現行資料包：`C:/OFDFT/NAS_SHARE/Al_vacancy_CURRENT`；入口`00_START_HERE.html`，英文`LUDER_SUMMARY_EN.txt`，資料取用`DATA_GUIDE_ZH.txt`。35筆現行總表中R20兩組已標clean stop/unaccepted，不再寫running。bulk canonical16筆及原六圖不變。
- 重新計算49項形成能／結合能／體積等算術核查通過，保留既有308/652來源核查。現行報告涵蓋bulk六構型與k敏感性、mono體積、nano能量／配對結構、EOS／文獻比較與逐項待驗證工作。R15 QE−OF形成能差148.967meV，軸長反應相反；這仍不能證實OF過度塌陷，bulk舊QE排序仍未驗證，OF nano stress仍為pilot限制。
- 按使用者最新清理授權，舊outputs、研究更新AL_*、NAS Al_*及舊work研究檔已整併，實際移除7519檔／7,718,269,649bytes。新完整資料包約611.6MB，保守淨減少約7.11GB（檔案內容總量，非磁碟配置空間測量）。使用者Downloads／附件／OneDrive、研究專案程式與遠端scratch均未刪。
- 此最新整理授權優先於較早「保留每個完整舊資料夾／ZIP副本」記錄。原始唯一科學證據保留一次，過時派生報告/PPT/渲染暫存刪除；舊ZIP/TAR拆解後只保留唯一內容。舊路徑改由`04_RAW/evidence_index.json`查回，`evidence_catalog.tsv`可檢索，`08_CODE/extract_evidence.py`可按SHA提取並驗證。不要因舊AGENTS路徑失效就認定原始數據遺失。
- 刪除後1718份保留內容全部重新核SHA通過，15個本地HTML連結/錨點通過，提取與拒絕覆寫通過。清理發現3個node_modules目錄junction，已排除23388個外部檔案參照並僅移除本地連結；共享runtime完整。逐檔刪除與存證在07_AUDIT，沒有NAS上傳或寄信。

## 2026-10-05：使用者要求 NAS 只放 QE／DFTpy 數據與結構

- 使用者中途明確收斂需求：不要分析圖、報告、簡報或文獻，只要分類清楚的計算數據與結構。上傳用純資料夾改為 `C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA`；不要再指示上傳含報告的 Al_vacancy_CURRENT。後者仍為本機完整歷史證據與分析主包，未刪除。
- 純資料包35組／690檔／746,557,060bytes。QE23組：8 bulk k3鬆弛、11 k點單點、2 R15鬆弛、2 R20未收斂。DFTpy12組：8 bulk鬆弛、4 nano pilot。各類有獨立目錄；00_NUMERICAL_TABLES保留既有数值表，case_index.json逐組說明狀態。
- 從已核實的canonical選定attempt提取原始輸入、stdout、result、XML、PP、已下載density、初終態及軌跡；原始bytes/hash不變。QE initial_from_pw_input.extxyz只作座標轉換；R20最後完整幾何明名NOT_RELAXED，不混下一個未收斂XML output。file_manifest.json列689個payload hash與原來源，逐份寫入讀回核對通過。
- 已驗證35組核心輸出/結構齊全，資料包不含PDF/PPT/DOCX/HTML/圖片/程式；只附短DATA_INDEX.txt及索引。無新增遠端查詢／計算／監控、無NAS上傳／寄信。遠端wavefunction/mixing scratch不在純資料包，這不是完整restart備份。舊失敗attempt仍保留於本機master evidence archive，未混入所選35組。
- Windows Explorer已啟動選取純資料夾，使用者自行上傳NAS。資料分類重建腳本位於本對話work/build_data_only_20261005.py，收據outputs/data_only_20261005。

## 2026-10-06：Luder要求報告、數據、產生報告的腳本同放指定NAS目錄

- 使用者最新明確授權補回報告與腳本，優先於10/5「只要純數據」範圍。沿用 `C:/OFDFT/NAS_SHARE/Al_QE_DFTpy_DATA` 加 REPORTS/、SCRIPTS/及00_READ_FIRST.txt；不複製第二套整包。現約752.53MB（新增約6MB），原35組計算資料不變。
- 指定NAS目的地 `/volume1/OFDFT/workitem_divacancy_formation_energy_20260929`。在瀏覽器實測使用者提供的連結，回報net::ERR_ADDRESS_UNREACHABLE；未上傳、未寄信，不能聲稱檔案已在指定目錄。不把URL中的session token寫入交付文件。
- REPORTS/20260929_bulk_progress_AS_SENT.pptx保留Downloads原17頁簡報。找回的build_deck.mjs是20頁底稿，不能宣稱逐位元重建使用者後來修改的17頁。README明列此限制；原始產圖腳本、canonical/數值表、文獻數值來源與底稿程式均附。
- REPORTS/current_review是10/4科學快照；只改交付相對連結與資料夾說明。SCRIPTS/reproduce_reports.py可在全新output重建現行HTML與原六圖；不跑DFT／網路／scheduler。本輪在獨立目錄實際重建，49項算術核對、六圖figure_data完全相等、14本地連結通過。WSL版本numpy2.4.6、matplotlib3.11.1、ase3.29.0記於requirements。
- 原689檔核hash通過；僅DATA_INDEX.txt更新提及報告／腳本，file_manifest有previous_sha256。其餘報告補件93檔列report_sources_manifest.json；原35組計算payload未修改。回信草稿應說prepared而不是uploaded，請Luder提供可用連結／確認目錄存取。

## 2026-10-10 11:47：經費恢復，R20兩組續跑已正式開始

- 使用者明確表示Luder已加457額度、SSH登入完成，要求續跑。既有master pid3383連線可用；queue空、無任何先前restart job／submission receipt。排程實報mst114175餘額457.8907，test-only通過。2221956是test-only，production為 **array2221957 index0–1%2**。
- 11:45:27兩組同時RUNNING：2221957_0＝nano_R20_0V（實際SlurmJobID2221958，cpn3023）；2221957_1＝nano_R20_1V（實際JobID2221957，cpn3178）。每組ct56、29ranks、256GiB、Slurm72h/QE255600秒。Slurm EndTime 10/13 11:45:27台灣時間。不得重複提交。
- 沿用已凍結snapshot `/work/dawson666/src/campaign_snapshots/al_nano_qe_restart_20261004_20261003T164119Z`，新root `/work/dawson666/qe_cases/qe_runs/AL_NANO_QE_RESTART_20261004`。11個source hash、QE binary及parent metadata核對通過；只有restart_mode／時間預算相對parent不同，原物理條件／停止門檻均保留。
- 原5tests會在複製readonly snapshot後因fixture無寫入權限出錯；當時尚未呼叫sbatch。改在獨立臨時副本將fixture設可寫後5tests全通過，凍結來源未修改。完整初次錯誤及修正後回執皆保留。
- 各121個scratch檔（8,945,697,413／8,897,463,457bytes）已在compute node完整copy且兩端SHA通過；原parent未覆寫。11:46:39／41 QE已launch，cgroup確實256GiB。11:47兩份stdout明示讀入舊晶胞、原子位置與charge density；尚在初始化、新SCF能量數0，stderr空、無Error in routine。不宣稱已有新的收斂步或已確認所有電子/BFGS歷史讀取完成。
- 本次僅有界啟動查核，沒有定時automation／長期poll。DFTpy無未完成作業，未重跑；五態bulk k4仍review-only，不因資金恢復就自動擴矩陣。
- 回執／查核原始JSON／STATUS_ZH在本對話outputs/qe_R20_resume_20261010；存證副本Al_vacancy_CURRENT/06_PENDING_COMPUTATIONS/R20_restart/submission_20261010。分享包另加RUN_STATUS_20261010.txt，不修改舊35組原始結果；沒有NAS上傳。
