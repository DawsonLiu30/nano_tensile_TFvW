R15 k7 最小診斷包（2026-10-17，尚未提交）

只使用已存在的兩個 conditional k7 SCF：
  index8 nano_R15_0V_k7_SCF_20261015，96 原子
  index9 nano_R15_1V_k7_SCF_20261015，95 原子

父 runner 原本明確拒絕 conditional_k7 的 run。這次保留該 guard，
另加經審查、只准8/9的入口；沒有修改原 snapshot、runner、input或prefix。
新入口直接呼叫原 verify、inspect、memory allocation guard、srun command。

重新審查的原因：k5相對k3同幾何的 Δ(E1−E0)=+36.291321936 meV，
超過5 meV操作門檻，力與軸應力也未通過。因此補k7判斷密度敏感度，
尚不把k5或k7宣稱最終收斂設定。原gate仍失敗，六新尺寸仍被阻擋。

新k7和已完成k5的輸入逐字比較，只有：
  1. K_POINTS 1 1 5 改成1 1 7，offset仍0 0 0。
  2. 原先已準備好的k7 prefix、case、output身份。
原子座標、晶胞、nat、nbnd、PP、PAW/PBE、800eV、MP .02Ry、
SCF 1e-10Ry、電子混合、from_scratch初始化與71h內停完全相同。

資源沿父k7 manifest原值：256GiB、29MPI ranks、72h、array8-9%2。
k5輸出打印RAM下限0V>72.44GB、1V>71.99GB，於192GiB配置正常完成；
這些是下限而非記憶體最高值。256GiB沿原預備配置，非新測得必要用量。

文件：
  run_reviewed_k7.py：新短入口，原科學檢查不重寫。
  submit_k7.sbatch：僅提交上述兩個array tasks的配置。
  review_manifest.json：原52檔hash、k5 gate/output證據、審查決定、新入口hash。
  prepare_review.py：由原凍結資料生成審查manifest。
  test_reviewed_k7.py / review_QA.json：12項程式測試全過；沒有跑DFT。

verify：
python -B run_reviewed_k7.py verify --parent-snapshot <10/15原snapshot> --root <原QE runroot>

遠端同一純程式測試：
python -B test_reviewed_k7.py --parent-snapshot <10/15原snapshot> --raw-root <原QE runroot>

正式run由sbatch給環境 REVIEW_SNAPSHOT / PARENT_SNAPSHOT / QE_RUN_ROOT，
只有root審查部署後提交。每個k7 case的attempt_001必須不存在；
沿新case鎖與mkdir保護，舊k3/k5不動；全scratch保留供需要時續算。

SCF accepted只代表該固定幾何診斷的電子與資料驗收通過。
最終力、最大分量與sigma_zz另列geometry_force_and_stress_targets_met，
高力固定幾何診斷不可稱為幾何收斂。後續比較k5→k7配對能差、力與應力後，
才決定必要的重新鬆弛設定；本包不自動執行後續矩陣。
