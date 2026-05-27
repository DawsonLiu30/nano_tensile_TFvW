# FINAL PPT Chinese Speaker Notes

PPT file:

```text
C:\Users\dawso\Desktop\FINAL_NUS_UPLOAD_20260526\00_FINAL_REPORT\FINAL_qe_bulk_vacancy_20260526.pptx
```

Use these notes as the detailed Chinese oral script for professor discussion. The tone should be careful, factual, and defensive: separate what is completed, what is numerically converged, and what is still pending.

## Slide 1: Work Hypothesis and Background

### 中文講稿

這一頁我想先說明這份報告的目的。這次的重點不是只報一個 vacancy formation energy 數字，而是先把整個 benchmark workflow 做乾淨，讓後面 QE 和 DFTpy 的比較有可靠基礎。

我目前把問題分成兩層。第一層是數值收斂問題，也就是 QE 裡面的 energy cutoff、k-point density，還有 DFTpy 裡面的 real-space grid spacing 和 supercell size / vacancy concentration。第二層才是方法本身的差異，也就是 Kohn-Sham DFT 的 QE 和 orbital-free DFT 的 DFTpy / TFvW 對 vacancy defect energy 的描述能力是否不同。

這個分法很重要，因為如果數值參數還沒收斂，我們不能直接說 DFTpy 方法本身有問題。反過來，如果 spacing、cutoff、cell size 這些都檢查過了，差異還很大，那才比較能討論 method limitation。

文獻參考部分，我使用 Gillan 1989 作為主要 benchmark。Gillan 對 aluminium vacancy formation energy 的計算值大約是 0.56 eV，文中引用的 experimental reference 大約是 0.66 eV。這兩個數字提供了一個物理合理範圍。因此，我現在不是只看 QE 和 DFTpy 哪個數字比較漂亮，而是檢查我們的 QE dense-k 結果是否落在這個文獻和實驗支持的範圍內，再用它作為 DFTpy 比較的 reference。

另外，之前 DFTpy 結構有被 VESTA 檢查出問題，所以這份報告也特別強調：舊的 primitive/slanted cell 結果已經撤回，不再當成 final conclusion。現在的 comparison 是基於 VESTA-checked conventional fcc cell 重新建立的。

### 這頁要講出的核心句

「這份報告的主軸是先建立可靠的 QE benchmark，再用 VESTA-checked conventional cell 重新評估 DFTpy。重點是把 numerical convergence 和 method difference 分開。」

### 教授可能問，建議回答

如果教授問為什麼要引用 Gillan：

可以回答，因為 Gillan 1989 直接研究 aluminium vacancy formation energy，而且提供 calculated value 和 experimental reference，所以很適合當作本研究 QE vacancy benchmark 的文獻錨點。

如果教授問 DFTpy 差很多是不是一開始就知道：

可以回答，我們不能先假設差異是 method limitation，所以這次先重建 conventional-cell benchmark，確認不是舊 cell geometry 或 grid spacing 沒收斂造成的。

## Slide 2: Computational / Technical Details

### 中文講稿

這一頁說明所有計算設定，主要分成三部分：QE bulk EOS、QE vacancy cell、DFTpy conventional-cell rerun。

首先是 QE bulk EOS。bulk EOS 使用 primitive fcc Al cell，一個 primitive cell 裡面有一顆 atom。這裡做 9 個 lattice constant 的 total-energy scan，a0 從 4.000 到 4.080 Angstrom。之後用 EOS fitting 得到 equilibrium lattice constant a0 和 bulk modulus B0。energy cutoff scan 做 300 到 1000 eV，k-point 則做 high-k check，一路補到 80×80×80。這一部分的目的，是先確認 QE bulk reference 本身是穩定的。

第二部分是 QE vacancy supercell。這裡我改用 conventional fcc cell，因為 conventional cell 在 VESTA 裡是直角 cell，比 primitive slanted cell 更直觀，也比較不會在結構檢查時造成誤解。conventional fcc cell 有 4 顆 atom，我用 2×2×4 repetition，所以 pristine cell 是 64 atoms，移除中心 atom 後 vacancy cell 是 63 atoms。這個 cell 的長度是 8.0797 × 8.0797 × 16.1594 Angstrom，三個角都是 90 度，volume 是 1054.909 Angstrom cubed。這些結構在報告前都要用 VESTA 檢查。

第三部分是 DFTpy。這裡要特別小心講：舊的 primitive/slanted DFTpy vacancy calibration 已經撤回，因為 VESTA inspection 顯示結構不適合作為 final benchmark。現在新的 DFTpy 比較分成兩條線。第一條是 direct same-cell comparison，使用和 QE 一樣的 conventional fcc 2×2×4 cell，也就是 64 到 63 atoms。第二條是 size / concentration scan，使用 conventional cubic fcc n×n×n cells，例如 2×2×2、3×3×3、4×4×4、5×5×5。這條不是跟 QE 的 2×2×4 aspect ratio 完全一樣，而是用來隔離 vacancy concentration 對 DFTpy 結果的影響。

最後，relaxation force 要分清楚 target 和 achieved。DFTpy same-cell spacing scan 的 actual final fmax 已經小於 0.002 eV per Angstrom。但是 QE 那邊目前只能說 relaxation target 是 0.002 eV per Angstrom，strict final-force verification 還要根據 output 仔細確認，不能直接講已完全達標。

### 這頁要講出的核心句

「QE vacancy 用 VESTA-checked conventional 2×2×4 cell 做 direct benchmark；DFTpy 則分成 same-cell comparison 和 conventional cubic size/concentration scan，兩者用途不同，不能混在一起。」

### 教授可能問，建議回答

如果教授問為什麼不用 primitive cell：

可以回答，primitive cell 在數學上不是錯，但在這次 benchmark 和 VESTA inspection 裡容易造成 visual ambiguity。為了避免結構被誤解，這次用 conventional fcc orthogonal cell 做教授可直接檢查的 benchmark。

如果教授問 4×4×4 cubic 和 2×2×4 same-cell 有什麼不同：

可以回答，2×2×4 是跟 QE 直接比較的 same-cell benchmark；cubic n×n×n 是用來看 vacancy concentration / size dependence，不是直接取代 2×2×4 comparison。

## Slide 3: Results: Bulk EOS Convergence

### 中文講稿

這一頁是 bulk EOS convergence。bulk 部分目前是最穩的一塊，因為我們已經對 energy cutoff 和 high k-point density 都做過檢查。

左邊圖是 B0 對 energy cutoff 的收斂。從 300 到 1000 eV，B0 在 400 eV 以上變化非常小，代表 cutoff 對 bulk modulus 的影響已經很小。這說明 600 eV 作為後續 QE vacancy benchmark 的 cutoff 是合理的。

右邊圖是 high-k mesh convergence。這裡補到 20、24、30、40、60、80 的 kmesh。最重要的比較是 40×40×40 和 80×80×80。40×40×40 得到 a0 = 4.039825 Angstrom，B0 = 77.991 GPa；80×80×80 得到 a0 = 4.039865 Angstrom，B0 = 77.963 GPa。兩者的差異只有 0.000040 Angstrom 和 0.028 GPa。

這個差異非常小，所以我把 40×40×40 當作 practical QE bulk reference，而 80×80×80 是 validation check。也就是說，不是我們只跑到 40×40×40 就停，而是我們有用 80×80×80 證明 40×40×40 已經足夠。

下面的表也把實驗值放進來。實驗的 aluminium lattice constant 大約是 4.05 Angstrom，bulk modulus 大約是 76 GPa。我們 QE 的 a0 大約 4.0398 Angstrom，B0 大約 78 GPa，跟實驗是合理接近的。

### 這頁要講出的核心句

「bulk EOS reference 已經很穩，40×40×40 和 80×80×80 的 B0 只差 0.028 GPa，所以 bulk reference 不是目前主要 uncertainty。」

### 教授可能問，建議回答

如果教授問 EOS 曲線是不是亂 fit：

可以回答，bulk EOS 不是只畫一條 arbitrary curve，而是用 9 個不同 lattice constant 的 total energy points 做 EOS fitting，從 equilibrium volume 附近的曲率得到 B0。圖中的 B0 convergence 是每組 EOS fit 後得到的 physical bulk modulus，不是單純二次曲線係數。

如果教授問為什麼 B0 比實驗略高：

可以回答，DFT/PBE pseudopotential、0 K 計算和實驗 finite-temperature 條件不同都會造成差異，但目前數字在合理範圍內，而且收斂性已經確認。

## Slide 4: Results: QE Vacancy Convergence with VESTA-Checked Conventional Cell

### 中文講稿

這一頁是目前 QE vacancy benchmark 的核心。這裡所有結果都已改成 VESTA-checked conventional fcc 2×2×4 cell，不再使用之前容易誤解的 primitive/slanted cell。

左上角先列出結構：pristine 是 64 atoms，vacancy 是 63 atoms，vacancy 放在中心位置，也就是 fractional coordinate 大約在 (0.5, 0.5, 0.5)。cell 是 8.0797 × 8.0797 × 16.1594 Angstrom，角度是 90 度。這裡要強調，這個 cell 已經用 VESTA 檢查過，所以現在的結構定義是可以直接給教授檢查的。

vacancy formation energy 的公式是：

E_f^vac = E_vac^(N-1) - ((N-1)/N) E_perfect^N。

這個公式的意思是，perfect cell 有 N 顆 atom，vacancy cell 有 N-1 顆 atom，所以要把 perfect cell 的 total energy 依照 atom number scale 到 N-1 顆 atom，再跟 vacancy cell 比較。這樣才是在同一個 chemical potential reference 下比較形成一個 vacancy 的能量。

右邊是 cutoff convergence。固定 kmesh = 2×2×2 時，300 到 800 eV 的 E_f 從 1.0588 到 1.0611 eV，cutoff 收斂非常穩。但是這組不能當 final reference，因為 2×2×2 k-point 明顯 under-sampled，數值偏高。

下面 dense-k cutoff check 固定在 5×5×5。400 到 800 eV 的結果從 0.600951 到 0.601226 eV，只差大約 0.000275 eV。這說明 cutoff 本身不是主要問題；真正主導 vacancy energy 的是 k-point sampling。

左下表是 kmesh scan。1×1×1 是 gamma-only，0.678 eV，但 gamma-only 對金屬 vacancy cell 不能當 final。2×2×2 是 1.061 eV，明顯 high，代表 under-sampling。3×3×3 是 0.724，4×4×4 是 0.544，5×5×5 是 0.601，6×6×6 是 0.678。這個序列不是單調收斂，而是 residual metallic k-point oscillation。

所以這頁最重要的結論不是「QE final = 某一個數字」，而是：QE dense-k values 落在 literature-consistent range，大約 0.544 到 0.678 eV，接近 Gillan 0.56 eV 和 experiment 0.66 eV，但仍然有 metallic k-point oscillation。因此，我目前不把 QE vacancy formation energy 宣稱為 single fully converged final number，而是把它當作 literature-consistent dense-k reference range。

另外 force wording 要小心。這裡寫的是 relaxation target fmax < 0.002 eV per Angstrom，不代表所有 QE output 都已經逐一確認 actual final force 達標。QE strict final-force verification 目前仍然 pending，這點不能講太滿。

### 這頁要講出的核心句

「QE cutoff 已經很穩，但 k-point 還有金屬震盪；6×6×6 已完成，得到 0.677782 eV，所以我報告 QE dense-k reference range，而不是單一 final number。」

### 教授可能問，建議回答

如果教授問為什麼 5×5×5 和 6×6×6 差這麼多：

可以回答，這是 metallic defect supercell 常見的 k-point / Fermi-surface sampling oscillation。vacancy formation energy 是兩個很大的 total energy 相減，小的 Fermi-surface sampling 差異會被放大到 formation energy 裡。這也是為什麼我不宣稱單一 fully converged final value，而是報 dense-k literature-consistent range。

如果教授問 2×2×2 cutoff convergence 有什麼意義：

可以回答，它證明 cutoff 對 vacancy energy 的影響很小，但因為 k-point under-sampling，所以 2×2×2 的 absolute value 不能作 final reference。

如果教授問為什麼 6×6×6 接近 experiment：

可以回答，6×6×6 的 0.6778 eV 接近 experimental reference 0.66 eV，而 4×4×4 的 0.5438 eV 接近 Gillan 的 0.56 eV，整個 dense-k range 與文獻範圍一致。

## Slide 5: Results: DFTpy Conventional Vacancy Rerun

### 中文講稿

這一頁是 DFTpy conventional rerun 的結果。這是回應之前 VESTA 檢查問題最重要的一頁。

首先要承認，舊的 primitive/slanted DFTpy cell 已經撤回，不再拿來當 final result。現在 DFTpy 使用 VESTA-checked conventional fcc cell 重新跑。

DFTpy 這裡用的是 TFvW kinetic energy density functional、BLPS local Al recpot，以及 PBE-GGA exchange-correlation。跟 QE 不同，DFTpy 是 orbital-free DFT，不做 QE 那種 Brillouin-zone k-point convergence。DFTpy 的主要 numerical convergence parameter 是 real-space grid spacing，或等價的 grid ecut analogue。

這裡有兩條 DFTpy 分析。第一條是 same-cell spacing scan，也就是跟 QE 一樣的 conventional fcc 2×2×4 cell，64 到 63 atoms。這組是 direct QE-DFTpy comparison。結果是 spacing 從 0.30 到 0.16 Angstrom，vacancy formation energy 都在大約 2.901 eV，而且 actual final fmax 都小於 0.002 eV per Angstrom。因此 DFTpy same-cell spacing convergence 是完成的。

第二條是 size / concentration scan，用 conventional cubic fcc n×n×n cells。這條不是 same aspect ratio comparison，而是用來看 vacancy concentration effect。accepted 的點是 conv02 到 conv05，conv06 目前不 accepted，因為它沒有達到 fmax target，所以不能放進正式趨勢。

表格中也列出 DFTpy-QE difference 是大約 2.22 到 2.30 eV。這個範圍是用 QE 5×5×5 和 6×6×6 作為 reference endpoints 來算的。即使用比較高的 QE 6×6×6 值 0.6778 eV，DFTpy 2.9008 eV 還是高出大約 2.22 eV。所以 DFTpy 高 vacancy energy 不是因為 QE 5×5×5 剛好偏低，也不是單純因為舊 primitive/slanted geometry。

這裡要小心措辭：我不說 DFTpy 結果「錯」，而是說在目前 TFvW / local pseudopotential setup 下，DFTpy 對 localized vacancy defect energy 給出顯著高於 QE 和文獻的結果。這比較科學，也比較防守。

### 這頁要講出的核心句

「DFTpy 舊 cell 已撤回；新 conventional same-cell rerun 仍然穩定在約 2.901 eV，而且 actual fmax 達標。因此高 DFTpy vacancy energy 不是單純舊結構 artifact。」

### 教授可能問，建議回答

如果教授問 DFTpy 不是之前結構錯才 2.9 eV 嗎：

可以回答，之前的結果確實撤回了，所以我重新用 VESTA-checked conventional same-cell 跑。新結果仍然是約 2.901 eV，而且 spacing convergence 和 fmax 都通過，所以舊結構不是唯一原因。

如果教授問 DFTpy size scan 有沒有考慮 vacancy concentration：

可以回答，有。size scan 使用 conventional cubic fcc n×n×n supercells，accepted range 從 32 atoms 到 500 atoms，vacancy concentration 從 3.125% 降到 0.2%。E_f 從 2.938 降到 2.884 eV，濃度有影響，但只有大約 0.054 eV，遠小於 DFTpy-QE 的 2.22 到 2.30 eV 差距。

如果教授問 conv06 為什麼不用：

可以回答，conv06 的 actual fmax 沒有達到 0.002 eV per Angstrom，所以不作為 accepted trend point，只列為 attempted but not accepted。

## Slide 6: Interpretation / Discussion

### 中文講稿

這一頁是把目前所有結果收斂成結論。

第一，bulk convergence 是已經可以防守的。QE bulk EOS 用 high-k mesh 補到 80×80×80，證明 40×40×40 作為 practical bulk reference 是足夠的。bulk a0 和 B0 也跟實驗值合理接近，所以 bulk reference 不是目前主要 uncertainty。

第二，QE vacancy structure 已經從舊的 ambiguous structure 改成 VESTA-checked conventional fcc 2×2×4 cell。這一點非常重要，因為現在教授可以直接用 VESTA 打開結構檢查，不會再看到 slanted primitive cell 被誤判成錯誤結構。

第三，QE vacancy formation energy 的 cutoff convergence 很穩，但是 k-point scan 仍然有 residual metallic oscillation。4×4×4 到 6×6×6 的 E_f 大約是 0.544 到 0.678 eV，其中 6×6×6 是 0.677782 eV。這個範圍跟 Gillan 0.56 eV 和 experiment 0.66 eV 是一致的，但是還不能說收斂成單一數字。

第四，force wording 要嚴謹。對 DFTpy，我們已經有 actual final fmax summary，same-cell spacing scan 都小於 0.002 eV per Angstrom。對 QE，我們目前只能說 relaxation target 是 0.002 eV per Angstrom，strict final-force verification 還需要逐一 output 確認，所以不能在口頭或 PPT 裡說 QE 已達成 fmax < 0.002，除非後續真的從 output 驗證。

第五，DFTpy conventional same-cell spacing scan 已經完成，converged value 大約 2.901 eV。這個數值明顯高於 QE dense-k range。用 QE 5×5×5 比，差距大約 2.30 eV；用 QE 6×6×6 比，差距大約 2.22 eV。這個差距遠大於 QE k-point oscillation，所以 DFTpy-QE discrepancy 是 robust 的。

最後，下一步是 nanocrystal tensile pilot。這裡要特別定義清楚，nanocolumn 和 nanocrystal 都是 z-direction periodic infinite structures，不是 finite length。nanocolumn 是 circular xy cross section，nanocrystal 是 polygonal xy cross section，例如 hexagon 或 triangle。後續 tensile 要加入 vacancy concentration，並測 inner、middle、outer 三個 vacancy positions。第一輪只做 pilot，先生成三個 VESTA-checked structures，再進行 tensile。

### 這頁要講出的核心句

「目前可以防守的結論是：bulk reference 穩、QE vacancy dense-k range 與文獻一致但仍震盪、DFTpy conventional same-cell rerun 仍高約 2.901 eV，因此 DFTpy-QE discrepancy 不是舊結構 artifact 單獨造成的。」

### 教授可能問，建議回答

如果教授問現在到底 final value 是多少：

可以回答，QE vacancy 我目前不報單一 final value，而是報 literature-consistent dense-k range，大約 0.544 到 0.678 eV。因為 5×5×5 到 6×6×6 還有約 0.077 eV 的 metallic oscillation。

如果教授問 DFTpy 是否還值得做：

可以回答，值得，因為現在我們已經證明 DFTpy numerical spacing 和 fmax 可以收斂，但它對 localized vacancy defect energy 的 prediction 明顯不同。這正是後續研究 OFDFT functional limitation 或 vacancy/nanostructure size effect 的重點。

如果教授問下一步為什麼要 inner/middle/outer vacancy：

可以回答，nanocrystal / nanocolumn 裡 vacancy 不只是 concentration 問題，還有 spatial position effect。inner vacancy 比較像 bulk-like defect，outer vacancy 會受到 surface 和 local reconstruction 影響，middle vacancy 介於兩者之間。三個位置可以把 vacancy concentration 和 vacancy position effect 分開看。

## Short Closing Script

如果教授要我用一分鐘總結，我會這樣講：

這次我先把舊的 primitive/slanted vacancy-cell workflow 撤回，改用 VESTA-checked conventional fcc cell。QE bulk convergence 已經很穩，40×40×40 和 80×80×80 的 bulk modulus 差只有 0.028 GPa。QE vacancy 的 cutoff convergence 也很穩，但 k-point scan 仍然有 metallic oscillation；4×4×4 到 6×6×6 給出大約 0.544 到 0.678 eV，跟 Gillan 和實驗範圍一致。DFTpy 重新用 conventional same-cell 跑完後，spacing convergence 和 actual fmax 都達標，但 vacancy formation energy 仍然約 2.901 eV，比 QE 高約 2.22 到 2.30 eV。因此這個差異不是舊 cell geometry artifact 單獨造成的。下一步我會把 nanocrystal tensile 做成 z-periodic infinite polygonal-cross-section nanocolumn，加入 vacancy concentration，並測 inner/middle/outer vacancy positions。
