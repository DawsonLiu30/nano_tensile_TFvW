# 本機與 GitHub 同步

本機 repository：`C:/OFDFT/nano_tensile_TFvW`。
遠端：`https://github.com/DawsonLiu30/nano_tensile_TFvW.git`。
目前研究分支：`codex/portable-qe-20260923`。`main` 尚未整併本分支。

Windows Git Credential Manager 已於 2026-10-21 使用 DawsonLiu30 完成登入，遠端讀取與 push 試跑通過。憑證由 credential manager 管理，不存進 repository。這項登入與 Codex GitHub 連接器分開。

## 從 GitHub 更新

先確認工作目錄沒有未保存的變更：

```powershell
cd C:/OFDFT/nano_tensile_TFvW
git status
git fetch origin
git pull --ff-only
```

有本機修改時，先選取需要保存的檔案 commit。若 `--ff-only` 因分支分歧拒絕更新，先比較兩端提交再決定如何整合，不用強制覆寫處理。

## 把本機研究更新推上 GitHub

```powershell
git status
git diff
# 使用實際要保存的檔案路徑；先確認輸入、版本與證據完整。
git add <file1> <file2>
git diff --cached --stat
git commit -m "Describe the research update"
git push
```

Push 傳送已提交的內容；未 commit、被忽略的檔案及 repository 外的資料不會一起上傳。同步為手動操作，沒有背景自動上傳。

## 驗證此次保存的版本

```powershell
python scripts/verify_research_snapshot.py
```

此工具驗證 2026-10-21 發佈快照的檔案 SHA-256 與匯入檔案；不執行 QE／DFTpy。日後刻意修改普通文件或程式可能使該歷史快照檢查失敗，應以對應 commit 驗證；不可為消除失敗而改寫凍結來源或其歷史 manifest。
