# 本機環境檢查（torch DLL / Anaconda）

## 檢查結果

- **Visual C++**：系統已有 `vcruntime140.dll`（在 `C:\Windows\System32` 與 Python 目錄），**不一定需要再裝** VC++ Redistributable。DLL 錯誤（1114）有時是「載入順序或 PATH 不同」導致，換終端可能就正常。
- **Anaconda**：本機有安裝 **Anaconda3**（路徑：`C:\Users\yiching\anaconda3`）。可改用 **Anaconda Prompt** 跑專案 .venv，有時 torch 會較容易載入成功。
- **torch**：專案 `.venv` 裡有 `c10.dll`，檔案存在，問題是「載入時」失敗。

## 建議：用 Anaconda Prompt 跑互動 SAM

1. 從開始功能表開啟 **「Anaconda Prompt」**（或 「Anaconda3 (64-bit)」底下的 Anaconda Prompt）。
2. 在 Anaconda Prompt 裡依序輸入：

```bat
cd /d c:\Users\yiching\ai_projects
.venv\Scripts\activate
python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9
```

3. 若成功，會跳出視窗、可點選後按 P 跑預測。若仍出現 DLL 錯誤，把錯誤訊息貼給開發者即可。

（用專案的 .venv 而不是 conda 的 base，是因為 sam2 是裝在 .venv 裡。）
