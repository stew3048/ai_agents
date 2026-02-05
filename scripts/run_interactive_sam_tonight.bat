@echo off
chcp 65001 >nul
REM 今晚互動 SAM：先試遮擋一張、再試天海一線一張
REM 請在專案根目錄執行，或 cd 到 ai_projects 後執行 scripts\run_interactive_sam_tonight.bat

cd /d "%~dp0.."
if not exist .venv\Scripts\activate.bat (
    echo [ERROR] 找不到 .venv，請在專案根目錄執行
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo ===== 步驟 1：先測試「只顯示圖片」是否正常 =====
echo 若出現視窗且看到圖片，表示路徑與視窗沒問題。關閉視窗後會繼續跑 SAM。
echo 若沒有圖或報錯，請把此視窗內的錯誤訊息複製下來。
echo.
python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9 --no-sam
if errorlevel 1 (
    echo [ERROR] 上述有錯誤。若看不到圖片，請把錯誤訊息複製給開發者。
    pause
    exit /b 1
)

echo.
echo ===== 第 1 張：heavy-occlusion (4795/9) 含 SAM =====
echo 在圖上「只點天空」處（避開建築頂），按 P 預測，按 Q 結束
echo.
python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9 --model_type sam2_hiera_small --device cpu
if errorlevel 1 (
    echo [ERROR] 執行時出錯，請把上方錯誤訊息複製下來。
    pause
    exit /b 1
)

echo.
echo ===== 第 2 張：sea-sky (3888/429) =====
echo 在「天空區」點幾點（盡量不點到海），按 P 預測，按 Q 結束
echo.
python scripts/sam2_interactive_one_image.py --camera_id 3888 --image_id 429 --model_type sam2_hiera_small --device cpu
if errorlevel 1 (
    echo [ERROR] 執行時出錯。
    pause
    exit /b 1
)

echo.
echo 完成。Overlay 存於 outputs\sam2_interactive\
pause
