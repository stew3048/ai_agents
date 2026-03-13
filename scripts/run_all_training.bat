@echo off
REM 批次執行所有訓練腳本（依序執行，使用 CPU）
REM Windows 批次檔版本

REM 設定強制使用 CPU（透過環境變數）
set CUDA_VISIBLE_DEVICES=

REM 記錄開始時間
set START_TIME=%date% %time%
echo ============================================================
echo   開始執行所有訓練腳本
echo   開始時間: %START_TIME%
echo   使用設備: CPU
echo ============================================================
echo.

REM 訓練腳本列表
set SCRIPTS=train_in_domain.py train_dino_mlp.py train_dino_simple_cnn.py train_dino_fpn.py train_dino_hybrid.py

REM 記錄檔案
set LOG_FILE=outputs\training_batch_log.txt
if not exist outputs mkdir outputs

REM 清空或建立記錄檔案
echo 訓練批次執行記錄 > "%LOG_FILE%"
echo 開始時間: %START_TIME% >> "%LOG_FILE%"
echo 使用設備: CPU >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM 依序執行每個訓練腳本
for %%s in (%SCRIPTS%) do (
    set SCRIPT_START=%date% %time%
    echo ============================================================
    echo   執行: %%s
    echo   開始時間: %SCRIPT_START%
    echo ============================================================
    echo.
    
    REM 記錄到 log 檔案
    echo --- >> "%LOG_FILE%"
    echo 腳本: %%s >> "%LOG_FILE%"
    echo 開始時間: %SCRIPT_START% >> "%LOG_FILE%"
    
    REM 執行訓練腳本（使用 CPU）
    python scripts\%%s --device cpu >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% EQU 0 (
        set SCRIPT_END=%date% %time%
        echo.
        echo   [成功] %%s 執行成功
        echo   結束時間: %SCRIPT_END%
        echo 結束時間: %SCRIPT_END% >> "%LOG_FILE%"
        echo 狀態: 成功 >> "%LOG_FILE%"
    ) else (
        set SCRIPT_END=%date% %time%
        echo.
        echo   [失敗] %%s 執行失敗（繼續執行下一個）
        echo   結束時間: %SCRIPT_END%
        echo 結束時間: %SCRIPT_END% >> "%LOG_FILE%"
        echo 狀態: 失敗 >> "%LOG_FILE%"
    )
    
    echo. >> "%LOG_FILE%"
    echo.
)

REM 記錄結束時間
set END_TIME=%date% %time%
echo ============================================================
echo   所有訓練腳本執行完成
echo   結束時間: %END_TIME%
echo ============================================================
echo.
echo 結束時間: %END_TIME% >> "%LOG_FILE%"

pause
