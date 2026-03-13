#!/bin/bash
# 批次執行所有訓練腳本（依序執行，使用 CPU）

# 設定強制使用 CPU
export CUDA_VISIBLE_DEVICES=""

# 記錄開始時間
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "============================================================"
echo "  開始執行所有訓練腳本"
echo "  開始時間: $START_TIME"
echo "  使用設備: CPU"
echo "============================================================"
echo ""

# 訓練腳本列表
TRAIN_SCRIPTS=(
    "train_in_domain.py"
    "train_dino_mlp.py"
    "train_dino_simple_cnn.py"
    "train_dino_fpn.py"
    "train_dino_hybrid.py"
)

# 記錄檔案
LOG_FILE="outputs/training_batch_log.txt"
mkdir -p outputs

# 清空或建立記錄檔案
echo "訓練批次執行記錄" > "$LOG_FILE"
echo "開始時間: $START_TIME" >> "$LOG_FILE"
echo "使用設備: CPU" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 依序執行每個訓練腳本
for script in "${TRAIN_SCRIPTS[@]}"; do
    SCRIPT_START=$(date +"%Y-%m-%d %H:%M:%S")
    echo "============================================================"
    echo "  執行: $script"
    echo "  開始時間: $SCRIPT_START"
    echo "============================================================"
    echo ""
    
    # 記錄到 log 檔案
    echo "---" >> "$LOG_FILE"
    echo "腳本: $script" >> "$LOG_FILE"
    echo "開始時間: $SCRIPT_START" >> "$LOG_FILE"
    
    # 執行訓練腳本（使用 CPU）
    if python scripts/"$script" --device cpu 2>&1 | tee -a "$LOG_FILE"; then
        SCRIPT_END=$(date +"%Y-%m-%d %H:%M:%S")
        echo ""
        echo "  ✓ $script 執行成功"
        echo "  結束時間: $SCRIPT_END"
        echo "結束時間: $SCRIPT_END" >> "$LOG_FILE"
        echo "狀態: 成功" >> "$LOG_FILE"
    else
        SCRIPT_END=$(date +"%Y-%m-%d %H:%M:%S")
        echo ""
        echo "  ✗ $script 執行失敗（繼續執行下一個）"
        echo "  結束時間: $SCRIPT_END"
        echo "結束時間: $SCRIPT_END" >> "$LOG_FILE"
        echo "狀態: 失敗" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    echo ""
done

# 記錄結束時間
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "============================================================"
echo "  所有訓練腳本執行完成"
echo "  結束時間: $END_TIME"
echo "============================================================"
echo ""
echo "結束時間: $END_TIME" >> "$LOG_FILE"
echo "總執行時間: $(($(date +%s) - $(date -d "$START_TIME" +%s))) 秒" >> "$LOG_FILE"
