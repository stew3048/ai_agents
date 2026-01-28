"""
修正 diagnostic_with_failure_modes.csv 的格式（正確處理編碼）

從 big5 編碼的 tab 分隔檔案轉換為 UTF-8-sig 編碼的逗號分隔 CSV

用法:
  python scripts/fix_csv_format_v2.py
"""

import os
import sys
import csv
import shutil

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def fix_csv_format(input_csv, output_csv):
    """修正 CSV 格式"""
    print(f"讀取: {input_csv}")
    
    # 讀取原始 CSV（tab 分隔，big5 編碼）
    rows = []
    fieldnames = None
    
    try:
        with open(input_csv, 'r', encoding='big5') as f:
            reader = csv.DictReader(f, delimiter='\t')
            fieldnames = reader.fieldnames
            
            for row in reader:
                rows.append(row)
        
        print(f"  ✓ 使用 big5 編碼讀取成功")
    except Exception as e:
        print(f"  ✗ big5 編碼讀取失敗: {e}")
        # 嘗試其他編碼
        for encoding in ['utf-8-sig', 'utf-8', 'gb2312']:
            try:
                with open(input_csv, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    fieldnames = reader.fieldnames
                    rows = list(reader)
                print(f"  ✓ 使用 {encoding} 編碼讀取成功")
                break
            except:
                continue
    
    if not rows:
        print("  ✗ 無法讀取檔案")
        return False
    
    print(f"  讀取了 {len(rows)} 筆資料")
    print(f"  欄位數: {len(fieldnames)}")
    
    # 寫入標準 CSV（逗號分隔，UTF-8-sig 編碼）
    print(f"\n寫入: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✓ 完成！已轉換為標準 CSV 格式（逗號分隔，UTF-8-sig 編碼）")
    
    # 驗證
    print(f"\n驗證輸出檔案...")
    with open(output_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        test_rows = list(reader)
        if test_rows:
            test_row = test_rows[0]
            print(f"  ✓ 成功讀取，欄位數: {len(test_row)}")
            print(f"  ✓ 範例 camera_id: {test_row.get('camera_id', 'N/A')}")
            print(f"  ✓ 範例 failure_mode: {test_row.get('failure_mode', 'N/A')}")
            reason = test_row.get('failure_reason', 'N/A')
            print(f"  ✓ 範例 failure_reason: {reason[:60] if reason else 'N/A'}")
            
            # 檢查是否有亂碼
            if 'å»º' in reason or 'ç¯' in reason:
                print(f"  ⚠ 警告：failure_reason 中可能仍有編碼問題")
            else:
                print(f"  ✓ 編碼看起來正常")
    
    return True


def main():
    input_csv = 'outputs/diagnostic_with_failure_modes.csv.backup'  # 從備份讀取
    output_csv = 'outputs/diagnostic_with_failure_modes.csv'
    
    print("=" * 60)
    print("  Fix CSV Format (Correct Encoding)")
    print("=" * 60)
    print()
    
    if not os.path.exists(input_csv):
        print(f"備份檔案不存在: {input_csv}")
        print("嘗試直接讀取原檔案...")
        input_csv = 'outputs/diagnostic_with_failure_modes.csv'
    
    if not os.path.exists(input_csv):
        print(f"檔案不存在: {input_csv}")
        return
    
    # 再次備份
    backup_csv2 = input_csv + '.backup2'
    if os.path.exists(output_csv):
        shutil.copy2(output_csv, backup_csv2)
        print(f"備份現有檔案: {backup_csv2}")
    
    print()
    success = fix_csv_format(input_csv, output_csv)
    
    print("\n" + "=" * 60)
    if success:
        print("  ✓ 完成")
    else:
        print("  ✗ 失敗")
    print("=" * 60)


if __name__ == '__main__':
    main()
