"""
修正 diagnostic_with_failure_modes.csv 的格式

將 tab 分隔符改為逗號，編碼改為 UTF-8-sig（Excel 可正確打開）

用法:
  python scripts/fix_csv_format.py
"""

import os
import sys
import csv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def fix_csv_format(input_csv, output_csv):
    """修正 CSV 格式"""
    print(f"讀取: {input_csv}")
    
    # 讀取原始 CSV（tab 分隔，big5 編碼）
    rows = []
    with open(input_csv, 'r', encoding='big5') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        
        for row in reader:
            rows.append(row)
    
    print(f"  讀取了 {len(rows)} 筆資料")
    print(f"  欄位數: {len(fieldnames)}")
    
    # 寫入標準 CSV（逗號分隔，UTF-8-sig 編碼）
    print(f"\n寫入: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  完成！已轉換為標準 CSV 格式（逗號分隔，UTF-8-sig 編碼）")
    
    # 驗證
    print(f"\n驗證輸出檔案...")
    with open(output_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        test_row = next(reader, None)
        if test_row:
            print(f"  ✓ 成功讀取，欄位數: {len(test_row)}")
            print(f"  ✓ 範例 camera_id: {test_row.get('camera_id', 'N/A')}")
            print(f"  ✓ 範例 failure_mode: {test_row.get('failure_mode', 'N/A')}")
            print(f"  ✓ 範例 failure_reason: {test_row.get('failure_reason', 'N/A')[:50]}")


def main():
    input_csv = 'outputs/diagnostic_with_failure_modes.csv'
    output_csv = 'outputs/diagnostic_with_failure_modes.csv'  # 覆蓋原檔案
    
    print("=" * 60)
    print("  Fix CSV Format")
    print("=" * 60)
    print()
    
    # 先備份
    backup_csv = input_csv + '.backup'
    print(f"備份原檔案: {backup_csv}")
    import shutil
    shutil.copy2(input_csv, backup_csv)
    print(f"  ✓ 備份完成")
    
    print()
    fix_csv_format(input_csv, output_csv)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)
    print(f"\n備份檔案: {backup_csv}")
    print(f"修正後檔案: {output_csv}")


if __name__ == '__main__':
    main()
