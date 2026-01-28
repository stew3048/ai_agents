"""
正確轉換 CSV 檔案格式

從 big5 編碼的 tab 分隔檔案正確轉換為 UTF-8-sig 編碼的逗號分隔 CSV

用法:
  python scripts/properly_convert_csv.py
"""

import os
import sys
import csv
import shutil

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def properly_convert_csv(backup_csv, output_csv):
    """正確轉換 CSV"""
    print(f"讀取備份檔案: {backup_csv}")
    
    rows = []
    fieldnames = None
    
    # 從備份檔案讀取（big5 編碼，tab 分隔）
    with open(backup_csv, 'r', encoding='big5') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        
        for row in reader:
            rows.append(row)
    
    print(f"  ✓ 讀取了 {len(rows)} 筆資料")
    print(f"  ✓ 欄位數: {len(fieldnames)}")
    
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
            # 檢查幾個關鍵案例
            test_cases = [
                ('10870', '98', '建築邊緣誤判'),
                ('4795', '11', '建築物頂部'),
                ('10066', '50', '山'),
                ('3888', '429', '海面'),
            ]
            
            print(f"  ✓ 成功讀取，總共 {len(test_rows)} 筆")
            
            for camera_id, image_id, keyword in test_cases:
                row = [r for r in test_rows if r['camera_id'] == camera_id and r['image_id'] == image_id]
                if row:
                    reason = row[0].get('failure_reason', '')
                    if keyword in reason:
                        print(f"  ✓ Camera {camera_id}, Image {image_id}: 編碼正確")
                    else:
                        print(f"  ⚠ Camera {camera_id}, Image {image_id}: 編碼可能有問題")
                        print(f"    failure_reason: {reason[:60]}")
                else:
                    print(f"  - Camera {camera_id}, Image {image_id}: 未找到")


def main():
    backup_csv = 'outputs/diagnostic_with_failure_modes.csv.backup'
    output_csv = 'outputs/diagnostic_with_failure_modes.csv'
    
    print("=" * 60)
    print("  Properly Convert CSV Format")
    print("=" * 60)
    print()
    
    if not os.path.exists(backup_csv):
        print(f"備份檔案不存在: {backup_csv}")
        return
    
    # 再次備份現有檔案
    if os.path.exists(output_csv):
        backup_current = output_csv + '.before_proper_convert'
        shutil.copy2(output_csv, backup_current)
        print(f"備份現有檔案: {backup_current}")
    
    print()
    properly_convert_csv(backup_csv, output_csv)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)
    print(f"\n輸出檔案: {output_csv}")
    print("現在應該可以在 Excel 中正確打開，所有欄位都會對齊！")


if __name__ == '__main__':
    main()
