"""
修復 CSV 檔案中的編碼問題

檢查並修復 failure_reason 欄位中的亂碼

用法:
  python scripts/fix_csv_encoding_issues.py
"""

import os
import sys
import csv
import re

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def has_garbled_text(text):
    """檢查文字是否包含亂碼"""
    if not text:
        return False
    # 檢查是否包含常見的亂碼模式
    garbled_patterns = [
        r'å»º',  # 常見的 UTF-8 亂碼模式
        r'ç¯',
        r'é',
        r'ç·£',
        r'èª¤',
        r'å¤',
        r'é',
        r'å',
    ]
    for pattern in garbled_patterns:
        if re.search(pattern, text):
            return True
    return False


def try_fix_encoding(text, source_encoding='big5'):
    """嘗試修復編碼"""
    if not text or not has_garbled_text(text):
        return text
    
    # 如果文字看起來像 UTF-8 亂碼，嘗試從 big5 解碼
    try:
        # 先將文字編碼為 bytes（假設當前是錯誤的 UTF-8）
        # 然後用 big5 解碼
        if isinstance(text, str):
            # 嘗試不同的修復方法
            # 方法1: 假設文字是 UTF-8 編碼的 big5 內容
            try:
                text_bytes = text.encode('latin1')  # 先轉為 bytes
                fixed = text_bytes.decode('big5')
                return fixed
            except:
                pass
            
            # 方法2: 直接嘗試 big5 解碼（如果已經是 bytes）
            try:
                if isinstance(text, bytes):
                    fixed = text.decode('big5')
                    return fixed
            except:
                pass
    except:
        pass
    
    return text


def fix_csv_encoding(input_csv, output_csv):
    """修復 CSV 編碼問題"""
    print(f"讀取: {input_csv}")
    
    rows = []
    garbled_count = 0
    
    # 讀取當前檔案（UTF-8-sig）
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            failure_reason = row.get('failure_reason', '')
            
            # 檢查是否有亂碼
            if has_garbled_text(failure_reason):
                garbled_count += 1
                # 嘗試修復
                fixed_reason = try_fix_encoding(failure_reason)
                if fixed_reason != failure_reason:
                    row['failure_reason'] = fixed_reason
                    print(f"  修復: camera_id={row.get('camera_id')}, image_id={row.get('image_id')}")
                    print(f"    原: {failure_reason[:50]}")
                    print(f"    新: {fixed_reason[:50]}")
            
            rows.append(row)
    
    print(f"\n  總共 {len(rows)} 筆資料")
    print(f"  發現 {garbled_count} 筆有亂碼")
    
    # 如果備份檔案存在，嘗試從備份修復
    backup_csv = input_csv + '.backup'
    if os.path.exists(backup_csv) and garbled_count > 0:
        print(f"\n嘗試從備份檔案修復: {backup_csv}")
        
        # 讀取備份（big5）
        backup_rows = {}
        try:
            with open(backup_csv, 'r', encoding='big5') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    key = (row.get('camera_id'), row.get('image_id'))
                    backup_rows[key] = row
            print(f"  從備份讀取了 {len(backup_rows)} 筆資料")
        except Exception as e:
            print(f"  無法讀取備份: {e}")
            backup_rows = {}
        
        # 用備份資料修復亂碼
        fixed_count = 0
        for row in rows:
            key = (row.get('camera_id'), row.get('image_id'))
            if key in backup_rows:
                backup_reason = backup_rows[key].get('failure_reason', '')
                current_reason = row.get('failure_reason', '')
                
                # 如果當前是亂碼，但備份不是（或備份也是但我們可以嘗試修復）
                if has_garbled_text(current_reason):
                    # 嘗試修復備份的編碼
                    fixed_backup = try_fix_encoding(backup_reason, 'big5')
                    if fixed_backup and not has_garbled_text(fixed_backup):
                        row['failure_reason'] = fixed_backup
                        fixed_count += 1
                        print(f"  從備份修復: camera_id={key[0]}, image_id={key[1]}")
        
        print(f"  從備份修復了 {fixed_count} 筆")
    
    # 寫入修正後的檔案
    print(f"\n寫入: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✓ 完成")
    
    # 驗證
    print(f"\n驗證...")
    with open(output_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        test_rows = list(reader)
        garbled_after = sum(1 for r in test_rows if has_garbled_text(r.get('failure_reason', '')))
        print(f"  剩餘亂碼: {garbled_after} 筆")
        
        if garbled_after == 0:
            print(f"  ✓ 所有編碼問題已修復！")
        else:
            print(f"  ⚠ 仍有 {garbled_after} 筆亂碼需要手動檢查")


def main():
    input_csv = 'outputs/diagnostic_with_failure_modes.csv'
    output_csv = 'outputs/diagnostic_with_failure_modes.csv'
    
    print("=" * 60)
    print("  Fix CSV Encoding Issues")
    print("=" * 60)
    print()
    
    # 備份
    backup_csv = output_csv + '.fix_backup'
    if os.path.exists(output_csv):
        import shutil
        shutil.copy2(output_csv, backup_csv)
        print(f"備份現有檔案: {backup_csv}")
    
    print()
    fix_csv_encoding(input_csv, output_csv)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
