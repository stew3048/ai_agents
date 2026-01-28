"""
準備 VLM/SAM vs DL 比較分析

選出每個情境的代表樣本，並準備 VLM/SAM 的預期行為假設

用法:
  python scripts/prepare_vlm_sam_comparison.py
"""

import os
import sys
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def parse_float(value):
    """解析浮點數"""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_csv(csv_path):
    """讀取 CSV"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 處理數值欄位
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows


def select_scenario_samples(rows, k=10):
    """選出每個情境的代表樣本"""
    scenarios = {
        'no-sky': {
            'condition': lambda r: r.get('has_sky', '').upper() == 'FALSE',
            'sort_key': lambda r: r.get('dl_pred_positive_ratio', 0) if r.get('dl_pred_positive_ratio') is not None else 0,
            'sort_reverse': True,
            'description': 'No-sky：has_sky==False 且 dl_pred_positive_ratio 最大的 top-k'
        },
        'sea-sky': {
            'condition': lambda r: r.get('sea_sky_confusable', '') == '1',
            'sort_key': lambda r: r.get('dl_fp_rate', 0) if r.get('dl_fp_rate') is not None else 0,
            'sort_reverse': True,
            'description': 'Sea-sky confusable：sea_sky_confusable==1 且 dl_fp_rate 最大的 top-k'
        },
        'heavy-occlusion': {
            'condition': lambda r: r.get('occlusion(none|partial|heavy)', '') == 'heavy',
            'sort_key': lambda r: (r.get('dl_fp_rate', 0) if r.get('dl_fp_rate') is not None else 0, 
                                   -(r.get('dl_iou', 1) if r.get('dl_iou') is not None else 1)),
            'sort_reverse': True,
            'description': 'Heavy occlusion：occlusion==heavy 且 dl_fp_rate 最大（或 dl_iou 最低）top-k'
        },
        'night': {
            'condition': lambda r: (r.get('light (day|dusk|night)', '').lower() == 'night' and 
                                   r.get('has_sky', '').upper() == 'TRUE'),
            'sort_key': lambda r: r.get('dl_iou', 1) if r.get('dl_iou') is not None else 1,
            'sort_reverse': False,  # 最低的在前
            'description': 'Night：light==night & has_sky==True 且 dl_iou 最低 top-k'
        },
        'urban': {
            'condition': lambda r: (r.get('scene(sea|urban|forest|other)', '').lower() == 'urban' and 
                                   r.get('has_sky', '').upper() == 'TRUE'),
            'sort_key': lambda r: r.get('dl_fp_rate', 0) if r.get('dl_fp_rate') is not None else 0,
            'sort_reverse': True,
            'description': 'Urban：scene==urban & has_sky==True 且 dl_fp_rate 最大 top-k'
        }
    }
    
    selected_samples = {}
    
    for scenario_name, config in scenarios.items():
        # 篩選符合條件的樣本
        filtered = [r for r in rows if config['condition'](r)]
        
        # 排序
        filtered_sorted = sorted(filtered, key=config['sort_key'], reverse=config['sort_reverse'])
        
        # 取 top-k
        selected = filtered_sorted[:k]
        
        selected_samples[scenario_name] = {
            'samples': selected,
            'total': len(filtered),
            'description': config['description']
        }
        
        print(f"{scenario_name}: 從 {len(filtered)} 個樣本中選出 {len(selected)} 個")
    
    return selected_samples


def generate_vlm_sam_hypotheses():
    """生成 VLM/SAM 的預期行為假設"""
    hypotheses = {
        'no-sky': {
            'vlm': 'VLM 可能透過語義理解「牆/雪地≠天空」，在語義層面較能拒絕誤判，但遇到強反光或高亮度區域時，視覺特徵可能仍會導致誤切。',
            'sam': 'SAM 主要依賴視覺特徵和邊界，對於沒有天空的場景，若提示不佳可能仍會誤判高亮度區域為天空，但透過點/框提示可能比純 DL 更容易控制。'
        },
        'sea-sky': {
            'vlm': 'VLM 可能利用「天空在上方/地平線」的語義知識，在海天顏色相似時表現較好，但在低對比度或霧天時，語義優勢可能減弱。',
            'sam': 'SAM 依賴視覺對比和邊界，在海天顏色相似時可能與 DL 類似地出現誤判，但透過互動式提示（點選天空區域）可能比 DL 更準確。'
        },
        'heavy-occlusion': {
            'vlm': 'VLM 可能透過語義理解「被遮擋的天空仍是天空」，在嚴重遮擋場景下表現較好，但若提示不佳可能漏檢小塊天空區域。',
            'sam': 'SAM 透過點/框互動可能比 DL 更容易抓到小塊天空區域，但若初始提示不佳，可能仍會漏檢被嚴重遮擋的天空。'
        },
        'night': {
            'vlm': 'VLM 在語義層面理解「夜晚也有天空」，但視覺訊號不足（低對比度、低亮度）時，表現可能仍不穩定，可能出現 FP（誤判亮光源為天空）或 FN（漏檢暗天空）。',
            'sam': 'SAM 主要依賴視覺對比，在夜間低對比度場景下可能表現較差，但透過點選天空區域的互動提示，可能比純 DL 更準確。'
        },
        'urban': {
            'vlm': 'VLM 可能利用語義知識排除建築物，在建築邊緣誤判問題上表現較好，但遇到建築物頂部反光或玻璃反光時，仍可能出現誤判。',
            'sam': 'SAM 依賴視覺特徵，在建築邊緣可能仍會誤判，但透過互動式提示（點選天空區域、框選建築物排除）可能比 DL 更容易控制。'
        }
    }
    return hypotheses


def generate_comparison_report(selected_samples, hypotheses, output_path):
    """生成比較分析報告"""
    print(f"\n生成報告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# VLM/SAM vs DL 比較分析準備\n\n")
        f.write("> 選出每個情境的代表樣本，並準備 VLM/SAM 的預期行為假設\n\n")
        
        # 每個情境的分析
        for scenario_name in ['no-sky', 'sea-sky', 'heavy-occlusion', 'night', 'urban']:
            if scenario_name not in selected_samples:
                continue
            
            data = selected_samples[scenario_name]
            samples = data['samples']
            
            f.write(f"## {scenario_name.upper()} 情境\n\n")
            f.write(f"### 樣本選擇\n\n")
            f.write(f"- **條件**：{data['description']}\n")
            f.write(f"- **候選樣本數**：{data['total']}\n")
            f.write(f"- **選出樣本數**：{len(samples)}\n\n")
            
            f.write("### 選出的樣本列表\n\n")
            f.write("| Camera | Image | FP Rate | FN Rate | IoU | Pred Positive Ratio | Failure Reason | Overlay |\n")
            f.write("|--------|-------|---------|---------|-----|---------------------|----------------|---------|\n")
            
            for row in samples:
                camera_id = row.get('camera_id', '')
                image_id = row.get('image_id', '')
                fp_rate = row.get('dl_fp_rate', 0)
                fn_rate = row.get('dl_fn_rate', 0)
                iou = row.get('dl_iou', 0)
                pred_ratio = row.get('dl_pred_positive_ratio', 0)
                reason = row.get('failure_reason', '')
                overlay_path = row.get('overlay_path', '')
                
                fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
                fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
                iou_str = f"{iou:.4f}" if iou is not None else "-"
                pred_str = f"{pred_ratio:.4f}" if pred_ratio is not None else "-"
                reason_short = reason[:30] + "..." if len(reason) > 30 else reason
                overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                
                f.write(f"| {camera_id} | {image_id} | {fp_str} | {fn_str} | {iou_str} | {pred_str} | {reason_short} | {overlay_link} |\n")
            
            f.write("\n")
            
            # VLM/SAM 預期行為假設
            if scenario_name in hypotheses:
                hyp = hypotheses[scenario_name]
                f.write("### VLM/SAM 預期行為假設\n\n")
                f.write(f"**VLM 預期行為**：\n")
                f.write(f"{hyp['vlm']}\n\n")
                f.write(f"**SAM 預期行為**：\n")
                f.write(f"{hyp['sam']}\n\n")
            
            f.write("---\n\n")
        
        # 總結
        f.write("## 總結\n\n")
        f.write("### 樣本統計\n\n")
        f.write("| 情境 | 候選樣本數 | 選出樣本數 |\n")
        f.write("|------|-----------|-----------|\n")
        for scenario_name, data in selected_samples.items():
            f.write(f"| {scenario_name} | {data['total']} | {len(data['samples'])} |\n")
        
        f.write("\n### 下一步\n\n")
        f.write("1. 使用選出的樣本跑 VLM/SAM inference\n")
        f.write("2. 計算 VLM/SAM 的指標（IoU, FP Rate, FN Rate）\n")
        f.write("3. 生成 VLM/SAM 的 overlay 圖\n")
        f.write("4. 對比 DL vs VLM/SAM 的表現\n")
        f.write("5. 驗證預期行為假設是否正確\n")


def export_sample_list(selected_samples, output_csv):
    """匯出樣本清單 CSV"""
    print(f"\n匯出樣本清單: {output_csv}")
    
    all_samples = []
    for scenario_name, data in selected_samples.items():
        for row in data['samples']:
            sample_row = row.copy()
            sample_row['scenario'] = scenario_name
            all_samples.append(sample_row)
    
    if not all_samples:
        print("  沒有樣本可匯出")
        return
    
    # 取得所有欄位
    fieldnames = list(all_samples[0].keys())
    if 'scenario' not in fieldnames:
        fieldnames.insert(0, 'scenario')
    
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)
    
    print(f"  匯出了 {len(all_samples)} 筆樣本")


def main():
    input_csv = 'outputs/diagnostic_with_failure_modes.csv'
    output_report = 'outputs/vlm_sam_comparison_preparation.md'
    output_csv = 'outputs/vlm_sam_comparison_samples.csv'
    
    print("=" * 60)
    print("  Prepare VLM/SAM Comparison")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {input_csv}")
    print(f"  輸出報告:     {output_report}")
    print(f"  輸出 CSV:     {output_csv}")
    print()
    
    print("讀取 CSV...")
    rows = load_csv(input_csv)
    print(f"  讀取了 {len(rows)} 筆資料")
    
    print("\n選出每個情境的代表樣本...")
    selected_samples = select_scenario_samples(rows, k=10)
    
    print("\n生成 VLM/SAM 預期行為假設...")
    hypotheses = generate_vlm_sam_hypotheses()
    
    generate_comparison_report(selected_samples, hypotheses, output_report)
    export_sample_list(selected_samples, output_csv)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
