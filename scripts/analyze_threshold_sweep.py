"""分析 threshold sweep 結果"""
import csv

with open('outputs/threshold_sweep_10870.csv', 'r', encoding='utf-8') as f:
    data = list(csv.DictReader(f))

overall_best = max(data, key=lambda x: float(x['overall_iou']))
night_best = max(data, key=lambda x: float(x['night_iou']))

print("=" * 60)
print("  最佳 Threshold 建議")
print("=" * 60)
print(f"  Overall IoU 最佳: t={overall_best['threshold']}")
print(f"    IoU: {float(overall_best['overall_iou']):.4f}")
print(f"    FP:  {float(overall_best['overall_fp']):.4f}")
print(f"    FN:  {float(overall_best['overall_fn']):.4f}")
print()
print(f"  Night IoU 最佳: t={night_best['threshold']}")
print(f"    IoU: {float(night_best['night_iou']):.4f}")
print(f"    FP:  {float(night_best['night_fp']):.4f}")
print(f"    FN:  {float(night_best['night_fn']):.4f}")
print()
print(f"  對照 t=0.5（目前使用）:")
t05 = next(r for r in data if r['threshold'] == '0.5')
print(f"    Overall IoU: {float(t05['overall_iou']):.4f}")
print(f"    Night IoU: {float(t05['night_iou']):.4f}, FN: {float(t05['night_fn']):.4f}")
print("=" * 60)
