import csv
import collections
import statistics

rows = []
with open('output/test_overlay/test_data_metrics.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        m = row.get('method', '').strip()
        if not m:
            break
        if m == 'method':
            continue
        cam = row.get('camera_id', '')
        if not cam.startswith('skyfinder_'):
            continue
        try:
            rows.append({
                'method': m,
                'camera_id': cam,
                'image_path': row['image_path'],
                'iou': float(row['iou']),
            })
        except Exception:
            pass

print(f'Loaded {len(rows)} rows')
# skyfinder_21444 的 GT mask 全為黑（無天空），GDino/CLIPSeg 回傳 has_gt=False
# → iou 空白，分析意義不大，單獨列出後排除
excluded = ['skyfinder_21444']
rows_valid = [r for r in rows if r['camera_id'] not in excluded]
cameras = sorted(set(r['camera_id'] for r in rows_valid))
methods = ['GroundingDINO+SAM', 'CLIPSeg', 'DINO-segmentation']
M = {
    'GroundingDINO+SAM': 'GDino',
    'CLIPSeg': 'CLIPSeg',
    'DINO-segmentation': 'DINO',
}
rows = rows_valid
print()
print('NOTE: skyfinder_21444 已排除（GT mask 全黑 = 無天空，GDino/CLIPSeg 無法計算 IoU）')
print(f'      僅分析 {len(cameras)} 個有效 camera，共 {len(rows)} 筆資料')

# ── Q1: 哪個 Camera 最難預測（三方法 IoU 平均最低）──
print()
print('=' * 70)
print('Q1：三方法平均 IoU（由低到高）')
print('=' * 70)
cam_avg = {}
for cam in cameras:
    vals = [r['iou'] for r in rows if r['camera_id'] == cam]
    cam_avg[cam] = statistics.mean(vals)

for cam, avg in sorted(cam_avg.items(), key=lambda x: x[1]):
    per = {
        m: statistics.mean([r['iou'] for r in rows if r['camera_id'] == cam and r['method'] == m])
        for m in methods
    }
    print(
        f"  {cam:25s}  avg={avg:.4f}"
        f"  GDino={per['GroundingDINO+SAM']:.4f}"
        f"  CLIPSeg={per['CLIPSeg']:.4f}"
        f"  DINO={per['DINO-segmentation']:.4f}"
    )

# ── Q2: 每個 Camera 最適合哪個方法 ──
print()
print('=' * 70)
print('Q2：每個 Camera 最適合的方法（IoU 最高）')
print('=' * 70)
for cam in cameras:
    per = {
        m: statistics.mean([r['iou'] for r in rows if r['camera_id'] == cam and r['method'] == m])
        for m in methods
    }
    best_m = max(per, key=per.get)
    print(
        f"  {cam:25s}  Best: {M[best_m]:7s} ({per[best_m]:.4f})"
        f"  GDino={per['GroundingDINO+SAM']:.4f}"
        f"  CLIPSeg={per['CLIPSeg']:.4f}"
        f"  DINO={per['DINO-segmentation']:.4f}"
    )

# ── Q3: 最能顯示方法差異的 image ──
print()
print('=' * 70)
print('Q3：方法差異最大的 images（max IoU - min IoU 排序）')
print('=' * 70)
img_data = collections.defaultdict(dict)
for r in rows:
    img_data[r['image_path']][r['method']] = r['iou']

candidates = []
for img, mvals in img_data.items():
    if len(mvals) < 3:
        continue
    best_m = max(mvals, key=mvals.get)
    worst_m = min(mvals, key=mvals.get)
    spread = mvals[best_m] - mvals[worst_m]
    candidates.append({
        'img': img,
        'vals': mvals,
        'spread': spread,
        'best': best_m,
        'worst': worst_m,
    })

candidates.sort(key=lambda x: -x['spread'])

print('Top 20 差異最大 images:')
for i, c in enumerate(candidates[:20]):
    fname = c['img'].replace('\\', '/').split('/')[-1]
    cam = [r['camera_id'] for r in rows if r['image_path'] == c['img']][0]
    print(
        f"  {i+1:2d}. {cam}/{fname}"
        f"  spread={c['spread']:.4f}"
        f"  best={M[c['best']]}({c['vals'][c['best']]:.4f})"
        f"  worst={M[c['worst']]}({c['vals'][c['worst']]:.4f})"
    )
    for m in methods:
        print(f"       {M[m]:7s}: {c['vals'].get(m, 0):.4f}")
