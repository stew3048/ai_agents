import csv, os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

targets = {
    ('skyfinder_10066', '046'):  '10066_046',
    ('skyfinder_10870', '040'):  '10870_040',
    ('skyfinder_3888',  '1480'): '3888_1480',
    ('skyfinder_4795',  '017'):  '4795_017',
    ('skyfinder_9483',  '017'):  '9483_017',
}

old_results = {}
with open(os.path.join(_root, 'output', 'test_overlay', 'test_data_metrics.csv'), encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        cam   = r.get('camera_id', '')
        img_p = r.get('image_path', '').replace('\\', '/')
        frame = img_p.split('/')[-1].split('.')[0]
        key   = (cam, frame)
        if key in targets:
            label = targets[key]
            method = r.get('method', '')
            old_results.setdefault(label, {})[method] = r

new_results = {}
with open(os.path.join(_root, 'output', 'dino_clipseg', 'metrics.csv'), encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        name = r['image'].replace('.jpg', '').replace('.png', '')
        if name in targets.values():
            new_results[name] = r

print("=" * 70)
print(f"{'影像':<15}  {'方法':<22}  {'IoU':>6}  {'Prec':>6}  {'Recall':>6}  {'FP':>6}  {'FN':>6}")
print("=" * 70)

labels_info = {
    '10066_046': '濃霧',
    '10870_040': '夜間建築物',
    '3888_1480': '天海一線',
    '4795_017':  '夜間強遮擋',
    '9483_017':  '日間建築物',
}

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "  N/A "

for label, scene in labels_info.items():
    print(f"\n[{label}] {scene}")
    old = old_results.get(label, {})
    for method in ['GroundingDINO+SAM', 'CLIPSeg', 'DINO-segmentation']:
        r = old.get(method)
        display = {'GroundingDINO+SAM': 'GDino+SAM', 'CLIPSeg': 'CLIPSeg', 'DINO-segmentation': 'DINOv2+CNN'}.get(method, method)
        if r:
            print(f"  {'(原) ' + display:<22}  {fmt(r['iou']):>6}  {fmt(r['precision']):>6}  {fmt(r['recall']):>6}  {fmt(r['fp_rate']):>6}  {fmt(r['fn_rate']):>6}")
        else:
            print(f"  {'(原) ' + display:<22}  {'N/A':>6}")
    nr = new_results.get(label)
    if nr:
        print(f"  {'(新) DINOv2+CLIPSeg':<22}  {fmt(nr['iou']):>6}  {fmt(nr['precision']):>6}  {fmt(nr['recall']):>6}  {fmt(nr['fp_rate']):>6}  {fmt(nr['fn_rate']):>6}")
    else:
        print(f"  {'(新) DINOv2+CLIPSeg':<22}  {'N/A':>6}")

# 平均
print("\n" + "=" * 70)
print("[平均]")
old_dino_ious  = []
new_dino_ious  = []
for label in labels_info:
    old = old_results.get(label, {})
    r_dino = old.get('DINO-segmentation')
    if r_dino and r_dino['iou']:
        try:
            old_dino_ious.append(float(r_dino['iou']))
        except ValueError:
            pass
    nr = new_results.get(label)
    if nr and nr['iou'] and nr['image'] != 'AVERAGE':
        try:
            new_dino_ious.append(float(nr['iou']))
        except ValueError:
            pass

if old_dino_ious:
    print(f"  (原) DINOv2+CNN 平均 IoU: {sum(old_dino_ious)/len(old_dino_ious):.4f}  (n={len(old_dino_ious)})")
if new_dino_ious:
    print(f"  (新) DINOv2+CLIPSeg 平均 IoU: {sum(new_dino_ious)/len(new_dino_ious):.4f}  (n={len(new_dino_ious)})")
