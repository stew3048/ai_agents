import csv

targets = [
    ("skyfinder_10066", "046",  "濃霧"),
    ("skyfinder_10870", "096",  "夜間建築物反光"),
    ("skyfinder_3888",  "1518", "天海一線"),
    ("skyfinder_4795",  "653",  "夜間強遮擋"),
    ("skyfinder_9112",  "097",  "日間建築物"),
]
methods = ["GroundingDINO+SAM", "CLIPSeg", "DINO-segmentation"]

rows = {}
with open("output/test_overlay/test_data_metrics.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        m = row.get("method", "").strip()
        if not m:
            break
        if m == "method":
            continue
        cam = row.get("camera_id", "")
        if not cam.startswith("skyfinder_"):
            continue
        fname = row["image_path"].replace("\\", "/").split("/")[-1].replace(".jpg", "")
        key = (cam, fname, m)
        try:
            rows[key] = {
                "iou":  float(row["iou"])       if row.get("iou")       else None,
                "prec": float(row["precision"]) if row.get("precision") else None,
                "rec":  float(row["recall"])    if row.get("recall")    else None,
                "fpr":  float(row["fp_rate"])   if row.get("fp_rate")   else None,
                "fnr":  float(row["fn_rate"])   if row.get("fn_rate")   else None,
            }
        except Exception:
            pass

def fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"

for cam, frame, label in targets:
    print(f"=== {label}  ({cam}/{frame}.jpg) ===")
    for m in methods:
        r = rows.get((cam, frame, m), {})
        iou  = fmt(r.get("iou"))
        prec = fmt(r.get("prec"))
        rec  = fmt(r.get("rec"))
        fpr  = fmt(r.get("fpr"))
        fnr  = fmt(r.get("fnr"))
        print(f"  [{m}]")
        print(f"    IoU={iou}  Precision={prec}  Recall={rec}  FP_rate={fpr}  FN_rate={fnr}")
    print()
