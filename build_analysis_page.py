"""
build_analysis_page.py
建立 output/test_overlay/analyze/ 資料夾，複製指定影像與 overlay，
並生成靜態 HTML 分析頁面。
"""
import os
import csv
import shutil

# ─── 設定 ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "data")
OVERLAY_DIR = os.path.join(BASE_DIR, "output", "test_overlay")
ANALYZE_DIR = os.path.join(OVERLAY_DIR, "analyze")
CSV_PATH   = os.path.join(OVERLAY_DIR, "test_data_metrics.csv")
HTML_PATH  = os.path.join(ANALYZE_DIR, "index.html")

# 目標影像清單：(camera, frame_stem, 場景說明)
TARGET_IMAGES = [
    ("skyfinder_10066", "046",  "濃霧場景"),
    ("skyfinder_10870", "096",  "夜間建築物"),
    ("skyfinder_3888",  "1518", "天海一線"),
    ("skyfinder_4795",  "653",  "夜間強遮擋"),
    ("skyfinder_9112",  "097",  "日間建築物"),
]

METHODS = [
    ("GroundingDINO+SAM", "grounding_dino_sam", ""),
    ("CLIPSeg",            "clipseg",            ""),
    ("DINO-segmentation",  "dino_segmentation",  ""),
]

# 顯示名稱映射（CSV key → 畫面顯示名稱）
DISPLAY_NAMES = {
    "GroundingDINO+SAM": "GroundingDINO+SAM",
    "CLIPSeg":           "CLIPSeg",
    "DINO-segmentation": "DINOv2 + CNN",
}

# 每個情境 × 每個方法的客製化評語
# key: (camera_id, frame_stem, method_key)
SCENE_COMMENTS = {
    ("skyfinder_10066", "046", "grounding_dino_sam"):
        "高 Recall (0.97) 但 FP 暴增，受霧氣干擾嚴重，將大量非天空區域誤判為天空。",
    ("skyfinder_10066", "046", "clipseg"):
        "表現最佳，語義理解在低對比度下最穩，幾乎沒有誤判 (FP=0.0001) 但偏保守。",
    ("skyfinder_10066", "046", "dino_segmentation"):
        "極度保守，雖然精準度高，但 FN 接近 40%，在模糊邊界處無法有效識別。",

    ("skyfinder_10870", "096", "grounding_dino_sam"):
        "徹底崩潰，FP 達 0.99，將所有黑暗中的建築物全數誤認為天空。",
    ("skyfinder_10870", "096", "clipseg"):
        "受黑暗嚴重誤導，僅能勉強區分部分區域，精準度與召回率皆不理想。",
    ("skyfinder_10870", "096", "dino_segmentation"):
        "展現強大魯棒性，IoU 達 0.90，能精確過濾複雜光影，識別出真正的天空輪廓。",

    ("skyfinder_3888", "1518", "grounding_dino_sam"):
        "無法區分海天，Recall 雖高但將海面全部納入，導致 FP 較高。",
    ("skyfinder_3888", "1518", "clipseg"):
        "雖然有語義概念，但對相似色塊的邊界區分力不足，IoU 僅 0.55。",
    ("skyfinder_3888", "1518", "dino_segmentation"):
        "表現優異，能從細微的質地差異切開海天邊界，IoU 突破 0.91。",

    ("skyfinder_4795", "653", "grounding_dino_sam"):
        "完全失效，IoU 僅 0.12，在雜亂的遮擋物與夜色中無法界定範圍。",
    ("skyfinder_4795", "653", "clipseg"):
        "表現尚可，但對遮擋物的邊緣切割不夠銳利，存在部分誤判。",
    ("skyfinder_4795", "653", "dino_segmentation"):
        "穩定性極高，幾乎不受夜間遮擋干擾，IoU 0.92 為三者之冠。",

    ("skyfinder_9112", "097", "grounding_dino_sam"):
        "表現最亮眼 (IoU 0.96)，在特徵明顯的日間場景，SAM 的強大邊緣分割能力優勢盡顯。",
    ("skyfinder_9112", "097", "clipseg"):
        "表現穩定但保守，FN 較高導致 IoU 輸給其他兩者。",
    ("skyfinder_9112", "097", "dino_segmentation"):
        "表現優異且平衡，Recall 高達 0.97，但在極細微邊緣處有輕微過度預測。",
}

os.makedirs(ANALYZE_DIR, exist_ok=True)
print(f"[analyze] 目錄：{ANALYZE_DIR}")

# ─── 讀取 CSV 指標 ──────────────────────────────────────
metrics = {}       # key: (method, cam, fname) → dict
summary_all = {}   # method → list of (iou, prec, rec, fpr, fnr)，全 116 張
neg_samples = {}   # method → list of fp_rate（僅 skyfinder_21444 的 10 張）

with open(CSV_PATH, "r", encoding="utf-8") as f:
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
        try:
            img_p = row["image_path"].replace("\\", "/").split("/")[-1]
            iou   = float(row["iou"])       if row.get("iou")       else None
            prec  = float(row["precision"]) if row.get("precision") else None
            rec   = float(row["recall"])    if row.get("recall")    else None
            fpr   = float(row["fp_rate"])   if row.get("fp_rate")   else None
            fnr   = float(row["fn_rate"])   if row.get("fn_rate")   else None
            metrics[(m, cam, img_p)] = dict(iou=iou, prec=prec, rec=rec, fpr=fpr, fnr=fnr)
            # 全 116 張納入統計
            summary_all.setdefault(m, []).append((iou, prec, rec, fpr, fnr))
            # 負樣本（無天空）專區
            if cam == "skyfinder_21444":
                neg_samples.setdefault(m, []).append(fpr)
        except Exception:
            pass

def get_metric(method, camera, frame):
    """回傳 (iou, prec, rec, fpr, fnr) 或全 None"""
    fname_jpg = frame + ".jpg"
    key = (method, camera, fname_jpg)
    r = metrics.get(key, {})
    return r.get("iou"), r.get("prec"), r.get("rec"), r.get("fpr"), r.get("fnr")

def avg(lst):
    return sum(lst) / len(lst) if lst else None

def safe_avg(lst):
    """平均值，自動忽略 None"""
    valid = [x for x in lst if x is not None]
    return sum(valid) / len(valid) if valid else None

# ─── 複製圖片 ────────────────────────────────────────────
def copy_file(src, dst_dir, new_name):
    dst = os.path.join(dst_dir, new_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return new_name
    else:
        print(f"  [警告] 找不到: {src}")
        return None

copied = {}  # key: (camera, frame, kind) → filename

for cam, frame, label in TARGET_IMAGES:
    # 原始影像
    src_img = os.path.join(DATA_DIR, cam, "images", frame + ".jpg")
    dst_name = f"{cam}_{frame}_original.jpg"
    copied[(cam, frame, "original")] = copy_file(src_img, ANALYZE_DIR, dst_name)

    for method_full, method_key, _ in METHODS:
        # overlay: {cam}_{cam}_{frame}_{method_key}_overlay.png
        overlay_fname = f"{cam}_{cam}_{frame}_{method_key}_overlay.png"
        overlay_dir   = os.path.join(OVERLAY_DIR, f"{method_key}_overlays")
        src_ov = os.path.join(overlay_dir, overlay_fname)
        dst_name_ov = f"{cam}_{frame}_{method_key}_overlay.png"
        copied[(cam, frame, method_key)] = copy_file(src_ov, ANALYZE_DIR, dst_name_ov)

print(f"[analyze] 圖片複製完成")

# ─── 生成 HTML ───────────────────────────────────────────
METHOD_COLORS = {
    "GroundingDINO+SAM": "#e74c3c",
    "CLIPSeg":            "#2980b9",
    "DINO-segmentation":  "#27ae60",
}

def fmt(v, digits=4):
    return f"{v:.{digits}f}" if v is not None else "<span style='color:#aaa'>N/A</span>"

def metric_badge(val, lower_is_better=False):
    if val is None:
        return "<span style='color:#aaa'>N/A</span>"
    good = val >= 0.8 if not lower_is_better else val <= 0.05
    color = "#27ae60" if good else ("#e67e22" if (val >= 0.6 if not lower_is_better else val <= 0.15) else "#e74c3c")
    return f"<b style='color:{color}'>{val:.4f}</b>"

# 全域 summary
# IoU：GDino/CLIPSeg 跳過 None（106 張有天空），DINO-seg 全 116 張（含無天空 IoU=1.0）
# 其餘指標：safe_avg 跳過 None
global_summary = {}
for m, vals in summary_all.items():
    ious  = [v[0] for v in vals]
    precs = [v[1] for v in vals]
    recs  = [v[2] for v in vals]
    fprs  = [v[3] for v in vals]
    fnrs  = [v[4] for v in vals]
    global_summary[m] = {
        "iou":  safe_avg(ious),   # None 自動略過，DINO-seg 無 None 故用足 116 張
        "prec": safe_avg(precs),
        "rec":  safe_avg(recs),
        "fpr":  safe_avg(fprs),
        "fnr":  safe_avg(fnrs),
        "n":    len(vals),
    }

# 負樣本 TNR（僅針對 skyfinder_21444 的 10 張）
# TNR = 1 - fp_rate；fp_rate = None 代表推論失敗
neg_tnr = {}
for m, fprs in neg_samples.items():
    valid = [1 - f for f in fprs if f is not None]
    neg_tnr[m] = avg(valid) if valid else None  # None = 完全無輸出

html_scenarios = []
for cam, frame, scene_label in TARGET_IMAGES:
    orig_file = copied.get((cam, frame, "original"))
    orig_tag  = f'<img src="{orig_file}" alt="原始影像" class="img-box">' if orig_file else "<p>找不到原始影像</p>"

    method_cards = ""
    for method_full, method_key, _ in METHODS:
        ov_file = copied.get((cam, frame, method_key))
        ov_tag  = f'<img src="{ov_file}" alt="{method_full}" class="img-box">' if ov_file else "<p>找不到 overlay</p>"
        iou, prec, rec, fpr, fnr = get_metric(method_full, cam, frame)
        color = METHOD_COLORS[method_full]
        comment = SCENE_COMMENTS.get((cam, frame, method_key), "")
        comment_html = f'<div class="comment">💬 {comment}</div>' if comment else ""
        display_name = DISPLAY_NAMES.get(method_full, method_full)
        method_cards += f"""
        <div class="method-card">
          <div class="method-title" style="border-left:4px solid {color}; padding-left:8px;">
            {display_name}
          </div>
          {ov_tag}
          <div class="metrics-row">
            <span>IoU: {metric_badge(iou)}</span>
            <span>Prec: {metric_badge(prec)}</span>
            <span>Recall: {metric_badge(rec)}</span>
            <span>FP%: {metric_badge(fpr, lower_is_better=True)}</span>
            <span>FN%: {metric_badge(fnr, lower_is_better=True)}</span>
          </div>
          {comment_html}
        </div>"""

    html_scenarios.append(f"""
    <section class="scenario">
      <h2>📷 {scene_label}
        <span class="cam-badge">{cam} / {frame}.jpg</span>
      </h2>
      <div class="image-grid">
        <div class="orig-card">
          <div class="method-title" style="border-left:4px solid #555; padding-left:8px;">原始影像</div>
          {orig_tag}
          <div class="comment">GT mask 色彩說明：🟢 綠=TP（正確天空）　🔴 紅=FP（誤標）　🔵 藍=FN（漏標）</div>
        </div>
        {method_cards}
      </div>
    </section>""")

# 全域比較表
def _get(m, k): return global_summary.get(m, {}).get(k)
best_iou  = max(global_summary, key=lambda m: _get(m,"iou")  or 0)
best_prec = max(global_summary, key=lambda m: _get(m,"prec") or 0)
best_rec  = max(global_summary, key=lambda m: _get(m,"rec")  or 0)
best_fpr  = min(global_summary, key=lambda m: _get(m,"fpr")  if _get(m,"fpr") is not None else 9)
best_fnr  = min(global_summary, key=lambda m: _get(m,"fnr")  if _get(m,"fnr") is not None else 9)
best_neg_tnr = max(neg_tnr,     key=lambda m: neg_tnr.get(m) or 0)

def cell(val, is_best, lower_better=False):
    style = " style='background:#d5f5e3; font-weight:bold;'" if is_best else ""
    return f"<td{style}>{fmt(val)}</td>"

summary_rows = ""
for m in ["GroundingDINO+SAM", "CLIPSeg", "DINO-segmentation"]:
    s = global_summary.get(m, {})
    color = METHOD_COLORS[m]
    tnr_val = neg_tnr.get(m)
    tnr_display = fmt(tnr_val) if tnr_val is not None else "<span style='color:#e74c3c;font-weight:bold;'>推論失敗</span>"
    tnr_style = " style='background:#d5f5e3; font-weight:bold;'" if m == best_neg_tnr and tnr_val is not None else ""
    display = DISPLAY_NAMES.get(m, m)
    n_iou = 116 if m == "DINO-segmentation" else 106
    summary_rows += f"""
    <tr>
      <td style="border-left:4px solid {color}; font-weight:bold; padding-left:8px;">{display}</td>
      <td>{n_iou}</td>
      {cell(s.get('iou'),  m==best_iou)}
      {cell(s.get('prec'), m==best_prec)}
      {cell(s.get('rec'),  m==best_rec)}
      {cell(s.get('fpr'),  m==best_fpr,  True)}
      {cell(s.get('fnr'),  m==best_fnr,  True)}
      <td{tnr_style}>{tnr_display}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>天空分割三模型比較分析</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif; background:#f4f6f8; color:#333; }}
  header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:white; padding:32px 40px; }}
  header h1 {{ font-size:2em; margin-bottom:6px; }}
  header p  {{ color:#aab; font-size:0.95em; }}
  main {{ max-width:1600px; margin:0 auto; padding:24px 32px; }}

  .scenario {{ background:white; border-radius:12px; margin-bottom:36px;
               box-shadow:0 2px 12px rgba(0,0,0,0.08); overflow:hidden; }}
  .scenario h2 {{ padding:18px 24px; background:#fafafa; border-bottom:1px solid #e8e8e8;
                  font-size:1.2em; display:flex; align-items:center; gap:12px; }}
  .cam-badge {{ background:#e8eaf6; color:#3949ab; border-radius:6px;
                padding:3px 10px; font-size:0.78em; font-weight:normal; font-family:monospace; }}

  .image-grid {{ display:grid; grid-template-columns: repeat(4, 1fr);
                 gap:0; }}
  .orig-card, .method-card {{ padding:16px; border-right:1px solid #f0f0f0; }}
  .method-card:last-child, .orig-card {{ border-right:none; }}
  .orig-card {{ background:#fafafa; }}

  .method-title {{ font-size:0.9em; font-weight:600; margin-bottom:10px; color:#333; }}
  .img-box {{ width:100%; aspect-ratio:4/3; object-fit:cover; border-radius:8px;
              border:1px solid #e0e0e0; display:block; }}

  .metrics-row {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; font-size:0.78em; color:#555; }}
  .metrics-row span {{ background:#f5f5f5; border-radius:4px; padding:3px 7px; }}

  .comment {{ margin-top:10px; font-size:0.82em; color:#666; line-height:1.5;
              background:#f9f9f9; border-radius:6px; padding:8px 10px; }}

  /* Summary table */
  .summary-section {{ background:white; border-radius:12px; margin-top:16px;
                      box-shadow:0 2px 12px rgba(0,0,0,0.08); overflow:hidden; }}
  .summary-section h2 {{ padding:18px 24px; background:#fafafa;
                         border-bottom:1px solid #e8e8e8; font-size:1.2em; }}
  .summary-section .note {{ padding:14px 24px; font-size:0.88em; color:#555; line-height:1.6; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.9em; }}
  th {{ background:#37474f; color:white; padding:12px 16px; text-align:center; }}
  td {{ padding:12px 16px; border-bottom:1px solid #f0f0f0; text-align:center; }}
  tr:hover td {{ background:#f9f9f9; }}
  th:first-child, td:first-child {{ text-align:left; }}
  .verdict {{ padding:16px 24px 24px; font-size:0.9em; line-height:1.7; }}
  .verdict strong {{ color:#1a237e; }}

  /* Radar section */
  .radar-section {{ }}
  .radar-body {{ display:flex; gap:32px; padding:24px; align-items:flex-start; }}
  .radar-img-wrap {{ flex:0 0 480px; }}
  .radar-img {{ width:100%; border-radius:10px; border:1px solid #e0e0e0; }}
  .radar-text {{ flex:1; display:flex; flex-direction:column; gap:20px; }}
  .radar-item {{ background:#f9f9f9; border-radius:8px; padding:16px 18px; }}
  .radar-method-title {{ font-size:1.05em; font-weight:700; margin-bottom:8px; }}
  .radar-item p {{ font-size:0.9em; color:#444; line-height:1.7; }}

  @media (max-width:900px) {{
    .image-grid {{ grid-template-columns: 1fr 1fr; }}
    .radar-body {{ flex-direction:column; }}
    .radar-img-wrap {{ flex:none; width:100%; }}
  }}
  @media (max-width:600px) {{
    .image-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>🌤 天空分割三模型比較分析</h1>
  <p>評估資料集：test_list（116 張，9 個場景）｜評估日期：2026-03-13｜環境：sky_sam2 conda (Python 3.10)</p>
</header>
<main>

{"".join(html_scenarios)}

<section class="summary-section radar-section">
  <h2>📡 極端天空情境雷達圖</h2>
  <div class="radar-body">
    <div class="radar-img-wrap">
      <img src="Model_Iou-Comparison.png" alt="天空場景雷達圖" class="radar-img">
    </div>
    <div class="radar-text">
      <div class="radar-item">
        <div class="radar-method-title" style="color:#1f77b4;">🔵 DINOv2 + CNN</div>
        <p>在所有情境下都保持了極高的 IoU（>0.90，除了濃霧情境）。這證明了它強大的魯棒性（Robustness），無論是面對夜間反光、海天一線還是強遮擋，它都能穩定輸出精確的分割結果。這是一個<strong>全能型且可落地</strong>的模型。</p>
      </div>
      <div class="radar-item">
        <div class="radar-method-title" style="color:#2ca02c;">🟢 Grounding DINO + SAM</div>
        <p>一個典型的「<strong>好天氣模型</strong>」。它在標準日間場景（IoU 0.96）表現甚至優於 DINO-seg，證明了 SAM 的邊緣分割實力。但在複雜光影和夜間，它的偵測框（Grounding DINO）會完全失效，拖累了 SAM 的發揮。</p>
      </div>
      <div class="radar-item">
        <div class="radar-method-title" style="color:#ff7f0e;">🟠 CLIPSeg</div>
        <p>表現相對中規中矩。它在濃霧情境下反而表現最好，證明了純語義理解的優勢。但在需要精確辨識邊界（如海天一線）或對抗光影干擾（如夜間反光）時，它的性能上限較低。</p>
      </div>
    </div>
  </div>
</section>

<section class="summary-section">
  <h2>📊 全資料集平均指標（116 張，含無天空場景）</h2>
  <div class="note">
    ✅ 綠底加粗 = 該指標最佳值　｜　IoU / Precision / Recall 越高越好　｜　FP rate / FN rate 越低越好<br>
    ⚠️ <strong>負樣本測試 (Negative Sample Testing)</strong>：資料集包含 10 張完全無天空影像（GT Mask 全黑），用於測試模型之抗干擾能力。<br>
    <strong>TNR（負樣本）</strong>：針對資料集中 10 張完全無天空影像（skyfinder_21444）單獨計算，= 1 − FP rate，衡量模型「在沒有天空時，能正確判斷沒有天空」的能力。
  </div>
  <table>
    <thead>
      <tr>
        <th>方法</th><th>樣本數</th>
        <th>IoU ↑</th><th>Precision ↑</th><th>Recall ↑</th>
        <th>FP rate ↓</th><th>FN rate ↓</th><th>TNR ↑</th>
      </tr>
    </thead>
    <tbody>
      {summary_rows}
    </tbody>
  </table>
  <div class="verdict">
    <strong>結論</strong><br>
    🥇 <strong>DINOv2 + CNN</strong> 在 IoU（{fmt(global_summary.get('DINO-segmentation',{}).get('iou'))}）、FP rate（{fmt(global_summary.get('DINO-segmentation',{}).get('fpr'))}）、FN rate（{fmt(global_summary.get('DINO-segmentation',{}).get('fnr'))}）三項均為最佳，整體表現全面領先。優勢來自在本資料集的 in-domain 訓練，能精準辨識天空邊界。<br><br>
    🥈 <strong>CLIPSeg</strong> IoU（{fmt(global_summary.get('CLIPSeg',{}).get('iou'))}）居中，Precision 最高（{fmt(global_summary.get('CLIPSeg',{}).get('prec'))}），且<b>不需要任何訓練資料</b>，是零樣本方法中最穩定的選擇。<br><br>
    🥉 <strong>GroundingDINO+SAM</strong> Recall 最高（{fmt(global_summary.get('GroundingDINO+SAM',{}).get('rec'))}），幾乎不漏掉天空，但 FP rate 最高（{fmt(global_summary.get('GroundingDINO+SAM',{}).get('fpr'))}），大量誤標非天空區域，拖低 IoU（{fmt(global_summary.get('GroundingDINO+SAM',{}).get('iou'))}）。適合對「<b>不漏掉任何天空</b>」有強烈需求、但能容忍誤報的場景。
  </div>
</section>

</main>
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[analyze] HTML 已生成：{HTML_PATH}")
print(f"[analyze] 完成！用瀏覽器開啟：{HTML_PATH}")
