"""
SAM 2.0 互動模式：在一張圖上點選「這裡是天空」，按 P 跑預測並顯示 overlay。

請在已安裝 sam2 的環境執行（例如專案 .venv）。若無 Tk，請 pip install pyqt5 以用 Qt 視窗。

用法:
  # 用 failure cases 第一張（會自動解析路徑）
  python scripts/sam2_interactive_one_image.py

  # 指定 camera_id + image_id（從我們的 data/skyfinder_* 讀圖）
  python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9

  # 指定任意圖片路徑
  python scripts/sam2_interactive_one_image.py --image data/skyfinder_4795/images/009.jpg

操作:
  左鍵點擊：加入「這裡是天空」的點（可多點）
  P：用目前的點跑 SAM 2.0 預測並顯示結果（overlay 會存到 outputs/sam2_interactive/）
  C：清除所有點，重新點
  Q：結束
"""

import os
import sys
import argparse
import csv
import traceback
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 不在此 import matplotlib 或 torch，改在 run_interactive 內依順序載入：
# 跑 SAM 時必須「先 import torch、再 import matplotlib」，否則 Windows 上易出現 c10.dll 錯誤。
MODEL_TYPE_TO_HF_ID = {
    'sam2_hiera_tiny': 'facebook/sam2-hiera-tiny',
    'sam2_hiera_small': 'facebook/sam2-hiera-small',
    'sam2_hiera_large': 'facebook/sam2-hiera-large',
}


def resolve_image_path(camera_id=None, image_id=None, image_path=None):
    """解析出一張圖的絕對路徑。優先 image_path，否則用 camera_id/image_id 從 CSV 或慣例路徑找。"""
    if image_path and os.path.exists(image_path):
        return os.path.abspath(image_path)
    if not camera_id or not image_id:
        # 從 failure_cases_analysis.csv 取第一筆
        csv_path = 'outputs/failure_cases_analysis.csv'
        if not os.path.exists(csv_path):
            print(f"[ERROR] 找不到 {csv_path}，請提供 --image 或 --camera_id --image_id")
            return None
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print("[ERROR] failure_cases_analysis.csv 為空")
            return None
        row = rows[0]
        camera_id = row.get('camera_id', '')
        image_id = row.get('image_id', '')
        print(f"使用 failure cases 第一筆: camera_id={camera_id}, image_id={image_id}")
    # 先試 diagnostic_with_failure_modes 的 path
    diagnostic_csv = 'outputs/diagnostic_with_failure_modes.csv'
    if os.path.exists(diagnostic_csv):
        with open(diagnostic_csv, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if str(row.get('camera_id')) == str(camera_id) and str(row.get('image_id')) == str(image_id):
                    path = row.get('path', '')
                    if path:
                        p = os.path.join('data', path.replace('\\', os.sep))
                        if os.path.exists(p):
                            return os.path.abspath(p)
    # 慣例路徑
    folder = f'skyfinder_{camera_id}'
    for name in [f'{int(image_id):03d}.jpg', f'{image_id}.jpg']:
        p = os.path.join('data', folder, 'images', name)
        if os.path.exists(p):
            return os.path.abspath(p)
    print(f"[ERROR] 找不到圖片: camera_id={camera_id}, image_id={image_id}")
    return None


def load_gt_mask_if_any(image_path, camera_id=None, image_id=None):
    """若有 GT mask 則回傳 (mask_np, True)，否則 (None, False)。"""
    path_parts = image_path.replace('\\', '/').split('/')
    if not camera_id or not image_id:
        for part in path_parts:
            if part.startswith('skyfinder_'):
                camera_id = part.replace('skyfinder_', '')
                break
        base = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image_id = str(int(base))
        except ValueError:
            image_id = base
    if not camera_id and not image_id:
        return None, False
    # 試：同目錄的 ../masks/xxx.png
    img_dir = os.path.dirname(image_path)
    if 'images' in img_dir:
        mask_dir = img_dir.replace('images', 'masks')
        mask_path = os.path.join(mask_dir, f'{int(image_id):03d}.png')
    else:
        mask_path = None
    if not mask_path or not os.path.exists(mask_path):
        folder = f'skyfinder_{camera_id}' if camera_id else None
        if folder:
            mask_path = os.path.join('data', folder, 'masks', f'{int(image_id):03d}.png')
        else:
            return None, False
    if not os.path.exists(mask_path):
        return None, False
    mask_np = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
    return mask_np, True


def compute_metrics(pred_mask, gt_mask=None, has_gt=True, smooth=1e-6):
    """計算 IoU, FP rate, FN rate（與 eval_sam2_failure_cases 一致）。"""
    pred_b = (pred_mask > 0.5).astype(np.float32)
    total_pixels = pred_mask.size
    pred_positive_ratio = float(pred_b.sum() / (total_pixels + smooth))
    if not has_gt or gt_mask is None:
        return {'iou': None, 'fp_rate': pred_positive_ratio, 'fn_rate': None}
    gt_b = (gt_mask > 0.5).astype(np.float32)
    if pred_b.shape != gt_b.shape:
        from PIL import Image as PImage
        pred_img = PImage.fromarray((pred_b * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_b.shape[1], gt_b.shape[0]), PImage.NEAREST)
        pred_b = np.array(pred_img, dtype=np.float32) / 255.0
        pred_b = (pred_b > 0.5).astype(np.float32)
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 1.0
    fp_rate = float(fp / (tn + fp + smooth))
    fn_rate = float(fn / (tp + fn + smooth))
    return {'iou': iou, 'fp_rate': fp_rate, 'fn_rate': fn_rate}


def append_experiment_log(log_path, experiment_name, overlay_filename, image_basename, camera_id, image_id, iou, fp_rate, fn_rate):
    """將一筆實驗記錄追加到 CSV（Excel 可開）。"""
    import csv
    from datetime import datetime
    file_exists = os.path.isfile(log_path)
    row = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'camera_id': camera_id or '',
        'image_id': image_id or '',
        'experiment': experiment_name or '(default)',
        'overlay_file': overlay_filename,
        'image': image_basename,
        'IoU': '' if iou is None else f'{iou:.6f}',
        'FP_rate': f'{fp_rate:.6f}' if fp_rate is not None else '',
        'FN_rate': '' if fn_rate is None else f'{fn_rate:.6f}',
    }
    fieldnames = ['time', 'camera_id', 'image_id', 'experiment', 'overlay_file', 'image', 'IoU', 'FP_rate', 'FN_rate']
    with open(log_path, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def build_overlay(image_np, pred_mask, gt_mask=None, alpha=0.92):
    """Overlay 只顯示綠/紅/藍三色（高不透明度，不與原圖混成第四色）。"""
    h, w = image_np.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        from PIL import Image as PImage
        pred_img = PImage.fromarray((np.clip(pred_mask, 0, 1) * 255).astype(np.uint8))
        pred_mask = np.array(pred_img.resize((w, h), PImage.NEAREST), dtype=np.float32) / 255.0
    pred_b = (pred_mask > 0.5)
    overlay = image_np.astype(np.float32).copy()
    green = np.array([50, 255, 50], dtype=np.float32)
    red = np.array([255, 50, 50], dtype=np.float32)
    blue = np.array([0, 100, 255], dtype=np.float32)
    pred_3d = np.stack([pred_b] * 3, axis=-1)
    # 綠：預測為天空（高 alpha，幾乎純綠不混色）
    overlay = np.where(pred_3d, overlay * (1 - alpha) + green * alpha, overlay)
    if gt_mask is not None:
        if gt_mask.shape[:2] != (h, w):
            from PIL import Image as PImage
            gt_img = PImage.fromarray((np.clip(gt_mask, 0, 1) * 255).astype(np.uint8))
            gt_mask = np.array(gt_img.resize((w, h), PImage.NEAREST), dtype=np.float32) / 255.0
        gt_b = (gt_mask > 0.5)
        fn_mask = gt_b & (~pred_b)
        fp_mask = pred_b & (~gt_b)
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        overlay = np.where(fp_3d, overlay * (1 - alpha) + red * alpha, overlay)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _setup_matplotlib():
    """設定 matplotlib 後端並回傳 (plt, backend_name)。"""
    import warnings
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
    import matplotlib
    try:
        import PyQt5
        matplotlib.use('Qt5Agg')
        backend = 'Qt5Agg'
    except ImportError:
        try:
            matplotlib.use('TkAgg')
            backend = 'TkAgg'
        except Exception:
            matplotlib.use('WebAgg')
            backend = 'WebAgg'
    import matplotlib.pyplot as plt
    return plt, backend


def run_interactive(image_path, model_type='sam2_hiera_small', device='cpu', no_sam=False, filename_suffix=''):
    if no_sam:
        plt, backend = _setup_matplotlib()
        print(f"Matplotlib 後端: {backend}")
        # 只顯示圖片 + 點擊／按鍵互動（不載入 SAM），用來確認視窗與互動正常
        print("載入圖片（測試模式：可點選、按 C 清除、按 P 會提示未載入 SAM）...")
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        points_list = []

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image_np)
        ax.set_title('Test: Left-click=add point | P=info | C=clear | Q=quit (no SAM)')
        scatter_artist = ax.scatter([], [], c='lime', s=80, marker='o', edgecolors='black', linewidths=2, zorder=5)

        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < w and 0 <= y < h:
                points_list.append((x, y))
                scatter_artist.set_offsets(points_list)
                fig.canvas.draw_idle()
                print(f"  已加點 ({len(points_list)}): x={x}, y={y}")

        def on_key(event):
            if event.key == 'q':
                plt.close(fig)
                return
            if event.key == 'c':
                points_list.clear()
                scatter_artist.set_offsets([])
                fig.canvas.draw_idle()
                print("  已清除所有點")
                return
            if event.key == 'p':
                if points_list:
                    print("  [測試模式] 已記錄", len(points_list), "個點；未載入 SAM 故不預測。關閉視窗後會跑含 SAM 的步驟。")
                else:
                    print("  請先至少點一個點再按 P（測試模式不會真的跑 SAM）")
                return

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()
        return
    # 跑 SAM：必須先 import torch、再 import matplotlib，否則 Windows 易出現 c10.dll 錯誤
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:
        print("[ERROR] 無法載入 sam2/torch，請確認 .venv 與環境：", e)
        return
    plt, backend = _setup_matplotlib()
    print(f"Matplotlib 後端: {backend}")
    print("載入 SAM 2.0...")
    model_id = MODEL_TYPE_TO_HF_ID.get(model_type, 'facebook/sam2-hiera-small')
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    print("載入圖片...")
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    points_list = []  # [(x, y), ...] 像素座標

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image_np)
    ax.set_title('Left-click=sky point | P=predict | C=clear | Q=quit')
    scatter_artist = ax.scatter([], [], c='lime', s=80, marker='o', edgecolors='black', linewidths=2, zorder=5)

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < w and 0 <= y < h:
            points_list.append((x, y))
            scatter_artist.set_offsets(points_list)
            fig.canvas.draw_idle()
            print(f"  已加點 ({len(points_list)}): x={x}, y={y}")

    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
            return
        if event.key == 'c':
            points_list.clear()
            scatter_artist.set_offsets([])
            ax.imshow(image_np)
            ax.set_title('Left-click=sky point | P=predict | C=clear | Q=quit')
            fig.canvas.draw_idle()
            print("  已清除所有點，可重新點選（例如故意點在建築上測試誤判）")
            return
        if event.key == 'p':
            if not points_list:
                print("  請先至少點一個點再按 P")
                return
            point_coords = np.array(points_list, dtype=np.float32)  # Nx2 (x,y)
            point_labels = np.ones(len(points_list), dtype=np.int32)
            print("  執行 SAM 2.0 預測...")
            predictor.set_image(image_np)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True,
            )
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx].astype(np.float32)
            gt_mask, has_gt = load_gt_mask_if_any(image_path)
            overlay = build_overlay(image_np, pred_mask, gt_mask)
            ax.set_title('Result: Green=pred sky | Red=FP | Blue=FN | P=re-run | Q=quit')
            ax.imshow(overlay)
            scatter_artist.set_offsets(point_coords)
            fig.canvas.draw_idle()
            # 存檔（若有 suffix 則加入檔名）
            out_dir = 'outputs/sam2_interactive'
            os.makedirs(out_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image_path))[0]
            safe_suffix = "".join(c for c in filename_suffix.strip() if c.isalnum() or c in '._- ') if filename_suffix else ""
            if safe_suffix:
                out_name = f'{basename}_interactive_overlay_{safe_suffix}.png'
            else:
                out_name = f'{basename}_interactive_overlay.png'
            out_path = os.path.join(out_dir, out_name)
            Image.fromarray(overlay).save(out_path)
            print(f"  Overlay 已存到: {out_path}")
            # 從路徑解析 camera_id / image_id（例如 .../skyfinder_4795/images/009.jpg）
            path_parts = image_path.replace('\\', '/').split('/')
            cam_id, img_id = '', ''
            for part in path_parts:
                if part.startswith('skyfinder_'):
                    cam_id = part.replace('skyfinder_', '')
                    break
            img_id = os.path.splitext(os.path.basename(image_path))[0]
            # 計算 IoU/FP/FN 並寫入同一份實驗記錄 CSV（Excel 可開）
            metrics = compute_metrics(pred_mask, gt_mask, has_gt=has_gt)
            log_path = os.path.join(out_dir, 'experiments.csv')
            append_experiment_log(
                log_path,
                experiment_name=safe_suffix or '(default)',
                overlay_filename=out_name,
                image_basename=os.path.basename(image_path),
                camera_id=cam_id,
                image_id=img_id,
                iou=metrics['iou'],
                fp_rate=metrics['fp_rate'],
                fn_rate=metrics['fn_rate'],
            )
            iou_s = f"{metrics['iou']:.4f}" if metrics['iou'] is not None else "N/A"
            fp_s = f"{metrics['fp_rate']:.4f}" if metrics['fp_rate'] is not None else "N/A"
            fn_s = f"{metrics['fn_rate']:.4f}" if metrics['fn_rate'] is not None else "N/A"
            print(f"  已寫入實驗記錄: {log_path} (IoU={iou_s}, FP={fp_s}, FN={fn_s})")

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='SAM 2.0 互動模式：點選天空後按 P 預測')
    parser.add_argument('--image', type=str, default=None, help='圖片路徑')
    parser.add_argument('--camera_id', type=str, default=None, help='camera_id（與 image_id 一起用）')
    parser.add_argument('--image_id', type=str, default=None, help='image_id')
    parser.add_argument('--model_type', type=str, default='sam2_hiera_small')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--no-sam', action='store_true', help='只顯示圖片不載入 SAM，用來測試視窗是否正常')
    parser.add_argument('--suffix', type=str, default='', help='overlay 檔名後綴，例如 Test1 會存成 009_interactive_overlay_Test1.png')
    args = parser.parse_args()

    image_path = resolve_image_path(
        camera_id=args.camera_id,
        image_id=args.image_id,
        image_path=args.image,
    )
    if not image_path:
        sys.exit(1)
    print(f"圖片: {image_path}")
    try:
        run_interactive(
            image_path,
            model_type=args.model_type,
            device=args.device,
            no_sam=getattr(args, 'no_sam', False),
            filename_suffix=getattr(args, 'suffix', '') or '',
        )
    except Exception as e:
        print("[ERROR] 執行時發生錯誤：")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
