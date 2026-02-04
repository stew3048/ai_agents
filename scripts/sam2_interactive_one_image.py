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
import numpy as np
from PIL import Image
import matplotlib
# 優先 Qt（需 pip install pyqt5），其次 Tk，最後用 WebAgg（瀏覽器開互動）
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('WebAgg')  # 會開瀏覽器，終端會印出 URL
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# 從 eval_sam2_failure_cases 用的 model 對照
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


def build_overlay(image_np, pred_mask, gt_mask=None, alpha=0.5):
    """產生 overlay 圖（與 eval_sam2_failure_cases 一致：紅 FP、藍 FN、無 GT 時綠為預測）。"""
    h, w = image_np.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        from PIL import Image as PImage
        pred_img = PImage.fromarray((np.clip(pred_mask, 0, 1) * 255).astype(np.uint8))
        pred_mask = np.array(pred_img.resize((w, h), PImage.NEAREST), dtype=np.float32) / 255.0
    pred_b = (pred_mask > 0.5)
    overlay = image_np.astype(np.float32).copy()
    if gt_mask is not None:
        if gt_mask.shape[:2] != (h, w):
            from PIL import Image as PImage
            gt_img = PImage.fromarray((np.clip(gt_mask, 0, 1) * 255).astype(np.uint8))
            gt_mask = np.array(gt_img.resize((w, h), PImage.NEAREST), dtype=np.float32) / 255.0
        gt_b = (gt_mask > 0.5)
        fn_mask = gt_b & (~pred_b)
        fp_mask = pred_b & (~gt_b)
        blue = np.array([0, 100, 255], dtype=np.float32)
        red = np.array([255, 50, 50], dtype=np.float32)
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        overlay = np.where(fp_3d, overlay * (1 - alpha * 0.9) + red * (alpha * 0.9), overlay)
    else:
        pred_3d = np.stack([pred_b] * 3, axis=-1)
        green = np.array([50, 255, 50], dtype=np.float32)
        overlay = np.where(pred_3d, overlay * (1 - alpha * 0.5) + green * (alpha * 0.5), overlay)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_interactive(image_path, model_type='sam2_hiera_small', device='cpu'):
    if not SAM2_AVAILABLE:
        print("[ERROR] 請在已安裝 sam2 的環境執行（例如 .venv）")
        return
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
    ax.set_title('左鍵：加點（天空）| P：預測 | C：清除點 | Q：結束')
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
            ax.set_title('左鍵：加點（天空）| P：預測 | C：清除點 | Q：結束')
            print("  已清除所有點")
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
            ax.set_title('預測結果（紅=FP 藍=FN 綠=預測）| 可再點選後按 P 重算 | Q：結束')
            ax.imshow(overlay)
            scatter_artist.set_offsets(point_coords)
            fig.canvas.draw_idle()
            # 可選存檔
            out_dir = 'outputs/sam2_interactive'
            os.makedirs(out_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_dir, f'{basename}_interactive_overlay.png')
            Image.fromarray(overlay).save(out_path)
            print(f"  Overlay 已存到: {out_path}")

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
    args = parser.parse_args()

    image_path = resolve_image_path(
        camera_id=args.camera_id,
        image_id=args.image_id,
        image_path=args.image,
    )
    if not image_path:
        sys.exit(1)
    print(f"圖片: {image_path}")
    run_interactive(image_path, model_type=args.model_type, device=args.device)


if __name__ == '__main__':
    main()
