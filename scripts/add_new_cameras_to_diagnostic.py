"""
從新的 camera（3888 / 4795 / 21444）各抽 20 張，加到 diagnostic_candidates.csv。

注意：
- 不修改既有列，只追加新列
- 3888: scene=sea, sea_sky_confusable=1, has_sky=TRUE
- 4795: occlusion=heavy
- 21444: has_sky=FALSE（視為無天空）
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from utils.image_utils import mean_luma_linear


CSV_PATH = Path("data/test_data/diagnostic_candidates.csv")


def list_images(camera_id: str, max_count: int = 20) -> List[Tuple[str, str]]:
    """
    列出 skyfinder_<camera_id>/images 下的圖片，回傳 (image_id, image_rel_path) 清單。
    依檔名排序後取前 max_count 張。
    """
    base = Path("data") / f"skyfinder_{camera_id}" / "images"
    if not base.exists():
        print(f"[警告] 找不到 images 目錄: {base}")
        return []

    files = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
                   key=lambda x: x.name)
    selected = files[:max_count]
    results: List[Tuple[str, str]] = []
    for p in selected:
        image_id = p.stem  # 例如 "001"
        rel_path = os.path.relpath(p, "data")  # skyfinder_xxxx/images/001.jpg
        results.append((image_id, rel_path))
    print(f"  camera {camera_id}: 選取 {len(results)} 張")
    return results


def infer_light(image_path: Path) -> str:
    """讀取圖片並根據 mean_luma_linear 推斷 light。"""
    img = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(img)
    luma = mean_luma_linear(tensor)
    if luma < 0.15:
        return "night"
    elif luma < 0.30:
        return "dusk"
    else:
        return "day"


def build_row(camera_id: str, image_id: str, rel_path: str) -> dict:
    """
    根據 camera_id 和圖片路徑建立一筆 diagnostic row。
    會自動計算 light，其他欄位依 camera 特性給預設值。
    """
    full_path = Path("data") / rel_path
    light = infer_light(full_path)

    # 預設值
    has_sky = "TRUE"
    sea_sky_confusable = "0"
    scene = "unknown"
    occlusion = "none"
    weather = "unknown"

    if camera_id == "3888":
        # 海天容易混淆的 case
        scene = "sea"
        sea_sky_confusable = "1"
        has_sky = "TRUE"
    elif camera_id == "4795":
        # 遮擋嚴重
        occlusion = "heavy"
    elif camera_id == "21444":
        # 幾乎沒有天空
        has_sky = "FALSE"

    # 目前無可靠 mask，可先將 pred_sky_area_ratio 設為空白或 0.0，之後若有需要可再補
    pred_ratio = ""

    return {
        "camera_id": camera_id,
        "image_id": image_id,
        "path": rel_path.replace("\\", "\\"),
        "has_sky": has_sky,
        "sea_sky_confusable": sea_sky_confusable,
        "scene(sea|urban|forest|other)": scene,
        "occlusion(none|partial|heavy)": occlusion,
        "weather(clear|cloudy|rain|fog|snow|unknown)": weather,
        "light (day|dusk|night)": light,
        "pred_sky_area_ratio": pred_ratio,
        "notes": "",
    }


def main():
    if not CSV_PATH.exists():
        print(f"[錯誤] 找不到 CSV 檔案: {CSV_PATH}")
        return

    # 讀取現有資料
    print("讀取現有 diagnostic_candidates.csv ...")
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    existing_keys = {(r["camera_id"], r["image_id"]) for r in rows}
    print(f"  目前共有 {len(rows)} 筆資料")

    new_rows: list[dict] = []
    for cid in ["3888", "4795", "21444"]:
        samples = list_images(cid, max_count=20)
        for image_id, rel_path in samples:
            key = (cid, image_id)
            if key in existing_keys:
                print(f"  [略過] camera {cid}, image_id {image_id} 已存在")
                continue
            row = build_row(cid, image_id, rel_path)
            new_rows.append(row)

    if not new_rows:
        print("沒有新的列需要新增。")
        return

    # 追加並寫回
    all_rows = rows + new_rows
    print(f"寫回 CSV，總筆數 {len(all_rows)}（新增 {len(new_rows)} 筆）")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # 生成縮圖 preview
    from PIL import Image as PILImage

    preview_root = Path("data/test_data/previews")
    thumb_size = 256

    print("\n生成新增圖片的縮圖 ...")
    for row in new_rows:
        cid = row["camera_id"]
        image_id = row["image_id"]
        rel_path = row["path"]
        img_path = Path("data") / rel_path
        if not img_path.exists():
            continue

        out_dir = preview_root / f"camera_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_id}.jpg"

        try:
            img = PILImage.open(img_path).convert("RGB")
            img.thumbnail((thumb_size, thumb_size), PILImage.Resampling.LANCZOS)

            canvas = PILImage.new("RGB", (thumb_size, thumb_size), (255, 255, 255))
            x = (thumb_size - img.width) // 2
            y = (thumb_size - img.height) // 2
            canvas.paste(img, (x, y))
            canvas.save(out_path, "JPEG", quality=85)
            print(f"  [OK] camera {cid} / {image_id}.jpg")
        except Exception as e:
            print(f"  [警告] 生成縮圖失敗 {img_path}: {e}")

    print("\n完成新增三個新 camera 的 20 張樣本到 diagnostic_candidates.csv。")


if __name__ == "__main__":
    main()

