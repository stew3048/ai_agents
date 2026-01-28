"""
整理新的 camera 資料夾（3888 / 4795 / 21444），對齊 skyfinder_* 格式，
並將 image/mask 重新命名成標準格式（001.jpg / 001.png ...）。

步驟（對每個 camera_id in [3888, 4795, 21444]）：
1. 將 data/<camera_id> 下所有圖片移到 images/ 資料夾
2. 將 mask/ 資料夾改名為 masks/
3. 將資料夾重新命名為 data/skyfinder_<camera_id>
4. 依照檔名排序，將 images/ 重新命名為 001.jpg, 002.jpg, ...
5. 使用 masks/ 中的單一 template mask，複製成 001.png, 002.png, ...
"""

import os
import shutil
from pathlib import Path
from typing import List


def ensure_images_and_masks(camera_folder: Path) -> Path:
    """
    將 data/<camera_id> 整理成 images/ 與 masks/ 結構，並回傳最終資料夾路徑。
    不負責重新命名為 skyfinder_*，那在外層做。
    """
    if not camera_folder.exists():
        print(f"[錯誤] 找不到資料夾: {camera_folder}")
        return camera_folder

    print("=" * 60)
    print(f"  整理 Camera 資料夾結構: {camera_folder}")
    print("=" * 60)

    # 1. 建立 images/ 目錄
    images_dir = camera_folder / "images"
    images_dir.mkdir(exist_ok=True)

    # 2. 移動所有頂層圖片到 images/
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    moved_count = 0
    for file in list(camera_folder.iterdir()):
        if file.is_file() and file.suffix in image_extensions:
            target = images_dir / file.name
            if not target.exists():
                shutil.move(str(file), str(target))
                moved_count += 1

    print(f"  [1] 移動 {moved_count} 張圖片到 images/")

    # 3. 將 mask/ 改名為 masks/
    mask_dir = camera_folder / "mask"
    masks_dir = camera_folder / "masks"
    if mask_dir.exists() and mask_dir.is_dir():
        if masks_dir.exists():
            # 合併 mask → masks
            for file in mask_dir.iterdir():
                target = masks_dir / file.name
                if not target.exists():
                    shutil.move(str(file), str(target))
        else:
            mask_dir.rename(masks_dir)
        print("  [2] 重新命名 mask/ → masks/")
    elif masks_dir.exists():
        print("  [2] 已存在 masks/，略過重新命名")
    else:
        print("  [警告] 找不到 mask/ 或 masks/ 資料夾")

    print("  結構整理完成。")
    return camera_folder


def rename_to_standard_format_generic(camera_folder: Path) -> None:
    """
    將 camera_folder 下的 images/ 重新命名為 001.jpg, 002.jpg ...
    並將 masks/ 中的單一 mask 複製成 001.png, 002.png ...
    """
    images_dir = camera_folder / "images"
    masks_dir = camera_folder / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"[錯誤] 找不到 images/ 或 masks/：{camera_folder}")
        return

    print("=" * 60)
    print(f"  重新命名為標準格式: {camera_folder.name}")
    print("=" * 60)

    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images: List[Path] = sorted(
        [f for f in images_dir.iterdir() if f.is_file() and f.suffix in image_extensions],
        key=lambda x: x.name,
    )

    if not images:
        print("  [錯誤] 找不到任何 image 檔案")
        return

    # 取得 template mask（假設只有一張或隨便取第一張）
    mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file()], key=lambda x: x.name)
    if not mask_files:
        print("  [錯誤] masks/ 中沒有任何 mask 檔案")
        return
    template_mask = mask_files[0]

    print(f"  找到 {len(images)} 張 image，template mask: {template_mask.name}")

    # 先把 template mask 載入記憶體，之後可以安全刪除原檔
    import PIL.Image as PILImage
    tmpl_img = PILImage.open(template_mask).convert("L")

    # 建立臨時目錄避免命名衝突
    temp_images_dir = images_dir.parent / "images_temp"
    temp_images_dir.mkdir(exist_ok=True)

    # 1. 先將 images 按排序重新命名到臨時目錄
    renamed = 0
    for idx, img_file in enumerate(images, start=1):
        new_name = f"{idx:03d}.jpg"
        target = temp_images_dir / new_name
        shutil.move(str(img_file), str(target))
        renamed += 1

    print(f"  [1] 重新命名 {renamed} 張 images → 001.jpg, 002.jpg, ...")

    # 2. 清空原 masks 目錄內容，然後用 template 生成對應數量
    for mf in list(masks_dir.iterdir()):
        if mf.is_file():
            mf.unlink()

    for idx in range(1, renamed + 1):
        mask_name = f"{idx:03d}.png"
        out_path = masks_dir / mask_name
        tmpl_img.save(out_path)

    print(f"  [2] 產生 {renamed} 個 mask：001.png, 002.png, ...")

    # 3. 移回 images_temp → images
    for temp_img in temp_images_dir.iterdir():
        shutil.move(str(temp_img), str(images_dir / temp_img.name))
    temp_images_dir.rmdir()

    print("  [3] 目錄結構更新完成")
    print()


def prepare_camera(camera_id: str) -> None:
    """
    對單一 camera_id（例如 '3888'）執行所有整理步驟。
    """
    raw_folder = Path("data") / camera_id
    if not raw_folder.exists():
        print(f"[警告] 找不到 data/{camera_id}，略過")
        return

    # 1. 整理為 images/ + masks/
    organized_folder = ensure_images_and_masks(raw_folder)

    # 2. 重新命名為 skyfinder_<id>
    parent = organized_folder.parent
    target_name = f"skyfinder_{camera_id}"
    target_folder = parent / target_name
    if target_folder.exists():
        print(f"  [提示] {target_folder} 已存在，直接使用該資料夾")
        final_folder = target_folder
    else:
        shutil.move(str(organized_folder), str(target_folder))
        print(f"  [4] 重新命名資料夾: {organized_folder.name} → {target_name}")
        final_folder = target_folder

    # 3. 在 skyfinder_* 下標準化命名
    rename_to_standard_format_generic(final_folder)


def main():
    camera_ids = ["3888", "4795", "21444"]
    for cid in camera_ids:
        print("\n" + "#" * 60)
        print(f"# 處理 camera {cid}")
        print("#" * 60)
        prepare_camera(cid)


if __name__ == "__main__":
    main()

