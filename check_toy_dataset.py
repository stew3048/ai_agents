"""
Toy Dataset 檔案配對檢查工具
確認每張圖片都有對應的 mask，並統計配對結果
"""

import os
import argparse
from pathlib import Path


def check_dataset_pairing(images_dir, masks_dir, image_exts=None, mask_exts=None):
    """
    檢查圖片與 mask 的配對情況
    
    參數:
        images_dir: 圖片資料夾路徑
        masks_dir: mask 資料夾路徑
        image_exts: 圖片副檔名清單 (預設: ['.jpg', '.jpeg', '.png', '.bmp'])
        mask_exts: mask 副檔名清單 (預設: ['.png', '.jpg', '.bmp'])
    
    返回:
        dict: 包含統計結果的字典
    """
    if image_exts is None:
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm']
    if mask_exts is None:
        mask_exts = ['.png', '.jpg', '.bmp', '.pgm']
    
    # 確保副檔名都是小寫
    image_exts = [ext.lower() for ext in image_exts]
    mask_exts = [ext.lower() for ext in mask_exts]
    
    # 檢查資料夾是否存在
    if not os.path.exists(images_dir):
        return {
            'error': f"圖片資料夾不存在: {images_dir}",
            'success': False
        }
    
    if not os.path.exists(masks_dir):
        return {
            'error': f"Mask 資料夾不存在: {masks_dir}",
            'success': False
        }
    
    # 取得所有圖片檔案（不含副檔名的名稱）
    image_files = {}
    for f in os.listdir(images_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_exts:
            base_name = os.path.splitext(f)[0]
            image_files[base_name] = f
    
    # 取得所有 mask 檔案（不含副檔名的名稱）
    mask_files = {}
    for f in os.listdir(masks_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in mask_exts:
            base_name = os.path.splitext(f)[0]
            mask_files[base_name] = f
    
    # 進行配對檢查
    image_names = set(image_files.keys())
    mask_names = set(mask_files.keys())
    
    # 配對成功的
    paired = image_names & mask_names
    
    # 有圖片但沒有 mask
    missing_masks = image_names - mask_names
    
    # 有 mask 但沒有圖片
    missing_images = mask_names - image_names
    
    return {
        'success': True,
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'paired_count': len(paired),
        'paired_files': sorted(paired),
        'missing_masks': sorted(missing_masks),
        'missing_images': sorted(missing_images),
        'image_files': image_files,
        'mask_files': mask_files
    }


def print_report(result):
    """
    印出檢查報告
    
    參數:
        result: check_dataset_pairing 的返回結果
    """
    print("=" * 60)
    print("  Toy Dataset 檔案配對檢查報告")
    print("=" * 60)
    
    if not result['success']:
        print(f"\n❌ 錯誤: {result['error']}")
        return
    
    print(f"\n[圖片資料夾] {result['images_dir']}")
    print(f"[Mask 資料夾] {result['masks_dir']}")
    
    print("\n" + "-" * 60)
    print("  統計摘要")
    print("-" * 60)
    
    print(f"\n圖片總數:     {result['total_images']} 張")
    print(f"Mask 總數:    {result['total_masks']} 張")
    print(f"配對成功:     {result['paired_count']} 對")
    
    # 計算配對率
    if result['total_images'] > 0:
        pair_rate = result['paired_count'] / result['total_images'] * 100
        print(f"配對成功率:   {pair_rate:.1f}%")
    
    # 檢查是否完全配對
    if len(result['missing_masks']) == 0 and len(result['missing_images']) == 0:
        print("\n[OK] 所有檔案配對成功！")
    else:
        if len(result['missing_masks']) > 0:
            print(f"\n[WARNING] 有 {len(result['missing_masks'])} 張圖片缺少對應的 mask:")
            for name in result['missing_masks'][:10]:  # 最多顯示 10 個
                img_file = result['image_files'][name]
                print(f"   - {img_file} (缺少 mask)")
            if len(result['missing_masks']) > 10:
                print(f"   ... 還有 {len(result['missing_masks']) - 10} 個")
        
        if len(result['missing_images']) > 0:
            print(f"\n[WARNING] 有 {len(result['missing_images'])} 個 mask 缺少對應的圖片:")
            for name in result['missing_images'][:10]:  # 最多顯示 10 個
                mask_file = result['mask_files'][name]
                print(f"   - {mask_file} (缺少圖片)")
            if len(result['missing_images']) > 10:
                print(f"   ... 還有 {len(result['missing_images']) - 10} 個")
    
    # 顯示配對成功的檔案列表（前 5 個範例）
    if result['paired_count'] > 0:
        print("\n" + "-" * 60)
        print("  配對成功範例 (前 5 組)")
        print("-" * 60)
        for name in result['paired_files'][:5]:
            img_file = result['image_files'][name]
            mask_file = result['mask_files'][name]
            print(f"   {img_file} <-> {mask_file}")
        if result['paired_count'] > 5:
            print(f"   ... 還有 {result['paired_count'] - 5} 組")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='檢查 Toy Dataset 檔案配對')
    parser.add_argument('--images_dir', type=str, default='data/toy/images',
                        help='圖片資料夾路徑 (預設: data/toy/images)')
    parser.add_argument('--masks_dir', type=str, default='data/toy/masks',
                        help='Mask 資料夾路徑 (預設: data/toy/masks)')
    
    args = parser.parse_args()
    
    result = check_dataset_pairing(args.images_dir, args.masks_dir)
    print_report(result)
    
    # 返回狀態碼（方便自動化腳本使用）
    if result['success'] and len(result['missing_masks']) == 0 and len(result['missing_images']) == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
