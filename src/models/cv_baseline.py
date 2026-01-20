"""
傳統 CV Baseline：基於顏色規則的天空分割

方法：
1. 將 RGB 轉換為 HSV 色彩空間
2. 使用規則判斷「藍色且明亮」的像素
3. 後處理去雜點、補洞

輸入：單張 RGB image（numpy array 或 torch tensor）
輸出：pred_mask（二元 mask，{0, 1}）

注意：這是 rule-based 方法，不需要訓練
"""

import numpy as np
import cv2


def predict_sky_mask_cv(
    image,
    h_min=90,
    h_max=130,
    s_min=30,
    s_max=255,
    v_min=120,
    v_max=255,
    morph_kernel_size=5,
    apply_morphology=True
):
    """
    使用傳統 CV 方法預測天空 mask
    
    Args:
        image: RGB image，可以是以下格式之一：
               - numpy array [H, W, 3], uint8, range [0, 255]
               - numpy array [H, W, 3], float32, range [0, 1]
               - torch tensor [C, H, W], float32, range [0, 1]
        
        h_min, h_max: 色相 (Hue) 範圍，藍色大約在 90-130 (OpenCV 的 H 範圍是 0-179)
        s_min, s_max: 飽和度 (Saturation) 範圍
        v_min, v_max: 明度 (Value) 範圍
        morph_kernel_size: 形態學操作的 kernel 大小
        apply_morphology: 是否進行後處理（去雜點、補洞）
    
    Returns:
        pred_mask: numpy array [H, W], uint8, values {0, 1}
                   1 = sky, 0 = non-sky
    """
    # === Step 0: 輸入格式轉換 ===
    image_np = _convert_to_numpy_uint8(image)
    
    # === Step 1: RGB → HSV ===
    # OpenCV 預設讀取 BGR，但我們的輸入是 RGB，所以要轉換
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # === Step 2: 顏色閾值判斷 ===
    # 定義藍色天空的 HSV 範圍
    lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)
    
    # 產生初步 mask（在範圍內 = 255，否則 = 0）
    raw_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    
    # === Step 3: 後處理 ===
    if apply_morphology:
        processed_mask = _apply_morphology(raw_mask, morph_kernel_size)
    else:
        processed_mask = raw_mask
    
    # === Step 4: 轉換為 {0, 1} ===
    pred_mask = (processed_mask > 0).astype(np.uint8)
    
    return pred_mask


def _convert_to_numpy_uint8(image):
    """
    將各種輸入格式統一轉換為 numpy uint8 [H, W, 3]
    
    支援的輸入格式：
    - numpy array [H, W, 3], uint8, range [0, 255]
    - numpy array [H, W, 3], float32, range [0, 1]
    - torch tensor [C, H, W], float32, range [0, 1]
    """
    # 如果是 torch tensor，先轉成 numpy
    if hasattr(image, 'numpy'):
        # torch tensor: [C, H, W] → [H, W, C]
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0).numpy()
        else:
            image = image.numpy()
    
    # 確保是 numpy array
    image = np.asarray(image)
    
    # 如果是 float [0, 1]，轉成 uint8 [0, 255]
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 確保是 uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    return image


def _apply_morphology(mask, kernel_size=5):
    """
    形態學後處理：去雜點 + 補洞
    
    Args:
        mask: 二值 mask (0 或 255)
        kernel_size: kernel 大小
    
    Returns:
        processed_mask: 處理後的 mask
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    # Opening: 去除小白點（先侵蝕再膨脹）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing: 填補小黑洞（先膨脹再侵蝕）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


# === 額外工具函式 ===

def predict_sky_mask_cv_batch(images, **kwargs):
    """
    批次處理多張圖片
    
    Args:
        images: torch tensor [B, C, H, W] 或 list of images
        **kwargs: 傳給 predict_sky_mask_cv 的參數
    
    Returns:
        pred_masks: numpy array [B, H, W], values {0, 1}
    """
    if hasattr(images, 'shape') and len(images.shape) == 4:
        # torch tensor [B, C, H, W]
        batch_size = images.shape[0]
        masks = []
        for i in range(batch_size):
            mask = predict_sky_mask_cv(images[i], **kwargs)
            masks.append(mask)
        return np.stack(masks, axis=0)
    else:
        # list of images
        masks = [predict_sky_mask_cv(img, **kwargs) for img in images]
        return np.stack(masks, axis=0)


def get_default_params():
    """
    返回預設參數，方便調參時參考
    """
    return {
        'h_min': 90,      # 藍色色相下限 (OpenCV: 0-179)
        'h_max': 130,     # 藍色色相上限
        's_min': 30,      # 飽和度下限（排除灰色）
        's_max': 255,     # 飽和度上限
        'v_min': 120,     # 明度下限（排除暗色）
        'v_max': 255,     # 明度上限
        'morph_kernel_size': 5,  # 形態學 kernel 大小
        'apply_morphology': True  # 是否後處理
    }
