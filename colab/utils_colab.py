"""
Colab 專屬工具函數
包含環境偵測、device 解析等共用功能
"""

import os
import torch


def detect_environment():
    """偵測是否在 Colab 環境"""
    # 方法1：檢查環境變數
    if os.environ.get('COLAB_GPU') is not None:
        return 'colab'
    # 方法2：檢查當前工作目錄是否包含 /content
    if '/content' in os.getcwd():
        return 'colab'
    return 'local'


def get_default_device(env=None):
    """根據環境返回預設 device"""
    if env is None:
        env = detect_environment()
    
    if env == 'colab':
        # Colab 環境：如果有 GPU 可用，預設用 GPU
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        # 本機環境：預設用 CPU（避免本機 GPU 問題）
        return 'cpu'


def get_default_num_workers(env=None):
    """根據環境返回預設 num_workers"""
    if env is None:
        env = detect_environment()
    
    if env == 'colab':
        # Colab 環境：可用多進程
        return 2
    else:
        # 本機環境（Windows）：建議設為 0
        return 0


def resolve_device(device_arg, env=None):
    """解析 device 參數"""
    if env is None:
        env = detect_environment()
    
    if device_arg == 'auto':
        # auto：自動偵測（有 GPU 用 GPU，否則用 CPU）
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_arg in ['cpu', 'cuda']:
        # 明確指定
        return device_arg
    elif device_arg is None:
        # 未指定：使用環境預設值
        return get_default_device(env)
    else:
        # 其他情況：使用環境預設值
        return get_default_device(env)


def setup_device(device_arg=None, env=None):
    """設置 device 並返回 torch.device 和 use_amp"""
    if env is None:
        env = detect_environment()
    
    device_str = resolve_device(device_arg, env)
    
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            print(f"[警告] 指定使用 GPU 但 GPU 不可用，改用 CPU")
            device_str = 'cpu'
    
    device = torch.device(device_str)
    use_amp = (device.type == 'cuda')
    
    return device, use_amp, env
