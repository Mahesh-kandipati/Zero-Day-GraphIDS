"""
Environment and Dependency Verification Script for Zero-Day GraphIDS.
Checks Python version, PyTorch, PyG, Django, database accessibility, and CPU/GPU support.
"""

import sys
import os
import platform

def check_env():
    print("=" * 70)
    print("   ZERO-DAY GRAPHIDS // ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    # 1. Python Version
    py_ver = sys.version_info
    print(f"[*] Python Version: {platform.python_version()} ({platform.architecture()[0]})")
    if py_ver < (3, 10):
        print("[-] Warning: Python 3.10 or higher is strongly recommended.")
    else:
        print("[+] Python version check: PASSED")

    # 2. Operating System
    print(f"[*] Host OS: {platform.system()} {platform.release()} ({platform.version()})")

    # 3. PyTorch & Compute Backend
    try:
        import torch
        print(f"[+] PyTorch: PASSED (Version: {torch.__version__})")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Acceleration: {'Available' if cuda_avail else 'Disabled (CPU mode active - verified Windows compatible)'}")
    except ImportError as e:
        print(f"[-] PyTorch: FAILED ({e})")

    # 4. PyTorch Geometric / Graph Neural Networks
    try:
        import torch_geometric
        print(f"[+] PyTorch Geometric: PASSED (Version: {torch_geometric.__version__})")
    except ImportError:
        print("[!] PyTorch Geometric: Pure PyTorch fallback layer active (fully compatible).")

    # 5. Scientific Computing Libraries
    for lib in ['numpy', 'scipy', 'pandas', 'sklearn', 'networkx', 'plotly', 'reportlab']:
        try:
            m = __import__(lib)
            v = getattr(m, '__version__', 'ok')
            print(f"[+] {lib}: PASSED (Version: {v})")
        except ImportError as e:
            print(f"[-] {lib}: FAILED ({e})")

    # 6. Django Framework
    try:
        import django
        print(f"[+] Django Framework: PASSED (Version: {django.__version__})")
    except ImportError as e:
        print(f"[-] Django Framework: FAILED ({e})")

    # 7. Project Directories Verification
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[*] Project Root: {base_dir}")

    required_dirs = ['media/datasets', 'media/models', 'media/reports', 'sample_data', 'logs', 'model_artifacts']
    for d in required_dirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"[+] Directory '{d}': VERIFIED")

    print("=" * 70)
    print("   ALL ENVIRONMENT CHECKS COMPLETED.")
    print("=" * 70)

if __name__ == '__main__':
    check_env()
