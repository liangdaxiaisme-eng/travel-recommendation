#!/usr/bin/env python3
"""
检查飞猪推荐系统所需的依赖是否已安装
Windows/Linux/Mac 通用
"""

import subprocess
import sys

# 需要的依赖包
REQUIRED_PACKAGES = {
    'flask': 'Flask',
    'pandas': 'pandas', 
    'numpy': 'numpy',
    'torch': 'torch'
}

def check_package(package_name, import_name):
    """检查单个包是否已安装"""
    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 50)
    print("🔍 飞猪推荐系统 - 依赖检查")
    print("=" * 50)
    print()
    
    all_ok = True
    
    for pkg, import_name in REQUIRED_PACKAGES.items():
        ok, error = check_package(pkg, import_name)
        if ok:
            # 获取版本
            try:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', '未知版本')
                print(f"✅ {pkg}: 已安装 ({version})")
            except:
                print(f"✅ {pkg}: 已安装")
        else:
            print(f"❌ {pkg}: 未安装")
            all_ok = False
    
    print()
    print("=" * 50)
    
    if all_ok:
        print("🎉 所有依赖已安装！可以运行系统")
        print()
        print("启动命令：")
        print("  python recommendation_neumf_fixed.py")
        print()
        print("访问地址：http://127.0.0.1:4009")
    else:
        print("⚠️  有依赖未安装，请先安装：")
        print()
        print("安装命令：")
        print("  pip install flask pandas numpy torch")
        print()
        # 如果是 torch 安装失败，给出 CPU 版本建议
        print("如果 PyTorch 安装失败，用 CPU 版本：")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    
    print("=" * 50)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())