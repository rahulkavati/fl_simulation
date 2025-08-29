#!/usr/bin/env python3
"""
Simple dependency installer for the FL simulation project
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {package} installed successfully")
            return True
        else:
            print(f"   ❌ Failed to install {package}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error installing {package}: {e}")
        return False

def main():
    """Install required dependencies"""
    print("🚀 Installing FL Simulation Dependencies")
    print("=" * 40)
    
    # Core dependencies
    core_packages = [
        "numpy",
        "torch", 
        "scikit-learn",
        "pandas"
    ]
    
    print("📋 Installing core packages...")
    success_count = 0
    
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Successfully installed: {success_count}/{len(core_packages)} packages")
    
    if success_count == len(core_packages):
        print("\n🎉 All dependencies installed successfully!")
        print("💡 You can now run: python test_global_update_comprehensive.py")
        return True
    else:
        print(f"\n❌ {len(core_packages) - success_count} packages failed to install")
        print("💡 Try installing manually or check your Python environment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
