import os
import sys
import subprocess
import importlib.util

def check_package(display_name, import_name):
    """检查单个包是否存在并打印状态"""
    try:
        if importlib.util.find_spec(import_name) is not None:
            print(f"✓ {display_name} 已安装")
            return True
    except ImportError:
        pass
    except Exception:
        # 忽略部分导入错误
        pass
    
    print(f"❌ {display_name} 未安装")
    return False

def check_and_install_requirements():
    """检查依赖并从清华源安装"""
    
    # 1. 询问用户
    choice = input("是否检查并安装依赖包? (y/n, 默认: y): ").strip().lower()
    if choice and choice != 'y':
        print("已跳过依赖检查。")
        return

    print("检查依赖环境...")
    
    # 定义需要检查的库
    # 格式: (显示名称, 导入名称, pip安装包名)
    requirements = [
        ("torch", "torch", "torch"),
        ("torchvision", "torchvision", "torchvision"),
        ("ultralytics", "ultralytics", "ultralytics"),
        ("opencv-python", "cv2", "opencv-python-headless"), # 服务器环境推荐 headless 版本避免依赖 X11
        ("Pillow", "PIL", "Pillow"),
        ("numpy", "numpy", "numpy"),
        ("matplotlib", "matplotlib", "matplotlib"),
        ("streamlit", "streamlit", "streamlit"), # 必须检查
        ("plotly", "plotly", "plotly") 
    ]
    
    packages_to_install = []
    
    # 2. 遍历检查
    for display_name, import_name, package_name in requirements:
        if not check_package(display_name, import_name):
            packages_to_install.append(package_name)
            
    # 3. 安装缺失项
    if packages_to_install:
        print(f"\n检测到 {len(packages_to_install)} 个缺失的库，准备从清华源镜像安装...")
        print(f"缺失列表: {', '.join(packages_to_install)}")
        print("-" * 50)
        
        # 清华源地址
        mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"
        
        # 构建 pip 命令
        cmd = [sys.executable, "-m", "pip", "install"] + packages_to_install + ["-i", mirror_url]
        
        try:
            print(f"正在执行安装命令...")
            subprocess.check_call(cmd)
            print("\n✅ 所有依赖安装完成！")
        except subprocess.CalledProcessError:
            print("\n❌ 安装失败。建议手动运行以下命令安装:")
            print(f"pip install {' '.join(packages_to_install)} -i {mirror_url}")
            sys.exit(1)
    else:
        print("\n✅ 环境完美，所有依赖已就绪！")
    print("="*60)

def run_app():
    """启动 Streamlit 应用"""
    print("\n" + "="*60)
    print("🦷 牙科 AI 教学平台正在启动...")
    print("="*60)
    
    # 获取 web_ui.py 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "web_ui.py")
    
    if not os.path.exists(app_path):
        print(f"❌ 错误: 找不到 {app_path}")
        print("请确保 web_ui.py 与 run.py 在同一目录下")
        return

    # 默认端口
    port = "8501"
    
    print(f"🚀 服务即将启动！")
    print(f"🌍 请在浏览器访问提供的 Network URL (通常是 http://<云服务器IP>:{port})")
    print("="*60)
    
    # 启动 Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", port, "--server.address", "0.0.0.0"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")

if __name__ == "__main__":
    try:
        check_and_install_requirements()
        run_app()
    except KeyboardInterrupt:
        print("\n程序已退出")