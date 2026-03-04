import os
import subprocess
import sys
import shutil
import zipfile

# 配置
DATASET_DIR = "ChestX-ray14"
IMAGES_DIR = "images"

def check_existing_images():
    if not os.path.exists(IMAGES_DIR):
        return 0
    return len([name for name in os.listdir(IMAGES_DIR) if name.lower().endswith('.png')])

def move_images_to_root():
    print("正在整理文件结构，将分散的图片移动到 images 目录...")
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    
    # Kaggle 数据集结构通常是 images_001/images/xxx.png
    # 查找所有以 images_ 开头的目录
    subdirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('images_')]
    
    moved_count = 0
    for subdir in subdirs:
        # 检查子目录中是否有 images 文件夹
        source_images_path = os.path.join(subdir, 'images')
        if os.path.exists(source_images_path) and os.path.isdir(source_images_path):
            print(f"正在处理 {subdir} ...")
            for filename in os.listdir(source_images_path):
                if filename.lower().endswith('.png'):
                    src = os.path.join(source_images_path, filename)
                    dst = os.path.join(IMAGES_DIR, filename)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                        moved_count += 1
            
            # 清理空的子目录
            try:
                os.rmdir(source_images_path)
                os.rmdir(subdir)
            except OSError:
                pass # 目录非空则忽略
        else:
             # 有可能图片直接在 subdir 里 (虽然不太可能，做个防御性编程)
             pass

    print(f"✅ 文件整理完成，移动了 {moved_count} 张图片。")

import time

def download_via_kaggle():
    print("\n" + "="*60)
    print("启动 Kaggle 自动下载 (nih-chest-xrays/data)...")
    print("="*60)
    
    # 检查 API Key
    kaggle_config_dir = os.path.expanduser("~/.kaggle")
    kaggle_key_file = os.path.join(kaggle_config_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_key_file):
        print("❌ 未找到 Kaggle API Key 配置 (~/.kaggle/kaggle.json)。")
        return False

    zip_file = "data.zip"
    
    # 1. 下载 (带重试机制)
    max_retries = 10
    download_success = False
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(zip_file):
                print(f"检测到 {zip_file}，尝试断点续传 (尝试 {attempt+1}/{max_retries})...")
            else:
                print(f"开始下载 (尝试 {attempt+1}/{max_retries})...")
            
            # Kaggle CLI 默认支持断点续传
            cmd = ["kaggle", "datasets", "download", "-d", "nih-chest-xrays/data", "-p", "."]
            subprocess.run(cmd, check=True)
            
            # 验证文件完整性
            if os.path.exists(zip_file):
                if not zipfile.is_zipfile(zip_file):
                    print(f"⚠️ 下载的 {zip_file} 文件损坏或不完整。")
                    print("删除损坏文件并重新下载...")
                    os.remove(zip_file)
                    # 触发异常处理逻辑进入下一次重试
                    raise subprocess.CalledProcessError(1, cmd)
            
            download_success = True
            break # 下载成功且文件有效，跳出循环
            
        except subprocess.CalledProcessError:
            print(f"⚠️ 下载中断或失败 (尝试 {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = 10
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("❌ 达到最大重试次数，下载失败。")
    
    if not download_success:
        return False

    try:
        # 2. 解压
        if os.path.exists(zip_file):
            print(f"正在解压 {zip_file} (这可能需要几分钟)...")
            # 使用 unzip 命令如果可用，否则用 python zipfile
            try:
                subprocess.run(["unzip", "-q", "-o", zip_file], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("unzip 命令不可用或失败，使用 Python zipfile 解压...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(".")
            
            print("解压完成，删除压缩包以释放空间...")
            os.remove(zip_file)
        
        # 3. 整理文件
        move_images_to_root()
        
        return True

    except Exception as e:
        print(f"❌ 发生错误: {e}")
    
    return False

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    os.chdir(DATASET_DIR)
    print(f"工作目录: {os.getcwd()}")
    
    # 1. 检查现有图片
    count = check_existing_images()
    print(f"当前 images 目录下共有 {count} 张图片。")
    
    if count > 100000: # 总共约 112,120 张
        print("✅ 数据集似乎已完整下载。")
        return

    # 2. 检查是否有未整理的文件夹
    subdirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('images_')]
    if subdirs:
        print(f"检测到 {len(subdirs)} 个未整理的子目录，开始整理...")
        move_images_to_root()
        # 再次检查
        if check_existing_images() > 100000:
            print("✅ 整理后检查，数据集已完整。")
            return

    # 3. 下载
    print("准备开始下载...")
    download_via_kaggle()

if __name__ == "__main__":
    main()
