import os
import requests
import zipfile
from tqdm import tqdm

# 配置
DATASET_DIR = "Landslide4Sense"
FILES = {
    "TrainData.zip": {"url": "https://zenodo.org/api/records/10463239/files/TrainData.zip/content", "dir": "TrainData"},
    "ValidData.zip": {"url": "https://zenodo.org/api/records/10463239/files/ValidData.zip/content", "dir": "ValidData"},
    "TestData.zip": {"url": "https://zenodo.org/api/records/10463239/files/TestData.zip/content", "dir": "TestData"}
}

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")

def extract_zip(filename, extract_to):
    """解压 ZIP 文件"""
    print(f"正在解压 {filename} 到 {extract_to} ...")
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("解压完成。")
    except zipfile.BadZipFile:
        print(f"❌ 错误: {filename} 不是有效的 zip 文件或已损坏。")
        print("建议删除该文件后重新运行脚本以自动下载。")
    except Exception as e:
        print(f"❌ 解压 {filename} 时出错: {e}")

def main():
    # 确保目录存在
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    os.chdir(DATASET_DIR)
    print(f"进入目录: {os.getcwd()}")

    for filename, info in FILES.items():
        url = info["url"]
        extract_dir = info["dir"]

        # 1. 检查解压后的目录是否存在
        if os.path.exists(extract_dir) and os.path.isdir(extract_dir) and os.listdir(extract_dir):
            print(f"✅ {extract_dir} 目录已存在且不为空，跳过 {filename}。")
            continue

        # 2. 检查 ZIP 文件是否存在
        should_download = True
        if os.path.exists(filename):
            print(f"检查 {filename} 完整性...")
            try:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    if zip_ref.testzip() is None:
                        print(f"✅ {filename} 完整。")
                        should_download = False
                    else:
                        print(f"❌ {filename} 已损坏 (testzip 失败)。")
            except zipfile.BadZipFile:
                print(f"❌ {filename} 已损坏 (不是 zip 文件)。")
            except Exception as e:
                print(f"❌ 检查 {filename} 时出错: {e}")
        
        # 3. 下载 (如果需要)
        if should_download:
            if os.path.exists(filename):
                print(f"删除损坏的文件: {filename}")
                os.remove(filename)
            
            print(f"开始下载 {filename}...")
            try:
                download_file(url, filename)
            except KeyboardInterrupt:
                print("\n下载已取消。")
                if os.path.exists(filename):
                    os.remove(filename) # 清理未完成的文件
                return
            except Exception as e:
                print(f"下载失败: {e}")
                continue

        # 4. 解压
        extract_zip(filename, ".")

if __name__ == "__main__":
    main()
