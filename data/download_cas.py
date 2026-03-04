import os
import requests
import zipfile
import shutil
import time
from tqdm import tqdm

# 配置
ZENODO_RECORD_ID = "10294997"
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DATASET_DIR = "CAS_Landslide"

def download_file(url, filename, retries=5):
    temp_filename = filename + ".part"
    for attempt in range(retries):
        try:
            resume_header = {}
            mode = 'wb'
            initial_pos = 0
            
            # 检查是否有未完成的下载
            if os.path.exists(temp_filename):
                initial_pos = os.path.getsize(temp_filename)
                resume_header = {'Range': f'bytes={initial_pos}-'}
                mode = 'ab'
                print(f"检测到断点，从 {initial_pos} 字节继续下载...")

            response = requests.get(url, stream=True, timeout=60, headers=resume_header)
            
            if response.status_code == 200:
                if initial_pos > 0:
                    print("服务器不支持断点续传，重新开始下载...")
                    initial_pos = 0
                    mode = 'wb'
            elif response.status_code == 206:
                pass # 支持断点续传
            elif response.status_code == 416: 
                 print("文件可能已下载完成 (Range 错误).")
                 if os.path.exists(temp_filename):
                     os.rename(temp_filename, filename)
                 return True
            else:
                response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0)) + initial_pos
            block_size = 1024 * 8
            
            with open(temp_filename, mode) as file, tqdm(
                desc=filename,
                total=total_size,
                initial=initial_pos,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            
            os.rename(temp_filename, filename)
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout) as e:
            print(f"\n⚠️ 下载 {filename} 失败 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(5)
        except Exception as e:
            print(f"\n❌ 下载 {filename} 发生严重错误: {e}")
            return False
    return False

def extract_zip(zip_path, extract_to):
    print(f"解压 {zip_path} 到 {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ 解压完成: {zip_path}")
        
        # 创建完成标记
        with open(zip_path + ".done", 'w') as f:
            f.write("done")
            
        # 解压后删除 zip 文件以节省空间
        os.remove(zip_path)
    except zipfile.BadZipFile:
        print(f"❌ 解压失败: {zip_path} 不是有效的 zip 文件")
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"❌ 解压出错: {e}")

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    os.chdir(DATASET_DIR)
    print(f"进入目录: {os.getcwd()}")
    
    print(f"正在获取 Zenodo Record {ZENODO_RECORD_ID} 的文件列表...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        files = data.get('files', [])
        
        print(f"找到 {len(files)} 个文件。")
        
        for file_info in files:
            filename = file_info['key']
            url = file_info['links']['self']
            size = file_info['size']
            
            # 检查完成标记
            if os.path.exists(filename + ".done"):
                print(f"文件 {filename} 已处理完成 (存在标记)，跳过。")
                continue

            # 仅下载 zip 文件和 shp 相关文件
            if not filename.endswith('.zip') and not filename.endswith('.html'):
                 pass

            if os.path.exists(filename):
                print(f"文件 {filename} 已存在，验证中...")
                if filename.endswith('.zip'):
                     # 检查大小
                     if os.path.getsize(filename) == size:
                         print(f"大小匹配，尝试解压...")
                         try:
                            with zipfile.ZipFile(filename, 'r') as zip_ref:
                                if zip_ref.testzip() is None:
                                    extract_zip(filename, ".")
                                    continue
                                else:
                                    print(f"⚠️ {filename} 已损坏，重新下载。")
                                    os.remove(filename)
                         except zipfile.BadZipFile:
                             print(f"⚠️ {filename} 已损坏，重新下载。")
                             os.remove(filename)
                         except Exception:
                             continue 
                     else:
                         print(f"大小不匹配，重新下载/续传。")
                else:
                    continue
            
            # 检查是否可能已经解压了 (仅针对特定目录结构的检查，作为后备)
            possible_dir = filename.replace('.zip', '')
            if os.path.isdir(possible_dir) and os.listdir(possible_dir):
                print(f"目录 {possible_dir} 已存在且不为空，视为完成。")
                with open(filename + ".done", 'w') as f: f.write("done")
                continue
            
            print(f"准备下载: {filename}")
            if download_file(url, filename):
                if filename.endswith('.zip'):
                    extract_zip(filename, ".")
            else:
                print(f"❌ 跳过 {filename} (下载失败)")

        print("\n✅ CAS Landslide Dataset 处理完成！")

    except Exception as e:
        print(f"获取元数据失败: {e}")

if __name__ == "__main__":
    main()
