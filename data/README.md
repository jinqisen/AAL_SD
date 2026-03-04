# 数据集下载与准备指南

本项目需要以下三个主要数据集。请按照以下说明下载数据，并将其放置在相应的目录中。

## 1. Landslide4Sense (L4S)
**用途**: 核心训练与测试集。
**目录**: `Data/Landslide4Sense/`

### 下载方式
- **自动下载 (推荐)**:
  本项目提供了自动下载脚本，直接运行即可从 Zenodo 拉取数据并解压：
  ```bash
  python download_l4s.py
  ```
  *(注：数据总量约 3GB，请确保网络通畅)*

- **手动下载**: [Zenodo Repository](https://zenodo.org/records/10463239)

### 文件放置
下载并解压后，请确保目录结构如下：
```
Data/Landslide4Sense/
├── TrainData/
│   ├── img/
│   └── mask/
├── ValidData/
│   ├── img/
│   └── mask/
└── TestData/
    └── img/
```

---

## 2. CAS Landslide Dataset
**用途**: 辅助训练与测试集。
**目录**: `Data/CAS_Landslide/`

### 下载方式
- **自动下载 (推荐)**:
  我们已更新下载策略，现在可以直接从 Zenodo 自动下载：
  ```bash
  python download_cas.py
  ```
  *(脚本会自动从 Zenodo 获取所有子数据集 zip 包并解压)*

- **手动下载 (备选)**:
  - **Zenodo**: [https://zenodo.org/records/10294997](https://zenodo.org/records/10294997)
  - **百度网盘**: [链接](https://pan.baidu.com/s/1ofM0u0vYcB_gLEjVvrYoCw) (提取码: `i069`)

- **文件结构**:
  解压后，`Data/CAS_Landslide/` 应包含多个区域子目录（如 `Wenchuan`, `Lombok` 等）。

---

## 3. NIH Chest X-ray 14
**用途**: 泛化能力验证集。
**目录**: `Data/ChestX-ray14/`

### 下载方式
- **自动下载 (推荐)**:
  `download_chestxray.py` 脚本已完全集成 Kaggle API 下载与自动整理功能。
  由于数据集巨大 (45GB)，脚本会自动下载 `data.zip`，解压并整理分散的图片目录到标准的 `images/` 结构。
  ```bash
  python download_chestxray.py
  ```
  *(注：必须配置 Kaggle API (~/.kaggle/kaggle.json)。下载过程支持断点续传。)*

- **手动下载**: [NIH Clinical Center Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)
  1. 点击链接进入 NIH Box 页面。
  2. 下载所有文件。
  3. 将下载的 `images_*.zip` 和 `*.csv/txt` 文件放入 `Data/ChestX-ray14/` 目录。
  4. 解压 `images_*.zip` 到 `Data/ChestX-ray14/images/`。

- **Kaggle 下载 (备选)**:
  如果想直接使用命令行工具：
  ```bash
  kaggle datasets download -d nih-chest-xrays/data -p Data/ChestX-ray14
  ```
  *(下载后需解压并手动整理目录结构)*

### 文件放置
```
Data/ChestX-ray14/
├── images/
│   ├── 00000001_000.png
│   ├── ...
└── Data_Entry_2017.csv
```
