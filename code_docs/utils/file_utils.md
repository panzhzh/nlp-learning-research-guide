# 文件工具 File Utils

> 📁 **多格式文件读写、路径操作和批量处理工具集**

## 📋 功能说明

`file_utils.py` 提供统一的文件操作接口，支持多种文件格式的读写、路径处理、批量操作等功能，为项目提供完善的文件处理能力。

## 🎯 主要功能

### 文件读写功能
- **多格式支持**: JSON、YAML、CSV、Pickle、文本文件
- **编码处理**: 自动处理UTF-8编码和格式转换
- **安全读写**: 完善的异常处理和错误恢复
- **目录创建**: 自动创建必要的目录结构

### 路径操作功能
- **路径管理**: 目录创建、文件复制移动
- **文件信息**: 文件大小、扩展名、存在性检查
- **路径转换**: 扩展名更改、路径规范化
- **文件列表**: 支持通配符的文件搜索

### 图像处理功能
- **图像读写**: PIL图像的加载和保存
- **格式转换**: 图像格式转换和质量控制
- **信息提取**: 图像基本信息获取
- **尺寸调整**: 图像大小调整功能

### 数据集工具功能
- **MR2数据集**: 专门的MR2数据加载函数
- **特征保存**: 处理后特征的保存和加载
- **标注文件**: 检索标注文件路径获取
- **批量处理**: 文件批处理和转换

## 🚀 核心类和函数

### FileUtils 类

#### JSON操作方法
- `read_json(file_path, encoding='utf-8')`: 读取JSON文件
- `write_json(data, file_path, indent=2)`: 写入JSON文件

#### YAML操作方法
- `read_yaml(file_path, encoding='utf-8')`: 读取YAML文件
- `write_yaml(data, file_path)`: 写入YAML文件

#### CSV操作方法
- `read_csv(file_path, encoding='utf-8', **kwargs)`: 读取CSV文件
- `write_csv(df, file_path, index=False)`: 写入CSV文件

#### Pickle操作方法
- `read_pickle(file_path)`: 读取Pickle文件
- `write_pickle(obj, file_path)`: 写入Pickle文件

#### 文本操作方法
- `read_text(file_path, encoding='utf-8')`: 读取文本文件
- `write_text(text, file_path, mode='w')`: 写入文本文件
- `read_lines(file_path, strip=True)`: 按行读取文件
- `write_lines(lines, file_path, add_newline=True)`: 按行写入文件

### PathUtils 类

#### 路径管理方法
- `ensure_dir(dir_path)`: 确保目录存在
- `get_file_size(file_path)`: 获取文件大小
- `get_file_extension(file_path)`: 获取文件扩展名
- `change_file_extension(file_path, new_ext)`: 更改文件扩展名

#### 文件操作方法
- `list_files(dir_path, pattern='*', recursive=False)`: 列出目录文件
- `copy_file(src, dst, create_dirs=True)`: 复制文件
- `move_file(src, dst, create_dirs=True)`: 移动文件
- `delete_file(file_path, ignore_errors=False)`: 删除文件

### ImageUtils 类

#### 图像操作方法
- `load_image(image_path, mode='RGB')`: 加载图像文件
- `save_image(image, save_path, quality=95)`: 保存图像文件
- `get_image_info(image_path)`: 获取图像信息
- `resize_image(image, size, resample)`: 调整图像大小
- `is_valid_image(file_path)`: 检查图像有效性

### DatasetUtils 类

#### 数据集操作方法
- `load_mr2_dataset(data_dir, split)`: 加载MR2数据集
- `save_processed_features(features, data_dir, split)`: 保存处理特征
- `load_processed_features(data_dir, split)`: 加载处理特征
- `get_annotation_file(data_dir, split, item_id, type)`: 获取标注文件路径
- `batch_process_files(file_list, process_func)`: 批量处理文件

## 💡 使用示例

### JSON文件操作
```python
from utils.file_utils import FileUtils

# 读取JSON文件
data = FileUtils.read_json('config/data_configs.yaml')
print(f"加载的配置: {data.keys()}")

# 写入JSON文件
result_data = {'accuracy': 0.85, 'f1_score': 0.82}
FileUtils.write_json(result_data, 'outputs/results.json')
```

### 路径操作
```python
from utils.file_utils import PathUtils

# 确保目录存在
output_dir = PathUtils.ensure_dir('outputs/models')
print(f"输出目录: {output_dir}")

# 获取文件信息
size = PathUtils.get_file_size('data/dataset_items_train.json')
ext = PathUtils.get_file_extension('model.pth')
print(f"文件大小: {size} bytes, 扩展名: {ext}")

# 文件操作
PathUtils.copy_file('source.txt', 'backup/source.txt')
PathUtils.move_file('temp.log', 'logs/temp.log')
```

### 图像文件操作
```python
from utils.file_utils import ImageUtils

# 加载和处理图像
image = ImageUtils.load_image('data/train/img/example.jpg')
resized = ImageUtils.resize_image(image, (224, 224))
ImageUtils.save_image(resized, 'outputs/resized_image.jpg')

# 获取图像信息
info = ImageUtils.get_image_info('data/train/img/example.jpg')
print(f"图像尺寸: {info['width']} x {info['height']}")
print(f"文件大小: {info['file_size_mb']} MB")

# 检查图像有效性
is_valid = ImageUtils.is_valid_image('suspicious_image.jpg')
```

### 数据集操作
```python
from utils.file_utils import DatasetUtils

# 加载MR2数据集
train_data = DatasetUtils.load_mr2_dataset('data', 'train')
print(f"训练集样本数: {len(train_data)}")

# 保存和加载特征
features = {'item_1': [0.1, 0.2, 0.3], 'item_2': [0.4, 0.5, 0.6]}
DatasetUtils.save_processed_features(features, 'data', 'train', 'text_features')
loaded_features = DatasetUtils.load_processed_features('data', 'train', 'text_features')

# 获取标注文件路径
direct_file = DatasetUtils.get_annotation_file('data', 'train', 'item_123', 'direct')
inverse_file = DatasetUtils.get_annotation_file('data', 'train', 'item_123', 'inverse')
```

### 批量文件处理
```python
from utils.file_utils import DatasetUtils, PathUtils

# 获取所有图像文件
image_files = PathUtils.list_files('data/train/img', '*.jpg', recursive=False)

# 定义处理函数
def process_image(image_path):
    info = ImageUtils.get_image_info(image_path)
    return {'path': str(image_path), 'size': info['file_size']}

# 批量处理
results = DatasetUtils.batch_process_files(image_files, process_image)
print(f"处理了 {len(results)} 个图像文件")
```

## 📦 便捷函数

### 配置文件加载
```python
from utils.file_utils import load_config

# 自动识别格式并加载
config = load_config('config/data_configs.yaml')  # YAML
config = load_config('config/settings.json')     # JSON
```

### 结果保存
```python
from utils.file_utils import save_results

# 自动识别格式并保存
results = {'accuracy': 0.85, 'loss': 0.23}
save_results(results, 'outputs/experiment_results.json')    # JSON
save_results(results, 'outputs/experiment_results.yaml')    # YAML
save_results(results, 'outputs/experiment_results.pkl')     # Pickle
```

## 🔧 高级功能

### 文件格式自动检测
```python
# 根据扩展名自动选择读取方法
def load_data_file(file_path):
    ext = PathUtils.get_file_extension(file_path).lower()
    
    if ext == '.json':
        return FileUtils.read_json(file_path)
    elif ext in ['.yaml', '.yml']:
        return FileUtils.read_yaml(file_path)
    elif ext == '.csv':
        return FileUtils.read_csv(file_path)
    elif ext in ['.pkl', '.pickle']:
        return FileUtils.read_pickle(file_path)
    else:
        return FileUtils.read_text(file_path)
```

### 批量格式转换
```python
# 批量转换JSON到YAML
json_files = PathUtils.list_files('configs', '*.json')
for json_file in json_files:
    data = FileUtils.read_json(json_file)
    yaml_file = PathUtils.change_file_extension(json_file, '.yaml')
    FileUtils.write_yaml(data, yaml_file)
```

### 安全文件操作
```python
# 带备份的文件写入
def safe_write_json(data, file_path):
    if PathUtils.get_file_size(file_path) > 0:  # 文件存在且非空
        backup_path = file_path + '.backup'
        PathUtils.copy_file(file_path, backup_path)
    
    try:
        FileUtils.write_json(data, file_path)
    except Exception as e:
        print(f"写入失败，恢复备份: {e}")
        if backup_path.exists():
            PathUtils.move_file(backup_path, file_path)
        raise
```

## ⚠️ 重要说明

### 文件编码
- **默认编码**: 所有文本文件使用UTF-8编码
- **编码检测**: 自动处理编码问题和转换
- **特殊字符**: 正确处理中文和特殊字符

### 错误处理
- **异常捕获**: 完善的异常处理机制
- **错误恢复**: 提供错误恢复和重试机制
- **详细日志**: 记录详细的操作日志和错误信息

### 性能优化
- **大文件处理**: 对大文件使用流式处理
- **内存管理**: 及时释放不需要的文件对象
- **批量操作**: 批量操作比单个操作更高效

### 安全考虑
- **路径验证**: 防止路径遍历攻击
- **权限检查**: 检查文件读写权限
- **备份机制**: 重要操作前自动备份

---

**[⬅️ 配置管理器](config_manager.md) | [模型库模块 ➡️](../models/README.md)**
