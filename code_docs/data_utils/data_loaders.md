# 数据加载器 Data Loaders

> 🔄 **严格验证的PyTorch数据加载器，强制使用真实数据集**

## 📋 模块概览

`data_loaders.py`提供严格验证的数据加载器实现，不支持演示数据，所有功能都要求使用真实的MR2数据集。

## 🚀 核心类

### StrictDataLoaderConfig
严格的数据加载器配置类，自动验证数据要求：

```python
class StrictDataLoaderConfig:
    def __init__(self):
        # 自动检查数据要求
        check_data_requirements()
        self.config = self.load_config()
```

**特点**：
- 初始化时强制验证数据完整性
- 从配置文件自动加载参数
- 提供默认的安全配置
- 集成配置管理器

### StrictCollateFunction  
严格的批处理函数，确保数据质量：

```python
class StrictCollateFunction:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 验证批次数据完整性
        # 处理缺失图像情况
        # 创建标准张量格式
```

**功能**：
- 验证批次数据不为空
- 检查必要字段存在性
- 自动处理缺失图像的情况
- 创建统一的tensor格式

## 🔧 主要函数

### create_strict_dataloader
创建单个严格验证的数据加载器：

```python
def create_strict_dataloader(
    split: str = 'train',
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 4
) -> DataLoader:
    """
    创建严格的数据加载器
    
    Args:
        split: 数据划分 ('train', 'val', 'test')
        batch_size: 批次大小（None时使用配置默认值）
        shuffle: 是否打乱（None时训练集=True，其他=False）
        num_workers: 工作进程数
        
    Returns:
        严格验证的DataLoader对象
        
    Raises:
        FileNotFoundError: 数据文件不存在
        ValueError: 数据验证失败
        RuntimeError: 创建失败
    """
```

### create_all_dataloaders
创建所有数据加载器的批量函数：

```python
def create_all_dataloaders(
    batch_sizes: Optional[Dict[str, int]] = None
) -> Dict[str, DataLoader]:
    """
    创建所有数据加载器
    
    Args:
        batch_sizes: 各数据集的批次大小
                    默认: {'train': 32, 'val': 64, 'test': 64}
    
    Returns:
        数据加载器字典 {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        
    Raises:
        RuntimeError: 任何数据加载失败都会抛出详细错误信息
    """
```

## 📊 批处理数据格式

### 批处理输出格式
```python
# StrictCollateFunction输出的标准格式
batch_data = {
    # 基础字段
    'item_id': List[str],          # 数据项ID列表
    'text': List[str],             # 文本列表
    'caption': List[str],          # 兼容性文本字段
    'label': List[int],            # 原始标签列表
    
    # 张量字段  
    'labels': torch.Tensor,        # 标签张量 (batch_size,)
    'images': torch.Tensor,        # 图像张量 (batch_size, 3, H, W)
    
    # 元数据
    'has_image': List[bool],       # 图像有效性列表
    'image_path': List[str],       # 图像路径列表
    'text_length': List[int],      # 文本长度列表
    'token_count': List[int]       # 词数统计列表
}
```

### 缺失图像处理
```python
# 当图像不存在或加载失败时
if 'image' not in item or item['image'] is None:
    # 自动创建零张量
    images.append(torch.zeros(3, 224, 224))
else:
    images.append(item['image'])

# 批处理时堆叠所有图像
batch_data['images'] = torch.stack(images)
```

## 🎯 使用示例

### 创建单个数据加载器
```python
from data_utils.data_loaders import create_strict_dataloader

try:
    # 创建训练集数据加载器
    train_loader = create_strict_dataloader(
        split='train',
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    print(f"✅ 训练集加载器创建成功")
    print(f"   数据集大小: {len(train_loader.dataset)}")
    print(f"   批次数量: {len(train_loader)}")
    print(f"   批次大小: {train_loader.batch_size}")
    
except FileNotFoundError as e:
    print(f"❌ 数据文件不存在: {e}")
except ValueError as e:
    print(f"❌ 数据验证失败: {e}")
```

### 创建所有数据加载器
```python
from data_utils.data_loaders import create_all_dataloaders

try:
    # 创建所有数据加载器
    dataloaders = create_all_dataloaders(
        batch_sizes={'train': 16, 'val': 32, 'test': 32}
    )
    
    # 访问各个数据加载器
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    print("✅ 所有数据加载器创建成功")
    for split, loader in dataloaders.items():
        print(f"   {split}: {len(loader.dataset)} 样本, 批次大小 {loader.batch_size}")
        
except RuntimeError as e:
    print(f"❌ 数据加载器创建失败: {e}")
    # 错误信息包含详细的解决方案
```

### 批次数据迭代
```python
# 迭代数据加载器
for batch_idx, batch in enumerate(train_loader):
    print(f"批次 {batch_idx}:")
    print(f"  数据键: {list(batch.keys())}")
    print(f"  批次大小: {len(batch['labels'])}")
    print(f"  标签: {batch['labels']}")
    print(f"  文本样例: {batch['text'][0][:50]}...")
    print(f"  图像形状: {batch['images'].shape}")
    print(f"  有效图像数: {sum(batch.get('has_image', []))}")
    
    if batch_idx >= 2:  # 只显示前3个批次
        break
```

## 🧪 测试和验证

### test_dataloader 函数
内置的数据加载器测试函数：

```python
from data_utils.data_loaders import test_dataloader

# 测试数据加载器
test_dataloader(train_loader, max_batches=3)
```

**验证内容**：
- 批次数据类型检查
- 必要字段存在性验证
- 张量形状和类型验证
- 批次大小一致性检查
- 数据内容合理性验证

### 测试输出示例
```
🧪 测试数据加载器 (最多 3 个批次)
  批次 0:
    数据键: ['item_id', 'text', 'caption', 'label', 'labels', 'images', 'has_image']
    标签: tensor([1, 0, 2, 1, 0, 2, 1, 0])
    批次大小: 8
    文本样例: 这是一个测试文本 This is a test text...
    图像形状: torch.Size([8, 3, 224, 224])
  批次 1:
    ...
✅ 数据加载器测试通过
```

## ⚡ 性能优化

### 配置优化建议
```python
# 根据硬件调整参数
optimal_config = {
    # CPU核心数决定worker数量
    'num_workers': min(4, os.cpu_count()),
    
    # GPU内存决定批次大小
    'batch_size': 32 if torch.cuda.is_available() else 16,
    
    # 启用内存固定（GPU训练时）
    'pin_memory': torch.cuda.is_available(),
    
    # 持久化workers（减少进程创建开销）
    'persistent_workers': True
}
```

### 内存管理
```python
# 内存不足时的优化策略
low_memory_config = {
    'batch_size': 8,          # 减小批次大小
    'num_workers': 1,         # 减少worker进程
    'pin_memory': False,      # 关闭内存固定
    'drop_last': True         # 丢弃不完整批次
}
```

## 🔧 自定义配置

### 自定义批处理函数
```python
from data_utils.data_loaders import StrictCollateFunction

class CustomCollateFunction(StrictCollateFunction):
    def __call__(self, batch):
        # 调用父类方法获取基础批处理结果
        batch_data = super().__call__(batch)
        
        # 添加自定义字段
        batch_data['custom_field'] = [item.get('custom', None) for item in batch]
        
        return batch_data

# 使用自定义批处理函数
custom_loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=CustomCollateFunction()
)
```

### 配置文件集成
```python
# 从配置文件获取参数
from utils.config_manager import get_data_config

data_config = get_data_config()
dataloader_config = data_config.get('dataloader', {})

train_config = dataloader_config.get('train', {})
batch_size = train_config.get('batch_size', 32)
num_workers = train_config.get('num_workers', 4)
```

## 🚨 错误处理

### 常见错误类型
```python
# 1. 数据文件不存在
FileNotFoundError: "❌ 数据集文件不存在: data/dataset_items_train.json"
# 解决: 下载MR2数据集并解压到正确位置

# 2. 数据验证失败  
ValueError: "❌ train 数据集样本数不足: 5 < 10"
# 解决: 确保数据集包含足够的样本

# 3. 批次数据为空
ValueError: "❌ 批次数据为空，无法进行批处理"
# 解决: 检查数据集是否正确加载

# 4. 标签转换失败
ValueError: "❌ 标签转换失败: invalid literal for int()"
# 解决: 检查标签字段格式
```

### 完整错误处理示例
```python
try:
    # 创建数据加载器
    dataloaders = create_all_dataloaders()
    
    # 测试数据加载
    for split, loader in dataloaders.items():
        test_dataloader(loader, max_batches=1)
    
    print("✅ 所有数据加载器验证通过")
    
except FileNotFoundError as e:
    print(f"❌ 文件错误: {e}")
    print("解决方案:")
    print("1. 下载MR2数据集")
    print("2. 解压到项目根目录/data")
    print("3. 确保包含所有JSON文件")
    
except ValueError as e:
    print(f"❌ 验证错误: {e}")
    print("解决方案:")
    print("1. 检查数据文件完整性")
    print("2. 验证JSON格式正确性")
    print("3. 确保样本数量足够")
    
except RuntimeError as e:
    print(f"❌ 运行时错误: {e}")
    print("解决方案:")
    print("1. 检查系统内存")
    print("2. 减少batch_size和num_workers")
    print("3. 重启Python进程")
```

## 🔄 向后兼容

### 兼容性函数
```python
# 为了向后兼容，保留原有函数名
def create_simple_dataloader(*args, **kwargs):
    """向后兼容的函数名"""
    return create_strict_dataloader(*args, **kwargs)

def create_mr2_dataloaders(*args, **kwargs):
    """向后兼容的函数名"""
    return create_all_dataloaders(*args, **kwargs)
```

### 模块导入兼容性
```python
# __init__.py 中的智能导入
try:
    from .data_loaders import create_all_dataloaders, StrictDataLoaderConfig
    create_mr2_dataloaders = create_all_dataloaders
    DataLoaderFactory = StrictDataLoaderConfig
except ImportError as e:
    print(f"❌ 导入数据加载器失败: {e}")
    create_all_dataloaders = None
```

## 💡 最佳实践

### 开发阶段
```python
# 开发时使用较小的配置快速迭代
dev_dataloaders = create_all_dataloaders(
    batch_sizes={'train': 8, 'val': 16, 'test': 16}
)
```

### 生产阶段  
```python
# 生产时使用优化的配置
prod_dataloaders = create_all_dataloaders(
    batch_sizes={'train': 32, 'val': 64, 'test': 64}
)
```

### 调试技巧
```python
# 1. 先测试单个样本
dataset = MR2Dataset(data_dir='data', split='train')
sample = dataset[0]
print("单个样本测试通过")

# 2. 再测试小批次
small_loader = create_strict_dataloader(split='train', batch_size=2)
batch = next(iter(small_loader))
print("小批次测试通过")

# 3. 最后测试完整配置
full_loader = create_strict_dataloader(split='train', batch_size=32)
test_dataloader(full_loader, max_batches=3)
```

---

**[⬅️ MR2数据集](code_docs/data_loaders/mr2_dataset.md) | [数据分析 ➡️](code_docs/data_loaders/mr2_analysis.md)**