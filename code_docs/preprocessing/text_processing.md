# 文本处理器 Text Processing

> 📝 **中英文混合文本的分词、清洗、标准化处理工具**

## 📋 功能说明

`TextProcessor` 专门处理MR2数据集中的中英文混合文本，提供分词、清洗、特征提取等完整的文本预处理功能。

## 🎯 主要功能

### 语言处理功能
- **语言检测**: 自动识别中文、英文或混合文本
- **中文分词**: 基于jieba的智能中文分词
- **英文处理**: 基于NLTK的英文分词和词形处理
- **混合处理**: 中英文混合文本的联合处理

### 文本清洗功能
- **URL移除**: 清理HTTP链接和网址
- **提及清理**: 移除@用户名提及
- **话题标签**: 可选的#话题标签处理
- **表情符号**: emoji转文本描述或移除
- **HTML清理**: 移除HTML标签和实体

### 特征提取功能
- **基础特征**: 文本长度、词数、字符数
- **语言特征**: 语言类型、大写比例
- **标点统计**: 感叹号、问号等标点符号计数
- **内容特征**: URL数量、提及数量、emoji数量

## 🚀 核心类和方法

### TextProcessor 类

#### 初始化方法
```python
TextProcessor(language='mixed')
```

**参数说明:**
- `language`: 语言类型 ('chinese', 'english', 'mixed')

#### 主要方法

##### 语言检测
- `detect_language(text)`: 检测文本语言类型
- 返回: 'chinese', 'english', 'mixed', 'unknown'

##### 文本清洗
- `clean_text(text)`: 综合文本清洗
- 移除URL、提及、HTML标签等噪声
- 标准化空白字符和标点符号

##### 分词功能
- `tokenize(text, language=None)`: 统一分词接口
- `tokenize_chinese(text)`: 中文专用分词
- `tokenize_english(text)`: 英文专用分词
- `tokenize_mixed(text)`: 混合语言分词

##### 特征提取
- `extract_features(text)`: 提取完整文本特征
- `preprocess_batch(texts)`: 批量文本预处理

## 📦 处理配置

### 文本清洗配置
- `remove_urls`: 是否移除URL (默认True)
- `remove_mentions`: 是否移除@提及 (默认True)
- `remove_hashtags`: 是否移除#标签 (默认False)
- `normalize_whitespace`: 是否标准化空白 (默认True)

### 分词配置
- **中文配置**: jieba分词、自定义词典
- **英文配置**: NLTK分词、停用词过滤、词干提取

## 📊 返回数据格式

### 分词结果
```python
tokens = processor.tokenize(text)
# 返回: List[str] - 分词结果列表
```

### 特征提取结果
```python
features = processor.extract_features(text)
# 返回字典:
{
    'text_length': int,           # 文本长度
    'word_count': int,            # 词数
    'char_count': int,            # 字符数
    'language': str,              # 语言类型
    'tokens': List[str],          # 分词结果
    'token_count': int,           # token数量
    'exclamation_count': int,     # 感叹号数量
    'question_count': int,        # 问号数量
    'uppercase_ratio': float,     # 大写字母比例
    'digit_count': int,           # 数字字符数
    'url_count': int,             # URL数量
    'mention_count': int,         # 提及数量
    'hashtag_count': int,         # 话题标签数量
    'emoji_count': int            # emoji数量
}
```

### 批量处理结果
```python
results = processor.preprocess_batch(texts)
# 返回: List[Dict] - 每个文本的处理结果
[
    {
        'original_text': str,     # 原始文本
        'cleaned_text': str,      # 清洗后文本
        'tokens': List[str],      # 分词结果
        'features': Dict          # 特征字典
    },
    ...
]
```

## 💡 使用示例

### 基础文本处理
```python
from preprocessing import TextProcessor

# 创建处理器
processor = TextProcessor(language='mixed')

# 文本清洗
text = "这是测试文本 This is test! @user #topic http://example.com"
cleaned = processor.clean_text(text)
print(f"清洗后: {cleaned}")

# 语言检测
language = processor.detect_language(text)
print(f"语言类型: {language}")

# 分词处理
tokens = processor.tokenize(text)
print(f"分词结果: {tokens}")
```

### 特征提取
```python
# 提取文本特征
features = processor.extract_features(text)
print(f"文本长度: {features['text_length']}")
print(f"词数: {features['word_count']}")
print(f"语言: {features['language']}")
print(f"URL数量: {features['url_count']}")
```

### 批量处理
```python
# 批量处理多个文本
texts = [
    "这是第一个测试文本",
    "This is the second test text!",
    "混合语言文本 mixed language text"
]

results = processor.preprocess_batch(texts)
for i, result in enumerate(results):
    print(f"文本{i+1}: {len(result['tokens'])} 个tokens")
```

### 自定义配置
```python
# 自定义清洗配置
processor = TextProcessor(language='chinese')
processor.remove_hashtags = True  # 移除话题标签
processor.remove_urls = False     # 保留URL

cleaned = processor.clean_text(text)
```

## 🔧 配置和优化

### 中文处理优化
- **自定义词典**: 添加领域相关词汇
- **停用词扩展**: 根据任务调整停用词列表
- **分词精度**: 调整jieba分词参数

### 英文处理优化
- **词形还原**: 启用lemmatization提高准确性
- **词干提取**: 使用stemming减少词汇变形
- **停用词过滤**: 根据任务需求调整停用词

### 性能优化
- **批量处理**: 使用batch方法提高处理效率
- **缓存机制**: 缓存常用词典和模型
- **并行处理**: 大数据量时使用多进程

## ⚠️ 重要说明

### 依赖库要求
- **jieba**: 中文分词库，需要预先下载词典
- **nltk**: 英文处理库，需要下载相关数据
- **emoji**: 表情符号处理库

### 性能考虑
- 大文本处理时注意内存使用
- 批量处理比单个处理更高效
- 中英文混合处理比单语言处理稍慢

### 编码问题
- 确保输入文本使用UTF-8编码
- 处理特殊字符时注意编码转换
- 输出结果保持编码一致性

---

**[⬅️ 预处理模块概览](README.md) | [图像处理 ➡️](image_processing.md)**
