# 文本处理器 Text Processing

> 📝 **专业的中英文混合文本预处理器，支持智能分词和多维特征提取**

## 📋 功能概览

`TextProcessor`是专门为MR2数据集设计的文本预处理器，支持中英文混合文本的分词、清洗、标准化和特征提取。

## 🚀 核心类

### TextProcessor
主要文本处理类，支持多语言智能处理：

```python
from preprocessing import TextProcessor

# 初始化处理器
processor = TextProcessor(language='mixed')
```

#### 初始化参数
```python
def __init__(self, language: str = 'mixed'):
    """
    初始化文本处理器
    
    Args:
        language: 语言类型
                 'chinese' - 纯中文处理
                 'english' - 纯英文处理  
                 'mixed' - 中英文混合处理（推荐）
    """
```

## 🔧 核心方法

### 语言检测
```python
def detect_language(self, text: str) -> str:
    """
    智能检测文本语言类型
    
    Args:
        text: 输入文本
        
    Returns:
        'chinese' | 'english' | 'mixed' | 'unknown'
    """
```

**检测逻辑**：
```python
# 统计字符类型比例
chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
english_chars = len(re.findall(r'[a-zA-Z]', text))

chinese_ratio = chinese_chars / total_chars
english_ratio = english_chars / total_chars

if chinese_ratio > 0.7:
    return 'chinese'
elif english_ratio > 0.7:
    return 'english'
else:
    return 'mixed'
```

### 文本清洗
```python
def clean_text(self, text: str) -> str:
    """
    深度文本清洗
    
    清洗内容:
    - URL移除 (http链接和www链接)
    - @提及移除
    - #话题标签处理（可选）
    - emoji转文本描述
    - HTML标签移除
    - 标点符号标准化
    - 空白字符规范化
    """
```

**清洗示例**：
```python
original = "这是测试 @user #topic https://example.com 😊!!!"
cleaned = processor.clean_text(original)
print(cleaned)
# 输出: "这是测试 😊"
```

### 智能分词

#### 统一分词接口
```python
def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
    """
    统一分词接口，自动选择分词策略
    
    Args:
        text: 输入文本
        language: 指定语言（None时自动检测）
        
    Returns:
        分词结果列表
    """
```

#### 中文分词
```python
def tokenize_chinese(self, text: str) -> List[str]:
    """
    中文专用分词
    
    特点:
    - 使用jieba分词引擎
    - 自动过滤停用词
    - 去除无意义字符
    - 支持自定义词典
    """
```

**中文停用词**（内置）：
```python
chinese_stopwords = {
    '的', '了', '在', '是', '和', '有', '我', '你', '他', '她', '它',
    '我们', '你们', '他们', '这', '那', '这个', '那个', '上', '下',
    '中', '大', '小', '多', '少', '好', '坏', '对', '错', '没', '不'
    # ... 更多停用词
}
```

#### 英文分词
```python
def tokenize_english(self, text: str) -> List[str]:
    """
    英文专用分词
    
    特点:
    - 使用NLTK word_tokenize
    - 自动小写化
    - 停用词过滤
    - 标点符号过滤
    - 词长过滤（>2字符）
    """
```

#### 混合分词
```python
def tokenize_mixed(self, text: str) -> List[str]:
    """
    中英文混合分词
    
    策略:
    1. 按中英文分割文本
    2. 中文部分使用jieba分词
    3. 英文部分使用NLTK分词
    4. 合并结果并去重
    """
```

**混合分词示例**：
```python
text = "这是测试文本 This is a test"
tokens = processor.tokenize_mixed(text)
print(tokens)
# 输出: ['这是', '测试', '文本', 'this', 'test']
```

## 📊 特征提取

### extract_features 方法
```python
def extract_features(self, text: str) -> Dict[str, Union[int, float, List[str]]]:
    """
    提取全面的文本特征
    
    返回特征类别:
    - 基础特征: 长度、词数、字符数
    - 语言特征: 语言类型、大写比例
    - 标点特征: 各种标点符号统计
    - 内容特征: URL、提及、emoji等计数
    - 分词特征: tokens和token数量
    """
```

### 完整特征说明
```python
features = {
    # === 基础文本特征 ===
    'text_length': len(text),           # 原始文本长度
    'word_count': len(text.split()),    # 简单空格分词词数
    'char_count': len(text),            # 字符总数
    'language': 'mixed',                # 检测的语言类型
    
    # === 分词特征 ===
    'tokens': ['token1', 'token2'],     # 智能分词结果
    'token_count': 10,                  # 智能分词token数
    
    # === 标点符号特征 ===
    'exclamation_count': 2,             # 感叹号数量
    'question_count': 1,                # 问号数量
    'period_count': 3,                  # 句号数量
    'comma_count': 5,                   # 逗号数量
    
    # === 英文特征 ===
    'uppercase_ratio': 0.15,            # 大写字母比例
    'digit_count': 4,                   # 数字字符数
    
    # === 内容特征 ===
    'url_count': 1,                     # URL数量
    'mention_count': 2,                 # @提及数量  
    'hashtag_count': 1,                 # #话题标签数量
    'emoji_count': 3                    # emoji数量
}
```

## 🔄 批量处理

### preprocess_batch 方法
```python
def preprocess_batch(self, texts: List[str]) -> List[Dict[str, any]]:
    """
    批量预处理文本列表
    
    Args:
        texts: 文本列表
        
    Returns:
        处理结果列表，每个元素包含:
        {
            'original_text': str,     # 原始文本
            'cleaned_text': str,      # 清洗后文本
            'tokens': List[str],      # 分词结果
            'features': Dict          # 特征字典
        }
    """
```

**批量处理示例**：
```python
texts = [
    "这是第一个测试文本",
    "This is the second test text!",
    "混合语言文本 mixed language text"
]

results = processor.preprocess_batch(texts)
for i, result in enumerate(results):
    print(f"文本{i+1}:")
    print(f"  原文: {result['original_text']}")
    print(f"  清洗: {result['cleaned_text']}")
    print(f"  分词: {result['tokens']}")
    print(f"  语言: {result['features']['language']}")
    print(f"  长度: {result['features']['text_length']}")
```

## ⚙️ 配置集成

### 自动配置加载
```python
# 从配置管理器自动加载参数
if USE_CONFIG:
    try:
        config = get_data_config()
        self.processing_config = config.get('processing', {}).get('text', {})
    except:
        self.processing_config = {}
else:
    self.processing_config = {}

# 设置处理参数
self.max_length = self.processing_config.get('max_length', 512)
self.remove_urls = self.processing_config.get('remove_urls', True)
self.remove_mentions = self.processing_config.get('remove_mentions', True)
self.remove_hashtags = self.processing_config.get('remove_hashtags', False)
```

### 可配置参数
| 配置参数 | 默认值 | 说明 |
|----------|--------|------|
| `max_length` | 512 | 最大文本长度 |
| `remove_urls` | True | 是否移除URL |
| `remove_mentions` | True | 是否移除@提及 |
| `remove_hashtags` | False | 是否移除#标签 |
| `normalize_whitespace` | True | 是否标准化空白字符 |
| `tokenization` | "mixed" | 分词类型 |

## 🔧 依赖库处理

### 智能依赖检测
```python
# 中文分词库检测
try:
    import jieba
    import jieba.posseg as pseg
    HAS_JIEBA = True
except ImportError:
    print("⚠️ jieba未安装，中文分词功能不可用")
    HAS_JIEBA = False

# 英文处理库检测
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    print("⚠️ nltk未安装，英文高级处理功能不可用")
    HAS_NLTK = False
```

### 降级处理策略
```python
# 中文分词降级
if not HAS_JIEBA:
    def tokenize_chinese(self, text: str) -> List[str]:
        # 简单的字符级分割
        return list(text.replace(' ', ''))

# 英文分词降级
if not HAS_NLTK:
    def tokenize_english(self, text: str) -> List[str]:
        # 简单的空格分割
        return text.lower().split()
```

## 🎯 使用示例

### 基础使用
```python
from preprocessing import TextProcessor

# 创建处理器
processor = TextProcessor(language='mixed')

# 处理单个文本
text = "这是一个测试文本 This is a test! @user #topic http://example.com"

# 1. 语言检测
language = processor.detect_language(text)
print(f"语言类型: {language}")  # 输出: mixed

# 2. 文本清洗
cleaned = processor.clean_text(text)
print(f"清洗后: {cleaned}")
# 输出: "这是一个测试文本 This is a test"

# 3. 分词
tokens = processor.tokenize(text)
print(f"分词结果: {tokens}")
# 输出: ['这是', '一个', '测试', '文本', 'this', 'test']

# 4. 特征提取
features = processor.extract_features(text)
print(f"特征摘要:")
print(f"  文本长度: {features['text_length']}")
print(f"  词数: {features['token_count']}")
print(f"  URL数: {features['url_count']}")
print(f"  提及数: {features['mention_count']}")
```

### 自定义配置
```python
# 创建自定义配置的处理器
processor = TextProcessor(language='chinese')

# 修改处理参数
processor.remove_hashtags = True   # 启用话题标签移除
processor.remove_urls = False      # 保留URL

# 处理文本
text = "关注 #AI技术 发展 https://ai-news.com"
cleaned = processor.clean_text(text)
print(cleaned)  # 输出: "关注 发展 https://ai-news.com"
```

### 多语言处理对比
```python
# 测试不同语言的处理效果
test_texts = {
    'chinese': "今天天气很好，适合出门散步。",
    'english': "Today is a beautiful day for a walk.",
    'mixed': "今天weather很好 perfect for walking"
}

for lang, text in test_texts.items():
    print(f"\n=== {lang.upper()} 文本处理 ===")
    print(f"原文: {text}")
    
    # 语言检测
    detected = processor.detect_language(text)
    print(f"检测语言: {detected}")
    
    # 分词
    tokens = processor.tokenize(text)
    print(f"分词结果: {tokens}")
    
    # 特征
    features = processor.extract_features(text)
    print(f"词数: {features['token_count']}")
```

## 🔍 高级功能

### 停用词自定义
```python
# 扩展中文停用词
processor.chinese_stopwords.update({
    '一些', '各种', '什么', '怎么', '为什么'
})

# 扩展英文停用词
processor.english_stopwords.update({
    'would', 'could', 'should', 'might'
})
```

### 词干提取和词形还原（英文）
```python
if HAS_NLTK:
    # 词干提取
    stemmed = processor.stemmer.stem('running')  # 输出: 'run'
    
    # 词形还原
    lemmatized = processor.lemmatizer.lemmatize('better', 'a')  # 输出: 'good'
```

### 正则表达式模式
```python
# 内置的清洗模式
patterns = {
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'www': r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'mention': r'@[a-zA-Z0-9_\u4e00-\u9fff]+',
    'hashtag': r'#[a-zA-Z0-9_\u4e00-\u9fff]+',
    'emoji': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
}
```

## 📊 性能分析

### 处理速度基准
```python
import time

# 性能测试
test_texts = ["测试文本 test text"] * 1000

start_time = time.time()
for text in test_texts:
    processor.tokenize(text)
end_time = time.time()

print(f"处理速度: {len(test_texts)/(end_time-start_time):.1f} 文本/秒")

# 批量处理性能
start_time = time.time()
results = processor.preprocess_batch(test_texts)
end_time = time.time()

print(f"批量处理速度: {len(test_texts)/(end_time-start_time):.1f} 文本/秒")
```

### 内存优化
```python
# 对于大批量处理，可以分块处理
def process_large_batch(texts, chunk_size=1000):
    results = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        chunk_results = processor.preprocess_batch(chunk)
        results.extend(chunk_results)
        
        # 可选：释放内存
        del chunk_results
        
    return results
```

## 🚨 错误处理

### 输入验证
```python
def clean_text(self, text: str) -> str:
    # 输入类型检查
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # 处理逻辑
        cleaned = self._apply_cleaning_steps(text)
        return cleaned
    except Exception as e:
        logger.warning(f"文本清洗失败: {e}")
        return text  # 返回原文本而不是抛出异常
```

### 编码问题处理
```python
def safe_encode_text(self, text: str) -> str:
    """安全处理文本编码问题"""
    try:
        # 确保UTF-8编码
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        # 处理特殊字符
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        return text
    except Exception:
        return ""
```

### 依赖库缺失处理
```python
def tokenize_with_fallback(self, text: str) -> List[str]:
    """带降级的分词方法"""
    try:
        if self.language == 'chinese' and HAS_JIEBA:
            return self.tokenize_chinese(text)
        elif self.language == 'english' and HAS_NLTK:
            return self.tokenize_english(text)
        else:
            # 降级到简单分词
            return self._simple_tokenize(text)
    except Exception as e:
        logger.warning(f"分词失败，使用简单分词: {e}")
        return text.split()
```

## 🔧 调试和测试

### 内置测试功能
```python
def test_processor():
    """测试文本处理器功能"""
    processor = TextProcessor(language='mixed')
    
    test_texts = [
        "这是一个测试文本 This is a test!",
        "包含URL的文本 https://example.com 和@username",
        "混合语言文本 with English words 中文字符"
    ]
    
    print("📝 === 文本处理测试 ===")
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text}")
        
        # 语言检测
        language = processor.detect_language(text)
        print(f"  语言: {language}")
        
        # 文本清洗
        cleaned = processor.clean_text(text)
        print(f"  清洗后: {cleaned}")
        
        # 分词
        tokens = processor.tokenize(text)
        print(f"  分词: {tokens[:5]}...")  # 只显示前5个
        
        # 特征
        features = processor.extract_features(text)
        print(f"  特征: 长度={features['text_length']}, "
              f"词数={features['token_count']}, "
              f"语言={features['language']}")

if __name__ == "__main__":
    test_processor()
```

### 调试输出示例
```
📝 === 文本处理测试 ===

测试 1: 这是一个测试文本 This is a test!
  语言: mixed
  清洗后: 这是一个测试文本 This is a test
  分词: ['这是', '一个', '测试', '文本', 'this']...
  特征: 长度=25, 词数=5, 语言=mixed

测试 2: 包含URL的文本 https://example.com 和@username
  语言: mixed  
  清洗后: 包含URL的文本 和
  分词: ['包含', 'url', '文本']...
  特征: 长度=30, 词数=3, 语言=mixed
```

## 💡 最佳实践

### 1. 语言类型选择
```python
# 根据数据特点选择语言模式
if dataset_is_chinese_only:
    processor = TextProcessor(language='chinese')
elif dataset_is_english_only:
    processor = TextProcessor(language='english')
else:  # 推荐用于MR2数据集
    processor = TextProcessor(language='mixed')
```

### 2. 配置优化
```python
# 针对谣言检测任务的优化配置
processor = TextProcessor(language='mixed')
processor.remove_urls = True        # 移除URL减少噪声
processor.remove_mentions = True    # 移除@提及
processor.remove_hashtags = False   # 保留话题标签（可能有信息价值）
processor.normalize_whitespace = True  # 标准化空白字符
```

### 3. 批量处理策略
```python
# 大数据集处理策略
def efficient_batch_processing(texts, batch_size=1000):
    processor = TextProcessor(language='mixed')
    
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = processor.preprocess_batch(batch)
        all_results.extend(batch_results)
        
        # 进度报告
        if (i // batch_size) % 10 == 0:
            print(f"已处理: {i + len(batch)}/{len(texts)}")
    
    return all_results
```

### 4. 特征选择
```python
# 针对机器学习任务的特征选择
def extract_ml_features(text, processor):
    """提取适合机器学习的特征"""
    features = processor.extract_features(text)
    
    # 选择数值特征
    ml_features = {
        'text_length': features['text_length'],
        'token_count': features['token_count'],
        'exclamation_count': features['exclamation_count'],
        'question_count': features['question_count'],
        'uppercase_ratio': features['uppercase_ratio'],
        'url_count': features['url_count'],
        'mention_count': features['mention_count'],
        'emoji_count': features['emoji_count']
    }
    
    # 语言类型编码
    ml_features['is_chinese'] = 1 if features['language'] == 'chinese' else 0
    ml_features['is_english'] = 1 if features['language'] == 'english' else 0
    ml_features['is_mixed'] = 1 if features['language'] == 'mixed' else 0
    
    return ml_features
```

## ⚠️ 注意事项

### 编码问题
- 确保输入文本使用UTF-8编码
- 处理特殊字符时注意编码转换
- 输出结果保持编码一致性

### 依赖库版本
- jieba: 建议使用0.42+版本
- nltk: 建议使用3.6+版本，需要下载相关数据包
- emoji: 建议使用1.6+版本

### 性能考虑
- 大文本处理时注意内存使用
- 批量处理比单个处理更高效
- 中英文混合处理比单语言处理稍慢

---

**[⬅️ 预处理模块概览](README.md) | [图像处理 ➡️](image_processing.md)**