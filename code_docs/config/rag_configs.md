# RAG配置 RAG Configs

> 🔍 **基于Qwen3-0.6B的检索增强生成系统完整配置**

## 📋 RAG系统概览

### rag_system - 系统基础配置
```yaml
rag_system:
  name: "MR2-RAG-System"
  version: "1.0.0"
  description: "基于Qwen3-0.6B的谣言检测RAG系统"
```

本配置文件专门为谣言检测任务设计的RAG系统，集成了知识检索、增强生成和多轮优化功能。

## 🗄️ 知识库配置

### knowledge_base - 知识来源管理
```yaml
knowledge_base:
  # 知识来源
  sources:
    - "predefined"     # 预定义专家知识
    - "dataset"        # MR2数据集样本
  
  # 预定义知识类别
  predefined:
    enabled: true
    categories:
      - "credibility"           # 可信度判断规则
      - "language_pattern"      # 语言模式识别
      - "rumor_pattern"        # 谣言特征模式
      - "verification_method"   # 事实验证方法
      - "domain_specific"      # 领域专门知识
```

### 索引配置
```yaml
  indexing:
    method: "semantic"        # semantic, tfidf
    embedding_model: "all-MiniLM-L6-v2"
    index_type: "faiss"       # faiss, sklearn
    normalize_embeddings: true
```

### 数据集知识配置
```yaml
  dataset:
    enabled: true
    max_samples: 200          # 最大样本数
    min_text_length: 20       # 最小文本长度
    sample_strategy: "balanced"  # 平衡采样策略
```

## 🔍 检索配置

### retrieval - 检索参数
```yaml
retrieval:
  # 基础检索参数
  default_top_k: 5           # 默认检索文档数
  max_top_k: 10             # 最大检索文档数
  min_similarity: 0.1       # 最小相似度阈值
  
  # 检索策略
  strategy: "semantic"       # semantic, keyword, hybrid
  rerank: true              # 重排序
  deduplicate: true         # 去重
  
  # 文档过滤
  filtering:
    min_score: 0.0          # 最小相关度分数
    max_documents: 10       # 最大文档数
    prefer_types: []        # 偏好文档类型
```

### 检索策略对比
| 策略 | 优势 | 适用场景 |
|------|------|----------|
| **semantic** | 语义理解准确 | 复杂查询、概念匹配 |
| **keyword** | 速度快、精确匹配 | 简单查询、关键词检索 |
| **hybrid** | 综合优势 | 大多数实际应用 |

## 🤖 生成配置

### generation - LLM模型配置
```yaml
generation:
  # 模型基础配置
  model:
    name: "Qwen/Qwen3-0.6B"
    use_lora: true
    load_in_4bit: false
    max_length: 512
  
  # 生成参数
  parameters:
    max_new_tokens: 200
    temperature: 0.3         # 较低温度确保准确性
    do_sample: true
    top_p: 0.9
    repetition_penalty: 1.1
```

### 提示模板配置
```yaml
  prompts:
    language: "mixed"         # chinese, english, mixed
    style: "formal"          # formal, conversational, detailed
    include_examples: false  # 是否包含示例
    max_context_length: 1000 # 最大上下文长度
```

## ⚡ RAG增强技术

### enhancement - 多种增强策略
```yaml
enhancement:
  # 多查询RAG
  multi_query:
    enabled: true
    max_variants: 3          # 最大查询变体数
    variant_strategies:
      - "keyword_extraction"
      - "question_reformulation" 
      - "domain_specific"
```

### 迭代RAG
```yaml
  iterative:
    enabled: true
    max_iterations: 3        # 最大迭代次数
    confidence_threshold: 0.8 # 置信度阈值
    improvement_threshold: 0.05 # 改进阈值
```

### 自适应检索
```yaml
  adaptive:
    enabled: true
    adjust_top_k: true       # 动态调整检索数量
    context_aware: true      # 上下文感知
    feedback_learning: true  # 反馈学习
```

## 📊 评估配置

### evaluation - 性能评估
```yaml
evaluation:
  # 评估指标
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "confidence"
  
  # 对比评估
  comparison:
    baseline_methods:
      - "standard_llm"       # 标准LLM方法
      - "simple_retrieval"   # 简单检索
  
  # 性能监控
  monitoring:
    track_retrieval_time: true
    track_generation_time: true
    track_memory_usage: false
    log_failed_cases: true
```

## 💾 缓存配置

### cache - 多层缓存策略
```yaml
cache:
  # 嵌入缓存
  embeddings:
    enabled: true
    max_size: 10000         # 最大缓存条目
    ttl: 3600              # 生存时间(秒)
  
  # 检索结果缓存
  retrieval_results:
    enabled: true
    max_size: 1000
    ttl: 1800
  
  # 生成结果缓存
  generation_results:
    enabled: false          # 默认关闭
    max_size: 500
    ttl: 900
```

## 📁 输出配置

### output - 结果管理
```yaml
output:
  # 保存路径
  save_paths:
    knowledge_base: "outputs/models/llms/rag_knowledge_base.json"
    evaluation_results: "outputs/models/llms/rag_evaluation.json"
    logs: "outputs/logs/rag_system.log"
  
  # 结果格式
  result_format:
    include_retrieved_docs: true
    include_confidence: true
    include_timing: false
    include_debug_info: false
  
  # 日志配置
  logging:
    level: "INFO"           # DEBUG, INFO, WARNING, ERROR
    console_output: true
    file_output: true
    max_log_size: "10MB"
```

## 🔒 安全和隐私配置

### security - 安全机制
```yaml
security:
  # 数据隐私
  privacy:
    anonymize_logs: false   # 日志匿名化
    mask_sensitive_data: false # 敏感数据掩码
  
  # 内容过滤
  content_filtering:
    enabled: false          # 内容过滤
    filter_personal_info: false # 过滤个人信息
    filter_harmful_content: false # 过滤有害内容
```

## 🧪 实验配置

### experimental - 前沿功能
```yaml
experimental:
  # 新功能开关
  features:
    multi_modal_rag: false   # 多模态RAG
    cross_lingual_rag: false # 跨语言RAG
    temporal_rag: false      # 时序RAG
    federated_rag: false     # 联邦RAG
  
  # A/B测试
  ab_testing:
    enabled: false
    test_ratio: 0.1         # 测试流量比例
    control_group: "standard_rag"
    test_group: "enhanced_rag"
```

## 📈 资源限制

### resource_limits - 性能约束
```yaml
resource_limits:
  # 内存限制
  memory:
    max_knowledge_base_size: "1GB"
    max_cache_size: "500MB"
    max_model_size: "5GB"
  
  # 计算限制
  compute:
    max_concurrent_requests: 10
    max_retrieval_time: 5.0   # 秒
    max_generation_time: 30.0 # 秒
  
  # 存储限制
  storage:
    max_log_files: 10
    max_result_files: 100
    cleanup_interval: 24     # 小时
```

## 💡 RAG配置使用示例

```python
from utils.config_manager import get_config_manager

# 获取配置管理器
config_mgr = get_config_manager()

# 检查RAG配置文件是否存在
rag_config_path = config_mgr.config_dir / 'rag_configs.yaml'
if rag_config_path.exists():
    rag_config = config_mgr.configs.get('rag', {})
    
    # 获取知识库配置
    kb_config = rag_config['knowledge_base']
    max_samples = kb_config['dataset']['max_samples']  # 200
    
    # 获取检索配置
    retrieval_config = rag_config['retrieval']
    top_k = retrieval_config['default_top_k']  # 5
    
    # 获取生成配置
    gen_config = rag_config['generation']
    model_name = gen_config['model']['name']  # "Qwen/Qwen3-0.6B"
    temperature = gen_config['parameters']['temperature']  # 0.3
```

## 🔧 RAG系统调优建议

### 检索优化
- **提高召回率**: 增加`top_k`值，降低`min_similarity`阈值
- **提高精确率**: 启用重排序，设置更严格的过滤条件
- **平衡性能**: 使用混合检索策略

### 生成优化
- **提高准确性**: 降低`temperature`(0.1-0.3)，增加检索文档质量
- **增加创造性**: 提高`temperature`(0.7-0.9)，但可能影响事实准确性
- **控制长度**: 调整`max_new_tokens`平衡详细程度和效率

### 知识库优化
- **领域专业性**: 增加预定义知识类别
- **数据质量**: 提高`min_text_length`，使用更好的采样策略
- **索引效率**: 选择合适的嵌入模型和索引方法

## ⚠️ 注意事项

### 模型要求
- Qwen3-0.6B需要足够的GPU内存(至少4GB)
- LoRA微调可以显著减少内存需求
- 建议使用混合精度训练

### 知识库质量
- 预定义知识需要人工精心构建
- 数据集采样要保持标签平衡
- 定期更新知识库内容

### 性能监控
- 监控检索时间和生成时间
- 跟踪缓存命中率
- 记录失败案例进行改进

---

**[⬅️ 支持模型列表](supported_models.md) | [数据工具模块 ➡️](../data_utils/README.md)**