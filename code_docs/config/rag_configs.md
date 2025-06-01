# RAG配置 RAG Configs

> 🔍 **检索增强生成系统的完整配置管理**

## 📋 功能说明

`rag_configs.yaml` 专门配置RAG (Retrieval-Augmented Generation) 系统，基于Qwen3-0.6B模型的谣言检测RAG系统。

## 🎯 主要配置块

### rag_system - RAG系统配置
- 系统名称和版本信息
- 基础模型配置 (Qwen3-0.6B)
- 系统描述信息

### knowledge_base - 知识库配置
- **知识来源**: predefined (预定义), dataset (数据集样本)
- **预定义知识类别**: 
  - credibility (可信度判断)
  - language_pattern (语言模式)
  - rumor_pattern (谣言特征)
  - verification_method (验证方法)
- **索引方法**: semantic (语义), tfidf
- **向量化配置**: 嵌入模型和索引类型

### retrieval - 检索配置
- **检索参数**: top_k数量、相似度阈值
- **检索策略**: semantic, keyword, hybrid
- **结果处理**: 重排序、去重、过滤
- **文档限制**: 最大文档数和最小分数

### generation - 生成配置
- **LLM模型**: Qwen/Qwen3-0.6B
- **微调配置**: LoRA微调开关
- **生成参数**: 
  - max_new_tokens (最大生成长度)
  - temperature (随机性控制)
  - top_p (核采样参数)
- **提示模板**: 语言风格、示例包含

### enhancement - RAG增强配置
- **多查询RAG**: 查询重写和扩展
- **迭代RAG**: 多轮检索优化
- **自适应RAG**: 动态调整检索策略
- **反馈学习**: 结果质量改进

### evaluation - 评估配置
- **评估指标**: accuracy, precision, recall, f1_score, confidence
- **对比方法**: standard_llm, simple_retrieval
- **性能监控**: 时间、内存使用跟踪

### cache - 缓存配置
- **嵌入缓存**: 向量表示缓存
- **检索结果缓存**: 查询结果缓存
- **生成结果缓存**: 生成内容缓存
- **缓存策略**: 大小限制和TTL设置

### output - 输出配置
- **保存路径**: 知识库、评估结果、日志文件
- **结果格式**: 包含检索文档、置信度、调试信息
- **日志级别**: DEBUG, INFO, WARNING, ERROR

### security - 安全配置
- **隐私保护**: 日志匿名化、敏感数据掩码
- **内容过滤**: 个人信息、有害内容过滤

### experimental - 实验配置
- **新功能开关**: 多模态RAG、跨语言RAG、时序RAG
- **A/B测试**: 测试流量分配和对照组设置

### resource_limits - 资源限制
- **内存限制**: 知识库、缓存、模型大小限制
- **计算限制**: 并发请求、检索时间、生成时间
- **存储限制**: 日志文件、结果文件管理

## 💡 使用场景

- 配置RAG检索策略
- 设置知识库构建参数
- 调整生成模型参数
- 配置缓存和性能优化
- 设置安全和隐私保护
- 管理实验和A/B测试

---

**[⬅️ 支持模型列表](supported_models.md) | [数据工具模块 ➡️](../data_utils/README.md)**
