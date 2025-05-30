# 附录B 常用工具与资源

## B.1 开发工具推荐

### B.1.1 编程环境

| 工具类型 | 推荐工具 | 特点 | 适用场景 | 获取方式 |
|----------|----------|------|----------|----------|
| **Python发行版** | Anaconda | 科学计算包齐全 | 数据科学研究 | anaconda.com |
| | Miniconda | 轻量级包管理 | 服务器环境 | docs.conda.io |
| **IDE** | PyCharm Professional | 功能强大 | 大型项目开发 | jetbrains.com/pycharm |
| | Visual Studio Code | 轻量级、插件丰富 | 日常开发 | code.visualstudio.com |
| | Jupyter Lab | 交互式开发 | 实验和原型 | jupyterlab.readthedocs.io |
| **代码编辑器** | Vim/Neovim | 高效编辑 | 服务器开发 | vim.org |
| | Sublime Text | 轻量快速 | 快速编辑 | sublimetext.com |

### B.1.2 版本控制与协作

| 工具名称 | 主要功能 | 使用场景 | 学习资源 |
|----------|----------|----------|----------|
| **Git** | 分布式版本控制 | 代码版本管理 | git-scm.com/doc |
| **GitHub** | 代码托管+协作 | 开源项目管理 | docs.github.com |
| **GitLab** | 企业级Git服务 | 私有项目管理 | docs.gitlab.com |
| **Bitbucket** | Atlassian Git服务 | 团队协作 | bitbucket.org/support |

### B.1.3 实验管理工具

| 工具名称 | 主要功能 | 优势 | 使用建议 |
|----------|----------|------|----------|
| **Weights & Biases** | 实验跟踪可视化 | 强大的可视化 | 学术研究首选 |
| **MLflow** | ML生命周期管理 | 开源免费 | 企业项目推荐 |
| **TensorBoard** | 深度学习可视化 | TensorFlow原生 | TensorFlow用户 |
| **Neptune** | 元数据管理 | 团队协作强 | 大团队项目 |
| **Comet** | 实验对比分析 | 对比功能强 | 超参数优化 |

## B.2 深度学习框架与库

### B.2.1 核心框架对比

| 框架 | 优势 | 劣势 | 适用场景 | 学习资源 |
|------|------|------|----------|----------|
| **PyTorch** | 动态图，调试友好 | 部署相对复杂 | 学术研究 | pytorch.org/tutorials |
| **TensorFlow** | 部署生态完善 | 学习曲线陡 | 工业部署 | tensorflow.org/learn |
| **JAX** | 函数式编程 | 生态相对小 | 高性能计算 | jax.readthedocs.io |
| **PaddlePaddle** | 中文支持好 | 国际化程度低 | 中文NLP | paddlepaddle.org.cn |

### B.2.2 NLP专用库

| 库名称 | 主要功能 | 支持模型 | 安装方式 |
|--------|----------|----------|----------|
| **Transformers** | 预训练模型库 | BERT, GPT, T5等 | `pip install transformers` |
| **spaCy** | 工业级NLP | 传统NLP任务 | `pip install spacy` |
| **NLTK** | 教学研究NLP | 基础NLP算法 | `pip install nltk` |
| **AllenNLP** | 深度学习NLP | 各种NLP模型 | `pip install allennlp` |
| **Flair** | 序列标注 | NER, POS等 | `pip install flair` |

### B.2.3 中文NLP工具

| 工具名称 | 主要功能 | 特色 | 推荐程度 |
|----------|----------|------|----------|
| **jieba** | 中文分词 | 简单易用 | ⭐⭐⭐⭐⭐ |
| **pkuseg** | 中文分词 | 多领域适应 | ⭐⭐⭐⭐ |
| **LAC** | 词法分析 | 百度开源 | ⭐⭐⭐⭐ |
| **HanLP** | 中文NLP套件 | 功能全面 | ⭐⭐⭐⭐ |
| **LTP** | 语言技术平台 | 哈工大开源 | ⭐⭐⭐⭐ |
| **THULAC** | 中文分词标注 | 清华开源 | ⭐⭐⭐ |

## B.3 数据集资源汇总

### B.3.1 通用NLP数据集

| 数据集名称 | 任务类型 | 规模 | 语言 | 获取地址 |
|------------|----------|------|------|----------|
| **GLUE** | 多任务基准 | 9任务 | 英文 | gluebenchmark.com |
| **SuperGLUE** | 高难度基准 | 8任务 | 英文 | super.gluebenchmark.com |
| **XTREME** | 跨语言基准 | 9任务40语言 | 多语言 | github.com/google-research/xtreme |
| **CLUE** | 中文语言理解 | 9任务 | 中文 | cluebenchmarks.com |

### B.3.2 情感分析数据集

| 数据集 | 领域 | 规模 | 标注类型 | 下载地址 |
|--------|------|------|----------|----------|
| **IMDB** | 电影评论 | 50K | 二分类 | ai.stanford.edu/~amaas/data/sentiment |
| **Stanford SST** | 电影评论 | 11K | 五分类 | nlp.stanford.edu/sentiment |
| **Amazon Reviews** | 产品评论 | 142M | 星级评分 | jmcauley.ucsd.edu/data/amazon |
| **SemEval ABSA** | 餐厅笔记本 | 6K | 方面情感 | alt.qcri.org/semeval2014/task4 |
| **中文情感数据** | 多领域 | 各异 | 多种标注 | github.com/SophonPlus/ChineseNlpCorpus |

### B.3.3 虚假信息检测数据集

| 数据集 | 平台来源 | 样本数 | 标注类型 | 获取方式 |
|--------|----------|---------|----------|----------|
| **LIAR** | PolitiFact | 12.8K | 6级真实度 | github.com/thunlp/LIAR |
| **FakeNewsNet** | 多平台 | 23K | 真假标注 | github.com/KaiDMML/FakeNewsNet |
| **PHEME** | Twitter | 5.8K | 谣言检测 | figshare.com/articles/PHEME_dataset |
| **CoAID** | 多平台 | 4.3K | COVID虚假信息 | github.com/cuilimeng/CoAID |

### B.3.4 多语言数据集

| 数据集 | 覆盖语言 | 任务类型 | 特色 | 访问地址 |
|--------|----------|----------|------|----------|
| **Universal Dependencies** | 100+语言 | 句法分析 | 语言学标准 | universaldependencies.org |
| **WikiANN** | 282语言 | 命名实体识别 | 维基百科数据 | github.com/afshinrahimi/mmner |
| **XNLI** | 15语言 | 自然语言推理 | 跨语言评估 | cims.nyu.edu/~sbowman/xnli |
| **TyDiQA** | 11语言 | 问答 | 多样化语言 | ai.google.com/research/tydiqa |

## B.4 预训练模型资源

### B.4.1 英文预训练模型

| 模型名称 | 参数量 | 适用任务 | 获取地址 | 使用建议 |
|----------|--------|----------|----------|----------|
| **BERT-base** | 110M | 文本理解 | huggingface.co/bert-base-uncased | 入门首选 |
| **RoBERTa-large** | 355M | 文本理解 | huggingface.co/roberta-large | 性能更好 |
| **GPT-2** | 1.5B | 文本生成 | huggingface.co/gpt2 | 生成任务 |
| **T5-base** | 220M | 文本到文本 | huggingface.co/t5-base | 统一框架 |
| **DeBERTa-v3** | 184M | 各种NLP任务 | huggingface.co/deberta-v3-base | 当前最佳 |

### B.4.2 中文预训练模型

| 模型名称 | 开发机构 | 特色 | 下载地址 |
|----------|----------|------|----------|
| **BERT-wwm-chinese** | 哈工大 | 全词遮蔽 | huggingface.co/hfl/chinese-bert-wwm |
| **RoBERTa-wwm-ext** | 哈工大 | 扩展训练 | huggingface.co/hfl/chinese-roberta-wwm-ext |
| **MacBERT** | 哈工大 | 中文优化 | huggingface.co/hfl/chinese-macbert-base |
| **ChineseBERT** | 清华 | 字形信息 | huggingface.co/ShannonAI/ChineseBERT-base |
| **ERNIE** | 百度 | 知识增强 | paddlenlp.readthedocs.io |

### B.4.3 多语言预训练模型

| 模型名称 | 语言覆盖 | 参数量 | 特点 | 使用场景 |
|----------|----------|--------|------|----------|
| **mBERT** | 104语言 | 179M | 多语言BERT | 跨语言迁移 |
| **XLM-R** | 100语言 | 270M/550M | 更好多语言性能 | 多语言任务 |
| **mT5** | 101语言 | 300M-13B | 多语言生成 | 多语言生成任务 |
| **RemBERT** | 110语言 | 559M | 改进的多语言模型 | 高质量多语言 |

## B.5 计算资源平台

### B.5.1 免费计算资源

| 平台名称 | 提供资源 | 使用限制 | 优缺点 | 注册地址 |
|----------|----------|----------|--------|----------|
| **Google Colab** | GPU/TPU | 12小时会话 | 免费但不稳定 | colab.research.google.com |
| **Kaggle Kernels** | GPU | 30小时/周 | 稳定但限制多 | kaggle.com |
| **Paperspace Gradient** | GPU | 免费额度 | 界面友好 | gradient.paperspace.com |
| **AWS Free Tier** | 各种服务 | 12个月 | 需要信用卡 | aws.amazon.com/free |

### B.5.2 付费计算平台

| 平台名称 | 定价模式 | 适用场景 | 特色功能 | 官网 |
|----------|----------|----------|----------|------|
| **Google Cloud** | 按需付费 | 大规模训练 | TPU支持 | cloud.google.com |
| **AWS** | 按需付费 | 企业级应用 | 服务丰富 | aws.amazon.com |
| **Azure** | 按需付费 | 微软生态 | 与Office集成 | azure.microsoft.com |
| **阿里云** | 按需付费 | 中国地区 | 网络优势 | aliyun.com |
| **腾讯云** | 按需付费 | 中国地区 | 游戏优化 | cloud.tencent.com |

### B.5.3 高校和科研资源

| 资源类型 | 获取途径 | 适用人群 | 申请要点 |
|----------|----------|----------|----------|
| **校内集群** | 学校申请 | 在校师生 | 展示计算需求 |
| **国家超算中心** | 项目申请 | 科研人员 | 需要正式项目 |
| **XSEDE** | 在线申请 | 美国研究者 | 详细研究计划 |
| **EuroHPC** | 项目申请 | 欧洲研究者 | 多国合作项目 |

## B.6 在线学习平台

### B.6.1 综合学习平台

| 平台名称 | 内容特色 | 课程质量 | 价格 | 推荐课程 |
|----------|----------|----------|------|----------|
| **Coursera** | 名校课程 | 高 | 免费+付费 | CS224N, Deep Learning |
| **edX** | 名校开放课程 | 高 | 免费+认证付费 | MIT NLP课程 |
| **Udacity** | 项目导向 | 中高 | 付费 | NLP Nanodegree |
| **Udemy** | 实用技能 | 中 | 付费 | PyTorch实战课程 |

### B.6.2 专业技术平台

| 平台名称 | 专业领域 | 内容形式 | 适合人群 | 网址 |
|----------|----------|----------|----------|------|
| **Hugging Face Course** | NLP/Transformers | 交互式教程 | NLP研究者 | huggingface.co/course |
| **Fast.ai** | 深度学习 | 实践导向 | 初学者 | course.fast.ai |
| **DeepLearning.ai** | AI/ML | 系统课程 | 全阶段学习者 | deeplearning.ai |
| **Weights & Biases** | MLOps | 实践教程 | ML工程师 | wandb.ai/site/tutorials |

### B.6.3 学术会议和讲座

| 资源类型 | 获取方式 | 内容质量 | 更新频率 | 推荐指数 |
|----------|----------|----------|----------|----------|
| **ACL Anthology** | 免费访问 | 极高 | 会议更新 | ⭐⭐⭐⭐⭐ |
| **VideoLectures** | 免费观看 | 高 | 不定期 | ⭐⭐⭐⭐ |
| **YouTube学术频道** | 免费观看 | 参差不齐 | 频繁 | ⭐⭐⭐ |
| **Bilibili技术区** | 免费观看 | 中等 | 频繁 | ⭐⭐⭐ |