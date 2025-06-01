# NLP 资源速查表（更新：2025‑06‑02）

> 🎯 *面向中文与英文自然语言处理学习者，涵盖开发环境、框架、免费 GPU、课程与小工具。所有链接均为官方或权威入口。*

---

## 1. Python 环境官方发行版

* **CPython 官方网站** [https://www.python.org](https://www.python.org)
* **Anaconda Distribution** [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
* **Miniconda (轻量 Conda)** [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* **Pyenv (多版本管理)** [https://github.com/pyenv/pyenv](https://github.com/pyenv/pyenv)
* **MambaForge (conda+mamba)** [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge)

## 2. 主流 Python IDE / 编辑器

* **Visual Studio Code** [https://code.visualstudio.com](https://code.visualstudio.com)（推荐插件：Python、Jupyter、Pylance）
* **PyCharm** [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)
* **JupyterLab** [https://jupyter.org](https://jupyter.org)
* **Sublime Text** [https://www.sublimetext.com](https://www.sublimetext.com)
* **Spyder** [https://www.spyder-ide.org](https://www.spyder-ide.org)

## 3. 深度学习 & NLP 框架官网 / 安装指南

* **PyTorch** [https://pytorch.org](https://pytorch.org)
* **TensorFlow** [https://www.tensorflow.org](https://www.tensorflow.org)
* **Hugging Face Transformers** [https://huggingface.co/transformers](https://huggingface.co/transformers)
* **spaCy** [https://spacy.io](https://spacy.io)
* **AllenNLP** [https://www.allennlp.org](https://www.allennlp.org)
* **NLTK** [https://www.nltk.org](https://www.nltk.org)
* **Gensim** [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
* **SentenceTransformers** [https://www.sbert.net](https://www.sbert.net)
* **PyTorch Geometric (GNN)** [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

## 4. 免费 GPU 在线运行环境

* **Google Colab** [https://colab.research.google.com](https://colab.research.google.com)（T4/A100，日配额）
* **Kaggle Notebooks** [https://www.kaggle.com/code](https://www.kaggle.com/code)
* **Paperspace Gradient (Free Tier)** [https://www.paperspace.com/gradient](https://www.paperspace.com/gradient)
* **Amazon SageMaker Studio Lab** [https://studiolab.sagemaker.aws/](https://studiolab.sagemaker.aws/)
* **Deepnote Community** [https://deepnote.com](https://deepnote.com)
* **Saturn Cloud Free** [https://saturncloud.io](https://saturncloud.io)

## 5. 学习教程 / 课程 / 社区

* **Hugging Face Course** [https://huggingface.co/course](https://huggingface.co/course)
* **fast.ai Practical DL** [https://course.fast.ai](https://course.fast.ai)
* **Stanford CS224N** [https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)
* **Coursera NLP Specialization** [https://www.coursera.org/specializations/natural-language-processing](https://www.coursera.org/specializations/natural-language-processing)
* **Kaggle Learn NLP** [https://www.kaggle.com/learn/nlp](https://www.kaggle.com/learn/nlp)
* **Real Python** [https://realpython.com](https://realpython.com)
* **OpenAI Cookbook** [https://github.com/openai/openai-cookbook](https://github.com/openai/openai-cookbook)

## 6. 中文 NLP 生态

* **jieba 分词** [https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba)
* **THULAC** [http://thulac.thunlp.org](http://thulac.thunlp.org)
* **HanLP** [https://hanlp.hankcs.com](https://hanlp.hankcs.com)
* **Chinese Transformers on Hugging Face** [https://huggingface.co/models?language=zh](https://huggingface.co/models?language=zh)

## 7. 数据集与评测基准

* **Hugging Face Datasets Hub** [https://huggingface.co/datasets](https://huggingface.co/datasets)
* **Papers With Code** [https://paperswithcode.com](https://paperswithcode.com)
* **GLUE / SuperGLUE** [https://gluebenchmark.com](https://gluebenchmark.com)

## 8. 实验管理与可视化

* **Weights & Biases (wandb)** [https://wandb.ai](https://wandb.ai)
* **MLflow** [https://mlflow.org](https://mlflow.org)
* **TensorBoard** [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)

## 9. 好用的小工具

* **Streamlit (快速原型 Web)** [https://streamlit.io](https://streamlit.io)
* **Gradio (交互式 Demo)** [https://www.gradio.app](https://www.gradio.app)
* **DVC (数据版本控制)** [https://dvc.org](https://dvc.org)

---

### 使用小贴士

* 使用 **conda + mamba** 快速安装深度学习框架，可通过 `conda install pytorch-cuda=12.1 -c pytorch -c nvidia` 直接获得 GPU 版 PyTorch。
* 在 **Google Colab** 或 **Kaggle** 上运行长时间任务时，注意保存断点（如使用 `torch.save` 或 `huggingface_hub` 同步）。
* 大模型微调推荐组合：*Transformers + PEFT + LoRA + BitsAndBytes*。
* 图神经网络入门：先浏览 **PyTorch Geometric** 官方教程，再阅读 **Stanford CS224W** 课程材料。
