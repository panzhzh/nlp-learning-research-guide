#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/prompt_engineering.py

"""
谣言检测提示工程模块
包含多种提示模板和策略，支持中英双语
优化LLM在谣言检测任务上的表现
"""

import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import random
from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):
    """提示模板基类"""
    
    def __init__(self, language: str = "mixed"):
        """
        初始化提示模板
        
        Args:
            language: 语言类型 ("chinese", "english", "mixed")
        """
        self.language = language
    
    @abstractmethod
    def create_prompt(self, text: str, **kwargs) -> str:
        """创建提示"""
        pass
    
    def format_labels(self) -> Dict[str, str]:
        """格式化标签"""
        if self.language == "chinese":
            return {
                "Non-rumor": "非谣言",
                "Rumor": "谣言", 
                "Unverified": "未验证"
            }
        elif self.language == "english":
            return {
                "Non-rumor": "Non-rumor",
                "Rumor": "Rumor",
                "Unverified": "Unverified"
            }
        else:  # mixed
            return {
                "Non-rumor": "Non-rumor (非谣言)",
                "Rumor": "Rumor (谣言)",
                "Unverified": "Unverified (未验证)"
            }


class RumorPromptTemplate(BasePromptTemplate):
    """谣言检测专用提示模板"""
    
    def __init__(self, language: str = "mixed", style: str = "formal"):
        """
        初始化谣言检测提示模板
        
        Args:
            language: 语言类型
            style: 提示风格 ("formal", "conversational", "detailed")
        """
        super().__init__(language)
        self.style = style
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """加载不同风格的提示模板"""
        templates = {
            "formal": {
                "chinese": """请对以下文本进行谣言检测分析。

文本内容：{text}

请从以下三个类别中选择最合适的分类：
1. 非谣言：内容真实可信，有可靠来源支撑
2. 谣言：内容虚假或误导，缺乏事实依据
3. 未验证：无法确定真伪，需要进一步核实

分析结果：""",
                
                "english": """Please analyze the following text for rumor detection.

Text content: {text}

Please choose the most appropriate classification from the following three categories:
1. Non-rumor: Content is truthful and credible, supported by reliable sources
2. Rumor: Content is false or misleading, lacking factual basis
3. Unverified: Cannot determine authenticity, requires further verification

Analysis result: """,
                
                "mixed": """请对以下文本进行谣言检测分析 (Please analyze the following text for rumor detection).

文本内容 (Text content): {text}

请从以下三个类别中选择最合适的分类 (Please choose from the following categories):
1. Non-rumor (非谣言): 内容真实可信 (Content is truthful and credible)
2. Rumor (谣言): 内容虚假或误导 (Content is false or misleading)
3. Unverified (未验证): 无法确定真伪 (Cannot determine authenticity)

分析结果 (Analysis result): """
            },
            
            "conversational": {
                "chinese": """你好！我需要你帮我判断一段文字是不是谣言。

这段文字是：{text}

你觉得这段话：
- 是真的吗？（非谣言）
- 是假的吗？（谣言）
- 还是不太确定？（未验证）

你的判断：""",
                
                "english": """Hi! I need your help to determine if a piece of text is a rumor.

Here's the text: {text}

What do you think about this text:
- Is it true? (Non-rumor)
- Is it false? (Rumor)
- Or are you unsure? (Unverified)

Your judgment: """,
                
                "mixed": """你好！我需要你帮我判断一段文字是不是谣言。
Hi! I need your help to determine if this text is a rumor.

文字内容 (Text): {text}

请选择 (Please choose):
- 非谣言 (Non-rumor): 内容真实
- 谣言 (Rumor): 内容虚假
- 未验证 (Unverified): 不确定

你的判断 (Your judgment): """
            },
            
            "detailed": {
                "chinese": """作为一名专业的事实核查专家，请对以下文本进行详细的谣言检测分析。

待分析文本：{text}

请按照以下步骤进行分析：
1. 识别文本中的关键信息和声明
2. 评估信息的可信度和来源
3. 考虑是否存在明显的误导性内容
4. 综合判断文本的真实性

分类标准：
- 非谣言：基于事实的真实信息，有可靠证据支持
- 谣言：包含虚假信息或严重误导性内容
- 未验证：信息真伪难以确定，需要更多证据

最终分类：""",
                
                "english": """As a professional fact-checker, please conduct a detailed rumor detection analysis of the following text.

Text to analyze: {text}

Please follow these analysis steps:
1. Identify key information and claims in the text
2. Assess the credibility and sources of information
3. Consider whether there is obviously misleading content
4. Make a comprehensive judgment on the authenticity of the text

Classification criteria:
- Non-rumor: Factual and truthful information with reliable evidence
- Rumor: Contains false information or seriously misleading content
- Unverified: Truth is difficult to determine, requires more evidence

Final classification: """,
                
                "mixed": """作为专业事实核查专家 (As a professional fact-checker)，请详细分析以下文本。

待分析文本 (Text to analyze): {text}

分析步骤 (Analysis steps):
1. 识别关键信息 (Identify key information)
2. 评估可信度 (Assess credibility)  
3. 检查误导性内容 (Check for misleading content)
4. 综合判断 (Comprehensive judgment)

分类标准 (Classification criteria):
- Non-rumor (非谣言): 真实可信的信息
- Rumor (谣言): 虚假或误导性信息
- Unverified (未验证): 真伪难以确定

最终分类 (Final classification): """
            }
        }
        
        return templates
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """
        创建谣言检测提示
        
        Args:
            text: 待分析的文本
            **kwargs: 其他参数
            
        Returns:
            格式化的提示字符串
        """
        template = self.templates[self.style][self.language]
        return template.format(text=text, **kwargs)
    
    def create_few_shot_prompt(self, text: str, examples: List[Dict], **kwargs) -> str:
        """
        创建少样本学习提示
        
        Args:
            text: 待分析的文本
            examples: 示例列表
            **kwargs: 其他参数
            
        Returns:
            少样本提示字符串
        """
        if self.language == "chinese":
            prompt = "以下是一些谣言检测的例子：\n\n"
            example_template = "文本：{text}\n分类：{label}\n"
            query_prefix = "现在请分析以下文本：\n文本：{text}\n分类："
        elif self.language == "english":
            prompt = "Here are some examples of rumor detection:\n\n"
            example_template = "Text: {text}\nClassification: {label}\n"
            query_prefix = "Now please analyze the following text:\nText: {text}\nClassification: "
        else:  # mixed
            prompt = "以下是谣言检测的例子 (Here are examples of rumor detection):\n\n"
            example_template = "文本 (Text): {text}\n分类 (Classification): {label}\n"
            query_prefix = "现在请分析 (Now analyze): {text}\n分类 (Classification): "
        
        # 添加示例
        for i, example in enumerate(examples[:5], 1):  # 最多5个例子
            prompt += f"例子{i} (Example {i}):\n"
            prompt += example_template.format(
                text=example.get('text', ''),
                label=self._format_label(example.get('label', ''))
            )
            prompt += "\n"
        
        # 添加查询
        prompt += query_prefix.format(text=text)
        
        return prompt
    
    def _format_label(self, label: Union[str, int]) -> str:
        """格式化标签"""
        if isinstance(label, int):
            label_map = {0: "Non-rumor", 1: "Rumor", 2: "Unverified"}
            label = label_map.get(label, "Non-rumor")
        
        format_map = self.format_labels()
        return format_map.get(label, label)
    
    def create_chain_of_thought_prompt(self, text: str, **kwargs) -> str:
        """创建思维链提示"""
        if self.language == "chinese":
            return f"""请一步步分析以下文本是否为谣言。

文本：{text}

请按以下步骤思考：
1. 这个文本在说什么？
2. 有什么证据支持或反驳这个说法？
3. 信息来源是否可靠？
4. 语言表达是否存在夸张或误导？
5. 基于以上分析，你的结论是什么？

逐步分析："""
        elif self.language == "english":
            return f"""Please analyze step by step whether the following text is a rumor.

Text: {text}

Please think through these steps:
1. What is this text claiming?
2. What evidence supports or contradicts this claim?
3. Is the information source reliable?
4. Is the language exaggerated or misleading?
5. Based on the above analysis, what is your conclusion?

Step-by-step analysis: """
        else:  # mixed
            return f"""请一步步分析以下文本 (Please analyze step by step):

文本 (Text): {text}

思考步骤 (Thinking steps):
1. 文本内容 (Content): 这在说什么？
2. 证据 (Evidence): 有什么支持或反驳？
3. 来源 (Source): 信息来源可靠吗？
4. 语言 (Language): 是否夸张误导？
5. 结论 (Conclusion): 最终判断是什么？

逐步分析 (Analysis): """


class ChainOfThoughtTemplate(BasePromptTemplate):
    """思维链提示模板"""
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """创建思维链提示"""
        if self.language == "chinese":
            return f"""让我们一步一步分析这个文本是否为谣言。

文本内容：{text}

分析过程：
第一步：识别关键信息
第二步：评估信息可信度
第三步：检查逻辑一致性
第四步：得出最终结论

开始分析："""
        else:
            return f"""Let's analyze step by step whether this text is a rumor.

Text: {text}

Analysis process:
Step 1: Identify key information
Step 2: Assess information credibility
Step 3: Check logical consistency
Step 4: Draw final conclusion

Begin analysis: """


class PromptManager:
    """提示管理器"""
    
    def __init__(self, default_language: str = "mixed", default_style: str = "formal"):
        """
        初始化提示管理器
        
        Args:
            default_language: 默认语言
            default_style: 默认风格
        """
        self.default_language = default_language
        self.default_style = default_style
        
        # 初始化不同类型的模板
        self.rumor_template = RumorPromptTemplate(default_language, default_style)
        self.cot_template = ChainOfThoughtTemplate(default_language)
        
        # 预定义的少样本示例
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """加载少样本示例"""
        return {
            "chinese": [
                {
                    "text": "中国科学院发布最新研究成果，在量子计算领域取得重大突破",
                    "label": "Non-rumor",
                    "explanation": "来自权威科研机构的官方发布"
                },
                {
                    "text": "网传某市明天将发生大地震，请大家做好准备",
                    "label": "Rumor", 
                    "explanation": "地震预测传言，缺乏科学依据"
                },
                {
                    "text": "据不完全统计，新产品市场反响良好",
                    "label": "Unverified",
                    "explanation": "信息模糊，缺乏具体数据支撑"
                }
            ],
            "english": [
                {
                    "text": "NASA announces successful launch of new space telescope mission",
                    "label": "Non-rumor",
                    "explanation": "Official announcement from authoritative space agency"
                },
                {
                    "text": "Breaking: Celebrity found dead in apparent suicide, family denies",
                    "label": "Rumor",
                    "explanation": "Sensational claim without verified sources"
                },
                {
                    "text": "Sources suggest major company merger talks are underway",
                    "label": "Unverified", 
                    "explanation": "Vague sources, requires official confirmation"
                }
            ],
            "mixed": [
                {
                    "text": "教育部发布新的高考改革政策，将于明年实施",
                    "label": "Non-rumor",
                    "explanation": "Official government policy announcement"
                },
                {
                    "text": "Viral video claims new health supplement cures all diseases",
                    "label": "Rumor",
                    "explanation": "Medical misinformation without scientific evidence"
                },
                {
                    "text": "业内人士透露，某互联网公司可能进行大规模裁员",
                    "label": "Unverified",
                    "explanation": "Industry rumors requiring official confirmation"
                }
            ]
        }
    
    def create_classification_prompt(self, text: str, 
                                   language: Optional[str] = None,
                                   style: Optional[str] = None) -> str:
        """
        创建分类提示
        
        Args:
            text: 待分析文本
            language: 语言（可选）
            style: 风格（可选）
            
        Returns:
            格式化的提示
        """
        lang = language or self.default_language
        sty = style or self.default_style
        
        template = RumorPromptTemplate(lang, sty)
        return template.create_prompt(text)
    
    def create_few_shot_prompt(self, text: str,
                              examples: Optional[List[Dict]] = None,
                              language: Optional[str] = None,
                              style: Optional[str] = None,
                              num_examples: int = 3) -> str:
        """
        创建少样本提示
        
        Args:
            text: 待分析文本
            examples: 自定义示例（可选）
            language: 语言（可选）
            style: 风格（可选）
            num_examples: 示例数量
            
        Returns:
            少样本提示
        """
        lang = language or self.default_language
        sty = style or self.default_style
        
        if examples is None:
            examples = self.few_shot_examples.get(lang, self.few_shot_examples["mixed"])
            examples = random.sample(examples, min(num_examples, len(examples)))
        
        template = RumorPromptTemplate(lang, sty)
        return template.create_few_shot_prompt(text, examples)
    
    def create_chain_of_thought_prompt(self, text: str,
                                     language: Optional[str] = None) -> str:
        """
        创建思维链提示
        
        Args:
            text: 待分析文本
            language: 语言（可选）
            
        Returns:
            思维链提示
        """
        lang = language or self.default_language
        template = RumorPromptTemplate(lang)
        return template.create_chain_of_thought_prompt(text)
    
    def create_multi_perspective_prompt(self, text: str,
                                      language: Optional[str] = None) -> str:
        """
        创建多角度分析提示
        
        Args:
            text: 待分析文本
            language: 语言（可选）
            
        Returns:
            多角度提示
        """
        lang = language or self.default_language
        
        if lang == "chinese":
            return f"""请从多个角度分析以下文本的真实性：

文本：{text}

分析角度：
1. 内容角度：信息是否符合常识和逻辑？
2. 来源角度：是否有可靠的信息来源？
3. 语言角度：表达方式是否客观中性？
4. 传播角度：传播方式是否异常？
5. 时效角度：信息是否具有时效性？

综合分析："""
        elif lang == "english":
            return f"""Please analyze the authenticity of the following text from multiple perspectives:

Text: {text}

Analysis perspectives:
1. Content: Does the information align with common sense and logic?
2. Source: Are there reliable information sources?
3. Language: Is the expression objective and neutral?
4. Dissemination: Is the spreading pattern unusual?
5. Timeliness: Is the information timely and relevant?

Comprehensive analysis: """
        else:  # mixed
            return f"""请从多角度分析文本真实性 (Analyze authenticity from multiple perspectives):

文本 (Text): {text}

分析角度 (Perspectives):
1. 内容 (Content): 是否符合逻辑？
2. 来源 (Source): 信息来源可靠吗？
3. 语言 (Language): 表达是否客观？
4. 传播 (Spread): 传播方式正常吗？
5. 时效 (Time): 信息是否及时？

综合分析 (Analysis): """
    
    def get_prompt_variants(self, text: str, prompt_type: str = "classification") -> List[str]:
        """
        获取同一文本的多种提示变体
        
        Args:
            text: 待分析文本
            prompt_type: 提示类型
            
        Returns:
            提示变体列表
        """
        variants = []
        
        if prompt_type == "classification":
            # 不同风格的分类提示
            for style in ["formal", "conversational", "detailed"]:
                variants.append(self.create_classification_prompt(text, style=style))
        
        elif prompt_type == "few_shot":
            # 不同示例数量的少样本提示
            for num in [1, 3, 5]:
                variants.append(self.create_few_shot_prompt(text, num_examples=num))
        
        elif prompt_type == "advanced":
            # 高级提示技术
            variants.extend([
                self.create_chain_of_thought_prompt(text),
                self.create_multi_perspective_prompt(text)
            ])
        
        return variants
    
    def save_templates(self, save_path: str):
        """保存提示模板"""
        templates_data = {
            "rumor_templates": self.rumor_template.templates,
            "few_shot_examples": self.few_shot_examples,
            "default_language": self.default_language,
            "default_style": self.default_style
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(templates_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 提示模板已保存到: {save_path}")
    
    def load_templates(self, load_path: str):
        """加载提示模板"""
        with open(load_path, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)
        
        self.rumor_template.templates = templates_data.get("rumor_templates", {})
        self.few_shot_examples = templates_data.get("few_shot_examples", {})
        self.default_language = templates_data.get("default_language", "mixed")
        self.default_style = templates_data.get("default_style", "formal")
        
        print(f"✅ 提示模板已从 {load_path} 加载")


def demo_prompt_engineering():
    """演示提示工程功能"""
    print("🎯 提示工程演示")
    print("=" * 50)
    
    # 创建提示管理器
    prompt_manager = PromptManager(default_language="mixed", default_style="formal")
    
    # 测试文本
    test_text = "网传某地将发生大地震，专家建议市民提前撤离"
    
    print(f"测试文本: {test_text}\n")
    
    # 1. 基础分类提示
    print("1. 基础分类提示:")
    basic_prompt = prompt_manager.create_classification_prompt(test_text)
    print(basic_prompt)
    print("-" * 30)
    
    # 2. 少样本学习提示
    print("2. 少样本学习提示:")
    few_shot_prompt = prompt_manager.create_few_shot_prompt(test_text)
    print(few_shot_prompt)
    print("-" * 30)
    
    # 3. 思维链提示
    print("3. 思维链提示:")
    cot_prompt = prompt_manager.create_chain_of_thought_prompt(test_text)
    print(cot_prompt)
    print("-" * 30)
    
    # 4. 多角度分析提示
    print("4. 多角度分析提示:")
    multi_prompt = prompt_manager.create_multi_perspective_prompt(test_text)
    print(multi_prompt)
    print("-" * 30)
    
    # 5. 不同风格对比
    print("5. 不同风格对比:")
    styles = ["formal", "conversational", "detailed"]
    for style in styles:
        print(f"\n{style.upper()} 风格:")
        style_prompt = prompt_manager.create_classification_prompt(test_text, style=style)
        print(style_prompt[:200] + "...")
    
    # 6. 保存模板
    save_path = "outputs/prompt_templates.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    prompt_manager.save_templates(save_path)
    
    print(f"\n✅ 提示工程演示完成!")


if __name__ == "__main__":
    demo_prompt_engineering()