#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/siliconflow_sentiment.py

"""
硅基流动方面级情感分析模块
基于Qwen/Qwen3-8B模型实现方面级情感分析
支持多维度情感识别和结构化输出
"""

import requests
import json
from typing import Dict, List, Optional, Union

class SiliconFlowSentimentAnalyzer:
    """硅基流动方面级情感分析器"""
    
    def __init__(self, api_key: str = "sk-pawkjgijrgqjubqatearwfyccmktwoddeptzjdzteeuxdrak"):
        """
        初始化分析器
        
        Args:
            api_key: 硅基流动API密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_aspect_sentiment_prompt(self, text: str) -> str:
        """
        创建方面级情感分析的prompt
        
        Args:
            text: 待分析的文本
            
        Returns:
            构造好的prompt字符串
        """
        prompt = f"""You are an expert in aspect-based sentiment analysis. Analyze the given sentence and identify:

1. **Aspects**: All specific aspects, features, or topics mentioned
2. **Sentiment**: The sentiment (positive, negative, or neutral) for each aspect
3. **Confidence**: Your confidence level (high, medium, low) for each prediction

Provide analysis in this JSON format:
{{
    "sentence": "the original sentence",
    "aspects": [
        {{
            "aspect": "aspect name",
            "sentiment": "positive/negative/neutral",
            "confidence": "high/medium/low",
            "reasoning": "brief explanation"
        }}
    ],
    "overall_sentiment": "positive/negative/neutral"
}}

Sentence: "{text}"

Respond with JSON only:"""
        
        return prompt
    
    def analyze_sentiment(self, text: str, model: str = "Qwen/Qwen3-8B") -> Dict:
        """
        执行方面级情感分析
        
        Args:
            text: 待分析的文本
            model: 使用的模型名称
            
        Returns:
            分析结果字典
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": self.create_aspect_sentiment_prompt(text)
                }
            ],
            "stream": False,
            "max_tokens": 1000,
            "temperature": 0.1,  # 低温度保证一致性
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.3,
            "n": 1,
            "response_format": {"type": "text"}
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            assistant_response = result['choices'][0]['message']['content']
            
            # 尝试解析JSON响应
            try:
                sentiment_analysis = json.loads(assistant_response)
                return {
                    "success": True,
                    "data": sentiment_analysis,
                    "raw_response": assistant_response
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse JSON response",
                    "raw_response": assistant_response
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "raw_response": None
            }
        except KeyError as e:
            return {
                "success": False,
                "error": f"Unexpected response format: {str(e)}",
                "raw_response": None
            }
    
    def batch_analyze(self, texts: List[str], model: str = "Qwen/Qwen3-8B") -> List[Dict]:
        """
        批量分析多个文本
        
        Args:
            texts: 文本列表
            model: 使用的模型名称
            
        Returns:
            分析结果列表
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text, model)
            results.append(result)
        return results
    
    def print_analysis_result(self, result: Dict, show_raw: bool = False):
        """
        格式化打印分析结果
        
        Args:
            result: 分析结果
            show_raw: 是否显示原始响应
        """
        if not result["success"]:
            print(f"❌ Error: {result['error']}")
            if show_raw and result.get('raw_response'):
                print(f"Raw response: {result['raw_response']}")
            return
        
        data = result["data"]
        print(f"📝 Sentence: {data.get('sentence', 'N/A')}")
        print(f"🎯 Overall Sentiment: {data.get('overall_sentiment', 'N/A')}")
        print("📊 Aspect Analysis:")
        
        for i, aspect in enumerate(data.get('aspects', []), 1):
            sentiment_emoji = {
                'positive': '😊',
                'negative': '😞',
                'neutral': '😐'
            }.get(aspect.get('sentiment', '').lower(), '❓')
            
            confidence_emoji = {
                'high': '🔥',
                'medium': '⚡',
                'low': '💭'
            }.get(aspect.get('confidence', '').lower(), '❓')
            
            print(f"   {i}. {aspect.get('aspect', 'N/A')} {sentiment_emoji}")
            print(f"      Sentiment: {aspect.get('sentiment', 'N/A')} {confidence_emoji}")
            print(f"      Reasoning: {aspect.get('reasoning', 'N/A')}")
        
        if show_raw:
            print(f"\n🔍 Raw Response:\n{result['raw_response']}")


def main():
    """主函数 - 测试示例"""
    
    # 初始化分析器
    analyzer = SiliconFlowSentimentAnalyzer()
    
    # 测试句子
    test_sentences = [
        "The food was delicious but the service was terrible and slow.",
        "I love the camera quality of this phone, but the battery life is disappointing.",
        "The hotel room was clean and comfortable, however the wifi was unreliable.",
        "Great product design and user interface, but customer support needs improvement.",
        "This restaurant has amazing atmosphere and friendly staff, though parking is limited."
    ]
    
    print("=" * 80)
    print("🔍 ASPECT-BASED SENTIMENT ANALYSIS USING SILICONFLOW")
    print("=" * 80)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📋 Example {i}:")
        print("─" * 50)
        
        # 执行分析
        result = analyzer.analyze_sentiment(sentence)
        
        # 打印结果
        analyzer.print_analysis_result(result)
        
        print("─" * 50)


def quick_test():
    """快速测试单个句子"""
    analyzer = SiliconFlowSentimentAnalyzer()
    
    sentence = "The restaurant has amazing food but terrible service."
    print(f"🧪 Quick Test: {sentence}")
    print("─" * 50)
    
    result = analyzer.analyze_sentiment(sentence)
    analyzer.print_analysis_result(result, show_raw=True)
    
    return result


if __name__ == "__main__":
    # 运行主测试
    main()
    
    print("\n" + "=" * 80)
    print("🧪 QUICK TEST")
    print("=" * 80)
    
    # 运行快速测试
    quick_test()