#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# preprocessing/text_processing.py

"""
MR2文本预处理模块
专门处理中英文混合文本的分词、清洗、标准化等功能
支持谣言检测任务的特殊需求
"""

import re
import string
import emoji
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import sys

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 中文分词
try:
    import jieba
    import jieba.posseg as pseg
    HAS_JIEBA = True
except ImportError:
    print("⚠️  jieba未安装，中文分词功能不可用")
    HAS_JIEBA = False

# 英文处理
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    HAS_NLTK = True
except ImportError:
    print("⚠️  nltk未安装，英文高级处理功能不可用")
    HAS_NLTK = False

# 配置管理
try:
    from utils.config_manager import get_data_config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

import logging
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    文本预处理器
    支持中英文混合文本的全面处理
    """
    
    def __init__(self, language: str = 'mixed'):
        """
        初始化文本处理器
        
        Args:
            language: 语言类型 ('chinese', 'english', 'mixed')
        """
        self.language = language
        self.setup_processors()
        self.load_stopwords()
        
        # 加载配置
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
        self.normalize_whitespace = self.processing_config.get('normalize_whitespace', True)
    
    def setup_processors(self):
        """设置各种处理器"""
        # 英文处理器
        if HAS_NLTK:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        
        # 中文停用词
        self.chinese_stopwords = {
            '的', '了', '在', '是', '和', '有', '我', '你', '他', '她', '它',
            '我们', '你们', '他们', '这', '那', '这个', '那个', '上', '下',
            '中', '大', '小', '多', '少', '好', '坏', '对', '错', '没', '不',
            '就', '都', '会', '说', '来', '去', '从', '到', '把', '被', '给',
            '向', '往', '里', '外', '前', '后', '左', '右', '东', '西', '南', '北'
        }
        
        # 英文停用词
        if HAS_NLTK:
            try:
                self.english_stopwords = set(stopwords.words('english'))
            except LookupError:
                print("⚠️  NLTK停用词未下载，使用默认停用词")
                self.english_stopwords = {
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through',
                    'during', 'before', 'after', 'above', 'below', 'up', 'down',
                    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                    'then', 'once'
                }
        else:
            self.english_stopwords = set()
    
    def load_stopwords(self):
        """加载停用词"""
        pass  # 已在setup_processors中处理
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言类型 ('chinese', 'english', 'mixed')
        """
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字符
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.7:
            return 'chinese'
        elif english_ratio > 0.7:
            return 'english'
        else:
            return 'mixed'
    
    def clean_text(self, text: str) -> str:
        """
        文本清洗
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. 移除URL
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 2. 移除@提及
        if self.remove_mentions:
            text = re.sub(r'@[a-zA-Z0-9_\u4e00-\u9fff]+', '', text)
        
        # 3. 移除#话题标签（可选）
        if self.remove_hashtags:
            text = re.sub(r'#[a-zA-Z0-9_\u4e00-\u9fff]+', '', text)
        
        # 4. 移除emoji（转换为文本描述）
        text = emoji.demojize(text, language='en')
        
        # 5. 移除多余的标点符号
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # 6. 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 7. 标准化空白字符
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 中文文本
            
        Returns:
            分词结果列表
        """
        if not HAS_JIEBA:
            # 简单的字符级分割
            return list(text.replace(' ', ''))
        
        # 使用jieba分词
        tokens = list(jieba.cut(text, cut_all=False))
        
        # 过滤停用词和无意义tokens
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if (len(token) > 1 and 
                token not in self.chinese_stopwords and
                not re.match(r'^[\s\d\W]+$', token)):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def tokenize_english(self, text: str) -> List[str]:
        """
        英文分词
        
        Args:
            text: 英文文本
            
        Returns:
            分词结果列表
        """
        if HAS_NLTK:
            try:
                tokens = word_tokenize(text.lower())
            except LookupError:
                # 如果nltk数据未下载，使用简单分割
                tokens = text.lower().split()
        else:
            tokens = text.lower().split()
        
        # 过滤停用词和标点符号
        filtered_tokens = []
        for token in tokens:
            if (token not in self.english_stopwords and
                token not in string.punctuation and
                len(token) > 2 and
                token.isalpha()):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def tokenize_mixed(self, text: str) -> List[str]:
        """
        中英文混合分词
        
        Args:
            text: 混合文本
            
        Returns:
            分词结果列表
        """
        tokens = []
        
        # 将文本按中英文分割
        parts = re.split(r'([\u4e00-\u9fff]+)', text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # 判断是中文还是英文
            if re.search(r'[\u4e00-\u9fff]', part):
                # 中文部分
                chinese_tokens = self.tokenize_chinese(part)
                tokens.extend(chinese_tokens)
            else:
                # 英文部分
                english_tokens = self.tokenize_english(part)
                tokens.extend(english_tokens)
        
        return tokens
    
    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        统一分词接口
        
        Args:
            text: 输入文本
            language: 指定语言，None则自动检测
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        # 清洗文本
        text = self.clean_text(text)
        
        if not text:
            return []
        
        # 确定语言
        if language is None:
            language = self.detect_language(text)
        
        # 根据语言选择分词方法
        if language == 'chinese':
            return self.tokenize_chinese(text)
        elif language == 'english':
            return self.tokenize_english(text)
        else:  # mixed or unknown
            return self.tokenize_mixed(text)
    
    def extract_features(self, text: str) -> Dict[str, Union[int, float, List[str]]]:
        """
        提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        # 基础特征
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'language': self.detect_language(text)
        }
        
        # 分词
        tokens = self.tokenize(text)
        features['tokens'] = tokens
        features['token_count'] = len(tokens)
        
        # 标点符号统计
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['period_count'] = text.count('.')
        features['comma_count'] = text.count(',')
        
        # 大写字母比例（英文）
        if re.search(r'[a-zA-Z]', text):
            english_chars = re.findall(r'[a-zA-Z]', text)
            if english_chars:
                uppercase_ratio = sum(1 for c in english_chars if c.isupper()) / len(english_chars)
                features['uppercase_ratio'] = uppercase_ratio
            else:
                features['uppercase_ratio'] = 0.0
        else:
            features['uppercase_ratio'] = 0.0
        
        # 数字统计
        features['digit_count'] = len(re.findall(r'\d', text))
        
        # URL和提及统计
        features['url_count'] = len(re.findall(r'http[s]?://\S+', text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        
        # emoji统计
        features['emoji_count'] = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        
        return features
    
    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        批量预处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            预处理结果列表
        """
        results = []
        for text in texts:
            result = {
                'original_text': text,
                'cleaned_text': self.clean_text(text),
                'tokens': self.tokenize(text),
                'features': self.extract_features(text)
            }
            results.append(result)
        return results


# 使用示例和测试代码
if __name__ == "__main__":
    print("🔄 测试文本处理模块")
    
    # 创建文本处理器
    processor = TextProcessor(language='mixed')
    
    # 测试文本
    test_texts = [
        "这是一个测试文本 This is a test text!",
        "今天天气不错，适合出门游玩。",
        "Breaking news: AI technology advances rapidly!",
        "混合语言文本 with English words and 中文字符",
        "包含URL的文本 https://example.com 和@username提及",
        "带有emoji的文本 😊 和 #hashtag 标签"
    ]
    
    print("\n📝 === 文本处理测试 ===")
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
        print(f"  分词结果: {tokens}")
        
        # 特征提取
        features = processor.extract_features(text)
        print(f"  特征: 长度={features['text_length']}, 词数={features['token_count']}, 语言={features['language']}")
    
    print("\n🔄 测试批量处理")
    batch_results = processor.preprocess_batch(test_texts[:3])
    for i, result in enumerate(batch_results):
        print(f"  批量结果 {i+1}: {len(result['tokens'])} 个token")
    
    print("\n✅ 文本处理模块测试完成")