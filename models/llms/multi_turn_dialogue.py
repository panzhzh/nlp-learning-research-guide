#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/multi_turn_dialogue.py

"""
多轮对话支持模块
实现基于Qwen3-0.6B的多轮对话系统，支持上下文记忆、对话状态管理
专门针对谣言检测任务的对话式交互
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import sys
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from utils.config_manager import get_config_manager, get_output_path
    from models.llms.prompt_engineering import PromptManager
    from models.llms.open_source_llms import QwenRumorClassifier
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False


class DialogueState(Enum):
    """对话状态枚举"""
    GREETING = "greeting"           # 问候阶段
    ANALYZING = "analyzing"         # 分析阶段
    EXPLAINING = "explaining"       # 解释阶段
    COLLECTING = "collecting"       # 收集更多信息
    CONFIRMING = "confirming"       # 确认结果
    FINISHED = "finished"           # 对话结束


@dataclass
class Message:
    """消息数据类"""
    role: str  # 'user' 或 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueContext:
    """对话上下文数据类"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    current_state: DialogueState = DialogueState.GREETING
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    user_intent: str = ""
    extracted_info: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        """初始化意图分类器"""
        self.intent_keywords = {
            'analyze_rumor': ['分析', '检测', '判断', '识别', '真假', '谣言', 'analyze', 'detect', 'check'],
            'ask_explanation': ['为什么', '怎么', '原因', '解释', 'why', 'how', 'explain'],
            'request_details': ['详细', '具体', '更多', '细节', 'detail', 'more', 'specific'],
            'confirm_result': ['确认', '肯定', '是的', '对的', 'yes', 'confirm', 'correct'],
            'deny_result': ['不对', '错误', '不是', '否定', 'no', 'wrong', 'incorrect'],
            'ask_help': ['帮助', '怎么用', '使用', 'help', 'usage', 'how to use'],
            'goodbye': ['再见', '结束', '谢谢', 'bye', 'goodbye', 'thanks', 'end']
        }
    
    def classify_intent(self, user_input: str) -> str:
        """
        分类用户意图
        
        Args:
            user_input: 用户输入
            
        Returns:
            意图类别
        """
        user_input_lower = user_input.lower()
        
        # 统计每个意图的关键词匹配数
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # 返回得分最高的意图
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'analyze_rumor'  # 默认意图


class DialogueManager:
    """对话管理器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        初始化对话管理器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.intent_classifier = IntentClassifier()
        self.prompt_manager = None
        
        # 活跃对话会话
        self.active_sessions: Dict[str, DialogueContext] = {}
        
        # 对话模板
        self.response_templates = self._load_response_templates()
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'llms/dialogue')
        else:
            self.output_dir = Path('outputs/models/llms/dialogue')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💬 对话管理器初始化完成")
        print(f"   模型: {model_name}")
        print(f"   输出目录: {self.output_dir}")
    
    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """加载响应模板"""
        return {
            DialogueState.GREETING.value: {
                'welcome': [
                    "您好！我是谣言检测助手，可以帮您分析文本内容的真实性。请发送您想要检测的文本内容。",
                    "欢迎使用谣言检测系统！请提供您需要分析的文本，我会帮您判断其真实性。",
                    "Hi! 我是专业的谣言检测AI助手，请把需要分析的内容发给我。"
                ],
                'help': [
                    "我可以帮您：\n1. 分析文本是否为谣言\n2. 解释判断的依据\n3. 提供防范谣言的建议\n请发送您要检测的文本内容。"
                ]
            },
            DialogueState.ANALYZING.value: {
                'processing': [
                    "正在分析您提供的文本内容，请稍等...",
                    "我正在仔细检查这段文本的真实性...",
                    "分析中，马上为您提供结果..."
                ],
                'need_more_info': [
                    "这段文本信息有限，您能提供更多上下文或相关信息吗？",
                    "为了更准确的分析，能否告诉我这段文本的来源或更多细节？"
                ]
            },
            DialogueState.EXPLAINING.value: {
                'explain_rumor': [
                    "这很可能是谣言，因为：",
                    "分析结果显示这是谣言，主要依据：",
                    "这段内容存在谣言特征："
                ],
                'explain_non_rumor': [
                    "这看起来是真实可信的信息，原因：",
                    "分析表明这是可信内容，依据：",
                    "这段文本具有真实信息的特征："
                ],
                'explain_unverified': [
                    "这段信息目前无法确定真伪，原因：",
                    "需要进一步验证，因为：",
                    "信息不够明确，建议谨慎对待："
                ]
            },
            DialogueState.COLLECTING.value: {
                'ask_source': [
                    "您能告诉我这个信息的来源吗？这有助于更准确的判断。",
                    "这个消息是从哪里看到的？来源信息很重要。"
                ],
                'ask_context': [
                    "能提供更多相关背景信息吗？",
                    "这个事件还有其他相关细节吗？"
                ]
            },
            DialogueState.CONFIRMING.value: {
                'confirm': [
                    "根据分析，我的判断是否符合您的预期？",
                    "您对这个分析结果还有什么疑问吗？",
                    "还需要我解释其他方面吗？"
                ]
            },
            DialogueState.FINISHED.value: {
                'goodbye': [
                    "很高兴为您服务！如果还有其他需要检测的内容，随时告诉我。",
                    "谢谢使用谣言检测服务！记得保持理性思考，谨慎传播信息。",
                    "再见！希望我的分析对您有帮助。"
                ]
            }
        }
    
    def setup_model(self) -> None:
        """设置模型和分词器"""
        try:
            print("📥 加载对话模型...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                pad_token='<|extra_0|>',
                eos_token='<|im_end|>'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to('cpu')
            
            self.model.eval()
            
            # 初始化提示管理器
            if USE_PROJECT_MODULES:
                self.prompt_manager = PromptManager()
            
            print(f"✅ 对话模型加载完成")
            
        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            raise
    
    def create_session(self) -> str:
        """
        创建新的对话会话
        
        Returns:
            会话ID
        """
        context = DialogueContext()
        self.active_sessions[context.session_id] = context
        
        # 添加欢迎消息
        welcome_msg = self._get_template_response(DialogueState.GREETING, 'welcome')
        self._add_message(context, 'assistant', welcome_msg)
        
        print(f"🆕 创建新会话: {context.session_id}")
        return context.session_id
    
    def process_user_input(self, session_id: str, user_input: str) -> str:
        """
        处理用户输入
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        if session_id not in self.active_sessions:
            return "会话不存在，请重新开始对话。"
        
        context = self.active_sessions[session_id]
        
        # 添加用户消息
        self._add_message(context, 'user', user_input)
        
        # 分类用户意图
        intent = self.intent_classifier.classify_intent(user_input)
        context.user_intent = intent
        
        # 根据当前状态和意图生成回复
        response = self._generate_response(context, user_input, intent)
        
        # 添加助手回复
        self._add_message(context, 'assistant', response)
        
        # 更新对话状态
        self._update_dialogue_state(context, intent)
        
        # 更新时间戳
        context.updated_at = datetime.now()
        
        return response
    
    def _generate_response(self, context: DialogueContext, 
                          user_input: str, intent: str) -> str:
        """
        生成回复
        
        Args:
            context: 对话上下文
            user_input: 用户输入
            intent: 用户意图
            
        Returns:
            助手回复
        """
        current_state = context.current_state
        
        # 根据状态和意图选择处理策略
        if intent == 'goodbye':
            context.current_state = DialogueState.FINISHED
            return self._get_template_response(DialogueState.FINISHED, 'goodbye')
        
        elif intent == 'ask_help':
            return self._get_template_response(DialogueState.GREETING, 'help')
        
        elif intent == 'analyze_rumor':
            # 执行谣言检测
            if current_state in [DialogueState.GREETING, DialogueState.COLLECTING]:
                return self._handle_rumor_analysis(context, user_input)
            else:
                return "请提供您要检测的文本内容。"
        
        elif intent == 'ask_explanation':
            if context.analysis_results:
                return self._provide_detailed_explanation(context)
            else:
                return "请先提供要分析的文本，我来为您检测。"
        
        elif intent == 'request_details':
            if context.analysis_results:
                return self._provide_additional_details(context)
            else:
                return "请先发送要检测的文本内容。"
        
        elif intent == 'confirm_result':
            return self._handle_confirmation(context, True)
        
        elif intent == 'deny_result':
            return self._handle_confirmation(context, False)
        
        else:
            # 默认处理逻辑
            if current_state == DialogueState.GREETING:
                return self._handle_rumor_analysis(context, user_input)
            elif current_state == DialogueState.ANALYZING:
                return "请稍等，我正在分析您的文本..."
            else:
                return "请告诉我您需要什么帮助？"
    
    def _handle_rumor_analysis(self, context: DialogueContext, text: str) -> str:
        """
        处理谣言分析请求
        
        Args:
            context: 对话上下文
            text: 要分析的文本
            
        Returns:
            分析结果回复
        """
        try:
            # 更新状态
            context.current_state = DialogueState.ANALYZING
            
            # 执行谣言检测
            analysis_result = self._analyze_text(text)
            context.analysis_results = analysis_result
            context.confidence_score = analysis_result.get('confidence', 0.0)
            
            # 提取关键信息
            context.extracted_info = {
                'analyzed_text': text,
                'prediction': analysis_result.get('predicted_class', 'Unknown'),
                'confidence': analysis_result.get('confidence', 0.0)
            }
            
            # 生成回复
            prediction = analysis_result.get('predicted_class', 'Unknown')
            confidence = analysis_result.get('confidence', 0.0)
            
            # 根据预测结果选择回复模板
            if prediction == 'Rumor':
                base_response = self._get_template_response(DialogueState.EXPLAINING, 'explain_rumor')
                context.current_state = DialogueState.EXPLAINING
            elif prediction == 'Non-rumor':
                base_response = self._get_template_response(DialogueState.EXPLAINING, 'explain_non_rumor')
                context.current_state = DialogueState.EXPLAINING
            else:  # Unverified
                base_response = self._get_template_response(DialogueState.EXPLAINING, 'explain_unverified')
                context.current_state = DialogueState.EXPLAINING
            
            # 构建详细回复
            detailed_response = f"""{base_response}

📋 **分析结果**
- 判断：{prediction}
- 置信度：{confidence:.2f}

📝 **分析的文本**
"{text}"

💡 **简要说明**
{self._get_brief_explanation(prediction, confidence)}

您想了解更多分析依据吗？或者有其他问题想要咨询？"""
            
            return detailed_response
            
        except Exception as e:
            logger.error(f"谣言分析失败: {e}")
            return f"抱歉，分析过程中出现错误：{str(e)}。请重新尝试。"
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析文本（简化版本）
        
        Args:
            text: 要分析的文本
            
        Returns:
            分析结果
        """
        # 这里应该调用实际的谣言检测模型
        # 为了演示，使用简单的规则
        
        text_lower = text.lower()
        
        # 简单的规则分类
        rumor_keywords = ['网传', '紧急', '速转', '不转不是', '震惊', '内幕', '秘密']
        reliable_keywords = ['官方', '权威', '研究', '科学', '正式', '公告']
        uncertain_keywords = ['据说', '听说', '可能', '或许', '疑似', '不确定']
        
        rumor_score = sum(1 for kw in rumor_keywords if kw in text)
        reliable_score = sum(1 for kw in reliable_keywords if kw in text)
        uncertain_score = sum(1 for kw in uncertain_keywords if kw in text)
        
        if rumor_score > max(reliable_score, uncertain_score):
            prediction = 'Rumor'
            confidence = min(0.8, 0.5 + rumor_score * 0.1)
        elif reliable_score > max(rumor_score, uncertain_score):
            prediction = 'Non-rumor'
            confidence = min(0.9, 0.6 + reliable_score * 0.1)
        else:
            prediction = 'Unverified'
            confidence = min(0.7, 0.4 + uncertain_score * 0.1)
        
        return {
            'predicted_class': prediction,
            'confidence': confidence,
            'reasoning': f'基于关键词分析：谣言特征{rumor_score}个，可信特征{reliable_score}个，不确定特征{uncertain_score}个'
        }
    
    def _get_brief_explanation(self, prediction: str, confidence: float) -> str:
        """获取简要解释"""
        if prediction == 'Rumor':
            if confidence > 0.7:
                return "文本包含明显的谣言特征，如煽动性语言、缺乏可靠来源等。"
            else:
                return "文本可能包含一些谣言特征，建议谨慎对待。"
        elif prediction == 'Non-rumor':
            if confidence > 0.8:
                return "文本来源可靠，内容符合事实，可信度较高。"
            else:
                return "文本基本可信，但建议核实具体细节。"
        else:  # Unverified
            return "文本信息有限或来源不明，需要进一步验证才能确定真伪。"
    
    def _provide_detailed_explanation(self, context: DialogueContext) -> str:
        """提供详细解释"""
        if not context.analysis_results:
            return "还没有分析结果可以解释。"
        
        prediction = context.analysis_results.get('predicted_class', 'Unknown')
        reasoning = context.analysis_results.get('reasoning', '无详细分析')
        confidence = context.analysis_results.get('confidence', 0.0)
        
        explanation = f"""📖 **详细分析解释**

🎯 **判断依据**
{reasoning}

📊 **置信度分析**
- 当前置信度：{confidence:.2f}
- 置信度解读：{'高' if confidence > 0.7 else '中等' if confidence > 0.5 else '较低'}

🔍 **建议行动**
"""
        
        if prediction == 'Rumor':
            explanation += """- 不建议转发或传播此信息
- 可以查找官方辟谣信息
- 提醒他人注意甄别"""
        elif prediction == 'Non-rumor':
            explanation += """- 信息较为可信，可以适当参考
- 如需转发，建议注明来源
- 保持理性判断"""
        else:
            explanation += """- 暂时不要传播此信息
- 等待更多权威信息
- 保持观望态度"""
        
        explanation += "\n\n您还有其他疑问吗？"
        
        return explanation
    
    def _provide_additional_details(self, context: DialogueContext) -> str:
        """提供额外细节"""
        if not context.analysis_results:
            return "请先提供要分析的文本。"
        
        analyzed_text = context.extracted_info.get('analyzed_text', '')
        
        details = f"""📋 **更多分析细节**

📝 **文本特征分析**
- 文本长度：{len(analyzed_text)} 字符
- 文本类型：{'短文本' if len(analyzed_text) < 100 else '长文本'}

🔍 **关键要素检查**
- 信息来源：{'明确' if any(kw in analyzed_text for kw in ['官方', '权威', '研究']) else '不明确'}
- 时间信息：{'包含' if any(char.isdigit() for char in analyzed_text) else '缺失'}
- 情感色彩：{'强烈' if any(kw in analyzed_text for kw in ['震惊', '紧急', '重大']) else '中性'}

💡 **识别谣言的技巧**
1. 查看信息来源是否权威
2. 注意是否使用煽动性语言
3. 核实具体时间、地点、人物
4. 查找相关官方辟谣信息

还需要我帮您分析其他内容吗？"""
        
        return details
    
    def _handle_confirmation(self, context: DialogueContext, is_confirmed: bool) -> str:
        """处理确认反馈"""
        if is_confirmed:
            context.current_state = DialogueState.FINISHED
            return """谢谢您的确认！我很高兴分析结果对您有帮助。

🎯 **温馨提示**
- 保持理性思考，独立判断
- 多渠道验证重要信息
- 不传播未经证实的消息

如果您还有其他需要检测的内容，随时告诉我！"""
        else:
            context.current_state = DialogueState.COLLECTING
            return """感谢您的反馈！看来我的分析可能不够准确。

🤔 **帮助我改进**
- 您认为正确的判断是什么？
- 能否提供更多背景信息？
- 有什么我遗漏的重要细节吗？

我会根据您的反馈重新分析。"""
    
    def _update_dialogue_state(self, context: DialogueContext, intent: str) -> None:
        """更新对话状态"""
        current_state = context.current_state
        
        if intent == 'goodbye':
            context.current_state = DialogueState.FINISHED
        elif intent == 'analyze_rumor' and current_state == DialogueState.GREETING:
            context.current_state = DialogueState.ANALYZING
        elif intent == 'ask_explanation' and current_state == DialogueState.EXPLAINING:
            context.current_state = DialogueState.EXPLAINING
        elif intent == 'deny_result':
            context.current_state = DialogueState.COLLECTING
        elif intent == 'confirm_result':
            context.current_state = DialogueState.CONFIRMING
    
    def _get_template_response(self, state: DialogueState, response_type: str) -> str:
        """获取模板回复"""
        templates = self.response_templates.get(state.value, {}).get(response_type, [])
        if templates:
            import random
            return random.choice(templates)
        return "我理解您的意思，让我来帮助您。"
    
    def _add_message(self, context: DialogueContext, role: str, content: str) -> None:
        """添加消息到上下文"""
        message = Message(role=role, content=content)
        context.messages.append(message)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if session_id not in self.active_sessions:
            return []
        
        context = self.active_sessions[session_id]
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'message_id': msg.message_id
            }
            for msg in context.messages
        ]
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        if session_id not in self.active_sessions:
            return {}
        
        context = self.active_sessions[session_id]
        return {
            'session_id': context.session_id,
            'current_state': context.current_state.value,
            'user_intent': context.user_intent,
            'confidence_score': context.confidence_score,
            'message_count': len(context.messages),
            'created_at': context.created_at.isoformat(),
            'updated_at': context.updated_at.isoformat(),
            'analysis_results': context.analysis_results,
            'extracted_info': context.extracted_info
        }
    
    def save_session(self, session_id: str) -> None:
        """保存会话"""
        if session_id not in self.active_sessions:
            return
        
        context = self.active_sessions[session_id]
        
        # 转换为可序列化的格式
        session_data = {
            'session_id': context.session_id,
            'current_state': context.current_state.value,
            'user_intent': context.user_intent,
            'confidence_score': context.confidence_score,
            'created_at': context.created_at.isoformat(),
            'updated_at': context.updated_at.isoformat(),
            'analysis_results': context.analysis_results,
            'extracted_info': context.extracted_info,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'message_id': msg.message_id,
                    'metadata': msg.metadata
                }
                for msg in context.messages
            ]
        }
        
        # 保存到文件
        save_file = self.output_dir / f"session_{session_id}.json"
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 会话已保存: {save_file}")
    
    def load_session(self, session_id: str) -> bool:
        """加载会话"""
        load_file = self.output_dir / f"session_{session_id}.json"
        
        if not load_file.exists():
            return False
        
        try:
            with open(load_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 重建对话上下文
            context = DialogueContext(
                session_id=session_data['session_id'],
                current_state=DialogueState(session_data['current_state']),
                user_intent=session_data['user_intent'],
                confidence_score=session_data['confidence_score'],
                analysis_results=session_data['analysis_results'],
                extracted_info=session_data['extracted_info'],
                created_at=datetime.fromisoformat(session_data['created_at']),
                updated_at=datetime.fromisoformat(session_data['updated_at'])
            )
            
            # 重建消息列表
            for msg_data in session_data['messages']:
                message = Message(
                    role=msg_data['role'],
                    content=msg_data['content'],
                    timestamp=datetime.fromisoformat(msg_data['timestamp']),
                    message_id=msg_data['message_id'],
                    metadata=msg_data.get('metadata', {})
                )
                context.messages.append(message)
            
            self.active_sessions[session_id] = context
            print(f"📂 会话已加载: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"加载会话失败: {e}")
            return False
    
    def end_session(self, session_id: str) -> None:
        """结束会话"""
        if session_id in self.active_sessions:
            # 保存会话
            self.save_session(session_id)
            
            # 从活跃会话中移除
            del self.active_sessions[session_id]
            
            print(f"🔚 会话已结束: {session_id}")
    
    def get_active_sessions_count(self) -> int:
        """获取活跃会话数量"""
        return len(self.active_sessions)


class DialogueInterface:
    """对话界面"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        初始化对话界面
        
        Args:
            model_name: 模型名称
        """
        self.dialogue_manager = DialogueManager(model_name)
        self.current_session_id = None
        
        print("🖥️  对话界面初始化完成")
    
    def start_conversation(self) -> str:
        """开始新对话"""
        # 设置模型（如果还未设置）
        if self.dialogue_manager.model is None:
            self.dialogue_manager.setup_model()
        
        # 创建新会话
        self.current_session_id = self.dialogue_manager.create_session()
        
        # 获取欢迎消息
        history = self.dialogue_manager.get_conversation_history(self.current_session_id)
        if history:
            return history[-1]['content']
        else:
            return "欢迎使用谣言检测对话系统！"
    
    def send_message(self, user_input: str) -> str:
        """发送消息"""
        if self.current_session_id is None:
            return self.start_conversation()
        
        return self.dialogue_manager.process_user_input(self.current_session_id, user_input)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if self.current_session_id is None:
            return []
        
        return self.dialogue_manager.get_conversation_history(self.current_session_id)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        if self.current_session_id is None:
            return {}
        
        return self.dialogue_manager.get_session_info(self.current_session_id)
    
    def end_conversation(self) -> None:
        """结束对话"""
        if self.current_session_id:
            self.dialogue_manager.end_session(self.current_session_id)
            self.current_session_id = None


def create_dialogue_system(model_name: str = "Qwen/Qwen3-0.6B") -> DialogueInterface:
    """
    创建对话系统的便捷函数
    
    Args:
        model_name: 模型名称
        
    Returns:
        对话界面实例
    """
    print("🚀 创建多轮对话系统...")
    
    dialogue_interface = DialogueInterface(model_name)
    
    return dialogue_interface


def demo_multi_turn_dialogue():
    """演示多轮对话功能"""
    print("💬 多轮对话系统演示")
    print("=" * 60)
    
    try:
        # 创建对话系统
        dialogue_system = create_dialogue_system()
        
        # 开始对话
        welcome_msg = dialogue_system.start_conversation()
        print(f"🤖 助手: {welcome_msg}")
        
        # 模拟对话场景
        test_conversations = [
            "你好，我想检测一段文本",
            "网传某地明天将发生大地震，请大家做好防护准备",
            "为什么你认为这是谣言？",
            "能提供更详细的分析吗？",
            "谢谢你的分析，很有帮助",
            "再见"
        ]
        
        print(f"\n💬 模拟对话:")
        for i, user_msg in enumerate(test_conversations, 1):
            print(f"\n👤 用户: {user_msg}")
            
            # 发送消息并获取回复
            response = dialogue_system.send_message(user_msg)
            print(f"🤖 助手: {response}")
            
            # 显示会话状态（部分轮次）
            if i in [2, 4]:
                stats = dialogue_system.get_session_stats()
                print(f"📊 当前状态: {stats.get('current_state', 'unknown')}")
                print(f"🎯 用户意图: {stats.get('user_intent', 'unknown')}")
        
        # 显示完整对话历史
        print(f"\n📚 完整对话历史:")
        history = dialogue_system.get_history()
        for i, msg in enumerate(history, 1):
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            print(f"{i:2d}. {role_emoji} {msg['role']}: {msg['content'][:100]}...")
        
        # 显示会话统计
        final_stats = dialogue_system.get_session_stats()
        print(f"\n📊 会话统计:")
        print(f"   会话ID: {final_stats.get('session_id', 'N/A')}")
        print(f"   消息数量: {final_stats.get('message_count', 0)}")
        print(f"   最终状态: {final_stats.get('current_state', 'unknown')}")
        print(f"   置信度: {final_stats.get('confidence_score', 0.0):.2f}")
        
        # 结束对话
        dialogue_system.end_conversation()
        
        print(f"\n✅ 多轮对话演示完成!")
        
    except Exception as e:
        print(f"❌ 对话演示失败: {e}")
        raise


if __name__ == "__main__":
    demo_multi_turn_dialogue()