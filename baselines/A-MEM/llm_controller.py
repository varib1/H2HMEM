#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_controller.py
LLM控制器，仅支持OpenAI后端，支持多模态输入
"""

import json
import re
import os
import time
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod


class BaseLLMController(ABC):
    """LLM控制器基类"""
    
    @abstractmethod
    def get_completion(self, 
                      messages: Union[str, List[Dict]], 
                      response_format: Optional[Dict] = None, 
                      temperature: float = 0.7) -> str:
        """
        获取LLM完成结果
        
        Args:
            messages: 可以是字符串（prompt）或OpenAI格式的消息列表
            response_format: 响应格式
            temperature: 温度参数
        
        Returns:
            LLM响应文本
        """
        pass


class OpenAIController(BaseLLMController):
    """OpenAI控制器 - 支持json_schema格式和多模态"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, 
                 base_url: str = "", max_retries: int = 3, retry_delay: float = 1.0):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url

        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def get_completion(self, 
                      messages: Union[str, List[Dict]], 
                      response_format: Optional[Dict] = None,
                      temperature: float = 0.7) -> str:
        """
        获取LLM完成结果，支持字符串或消息列表（多模态）
        
        Args:
            messages: 字符串（纯文本）或消息列表（可包含图片）
            response_format: 响应格式
            temperature: 温度参数
        """
        # 统一转换为消息列表格式
        if isinstance(messages, str):
            # 纯文本模式
            formatted_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]
        else:
            # 已经是消息列表格式（可能包含图片）
            # 检查是否已有system消息，如果没有则添加
            has_system = any(msg.get("role") == "system" for msg in messages)
            if not has_system:
                formatted_messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ] + messages
            else:
                formatted_messages = messages
        
        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        # OpenAI支持两种response_format: json_object 和 json_schema
        if response_format:
            if response_format.get("type") == "json_schema":
                kwargs["response_format"] = response_format
            else:
                kwargs["response_format"] = {"type": "json_object"}
        
        # 重试机制
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"⚠️ API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    print(f"   等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API调用失败，已达最大重试次数: {e}")
        
        raise last_exception


class LLMController:
    """统一的LLM控制器，仅支持OpenAI，支持多模态"""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: str = "",
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        
        self.model = model
        self.llm = OpenAIController(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def get_completion(self, 
                      messages: Union[str, List[Dict]], 
                      response_format: Optional[Dict] = None,
                      temperature: float = 0.7) -> str:
        """
        统一的获取LLM完成结果接口，支持多模态
        
        Args:
            messages: 字符串（纯文本）或消息列表（可包含图片）
            response_format: 响应格式
            temperature: 温度参数
        
        Returns:
            LLM响应文本
        
        Examples:
            # 纯文本调用
            response = llm_controller.get_completion("你好")
            
            # 多模态调用
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                        {"type": "text", "text": "请描述这张图片"}
                    ]
                }
            ]
            response = llm_controller.get_completion(messages)
        """
        return self.llm.get_completion(messages, response_format, temperature)
    
    def analyze_content(self, content: str, max_retries: int = 3) -> Dict[str, Any]:
        """分析内容，提取关键词、上下文和标签，带重试机制"""
        
        prompt = f"""Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object with keys: keywords, context, tags.

Content for analysis:
{content}"""
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "context": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["keywords", "context", "tags"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        # 重试机制
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.get_completion(
                    prompt,
                    response_format=response_format,
                    temperature=0.3
                )
                
                # 清理响应
                response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
                
                # 提取JSON
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    result = json.loads(response)
                
                return {
                    "keywords": result.get("keywords", []),
                    "context": result.get("context", "General"),
                    "tags": result.get("tags", [])
                }
                
            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"   响应内容: {response[:200]}...")
                    print(f"   等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ JSON解析失败，已达最大重试次数")
                    
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ LLM分析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"   等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM分析失败，已达最大重试次数: {e}")
        
        # 所有重试都失败，返回默认值
        print(f"使用默认分析结果")
        return {
            "keywords": ["对话"],
            "context": "General",
            "tags": ["对话"]
        }