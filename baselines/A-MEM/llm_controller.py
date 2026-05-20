#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_controller.py
LLM Controller, supports only OpenAI backend, supports multimodal input
"""

import json
import re
import os
import time
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod


class BaseLLMController(ABC):
    """Base class for LLM controller"""
    
    @abstractmethod
    def get_completion(self, 
                      messages: Union[str, List[Dict]], 
                      response_format: Optional[Dict] = None, 
                      temperature: float = 0.7) -> str:
        """
        Get LLM completion result
        
        Args:
            messages: Can be a string (prompt) or OpenAI format message list
            response_format: Response format
            temperature: Temperature parameter
        
        Returns:
            LLM response text
        """
        pass


class OpenAIController(BaseLLMController):
    """OpenAI controller - supports json_schema format and multimodal"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, 
                 base_url: str = "", max_retries: int = 3, retry_delay: float = 1.0):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.api_key = api_key
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def get_completion(self, 
                      messages: Union[str, List[Dict]], 
                      response_format: Optional[Dict] = None,
                      temperature: float = 0.7) -> str:
        """
        Get LLM completion result, supports string or message list (multimodal)
        
        Args:
            messages: String (text-only) or message list (can contain images)
            response_format: Response format
            temperature: Temperature parameter
        """
        # Convert to message list format
        if isinstance(messages, str):
            # Text-only mode
            formatted_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]
        else:
            # Already in message list format (may contain images)
            # Check if system message exists, add if not
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
        
        # OpenAI supports two response_format types: json_object and json_schema
        if response_format:
            if response_format.get("type") == "json_schema":
                kwargs["response_format"] = response_format
            else:
                kwargs["response_format"] = {"type": "json_object"}
        
        # Retry mechanism
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"⚠️ API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"   Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API call failed, reached maximum retries: {e}")
        
        raise last_exception


class LLMController:
    """Unified LLM controller, supports only OpenAI, supports multimodal"""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: str = "",
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.llm = OpenAIController(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def get_completion(self, 
                      messages: Union[str, List[Dict]], 
                      response_format: Optional[Dict] = None,
                      temperature: float = 0.7) -> str:
        """
        Unified interface for getting LLM completion results, supports multimodal
        
        Args:
            messages: String (text-only) or message list (can contain images)
            response_format: Response format
            temperature: Temperature parameter
        
        Returns:
            LLM response text
        
        Examples:
            # Text-only call
            response = llm_controller.get_completion("Hello")
            
            # Multimodal call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                        {"type": "text", "text": "Please describe this image"}
                    ]
                }
            ]
            response = llm_controller.get_completion(messages)
        """
        return self.llm.get_completion(messages, response_format, temperature)
    
    def analyze_content(self, content: str, max_retries: int = 3) -> Dict[str, Any]:
        """Analyze content, extract keywords, context, and tags, with retry mechanism"""
        
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
        
        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.get_completion(
                    prompt,
                    response_format=response_format,
                    temperature=0.3
                )
                
                # Clean response
                response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
                
                # Extract JSON
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
                    print(f"⚠️ JSON parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Response content: {response[:200]}...")
                    print(f"   Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ JSON parsing failed, reached maximum retries")
                    
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ LLM analysis failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM analysis failed, reached maximum retries: {e}")
        
        # All retries failed, return default values
        print(f"Using default analysis result")
        return {
            "keywords": ["conversation"],
            "context": "General",
            "tags": ["conversation"]
        }