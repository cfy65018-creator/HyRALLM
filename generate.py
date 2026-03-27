"""
LLM Generation Module.
Handles the interaction with various Large Language Model APIs (OpenAI, Claude, Gemini)
to generate source code summaries utilizing retrieved context.
"""

import os
import json
import requests
from typing import List, Dict, Any
import time

from evaluator import (
    calculate_rouge_simple,
    calculate_bleu4,
    calculate_metrics
)


class SummaryGenerator:
    """
    LLM API wrapper for code summary generation.
    Supports multiple APIs including OpenAI, Claude, and Gemini.
    """
    
    def __init__(
        self, 
        api_base_url: str = "",
        api_key: str = "",
        model: str = "",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_type: str = "auto",
        max_example_length: int = 1200,
        max_source_length: int = 1500
    ):
        
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_example_length = max_example_length
        self.max_source_length = max_source_length
        
        if api_type == "auto":
            model_lower = model.lower()
            api_type_map = {
                'claude': 'claude',
                'gpt': 'openai',

            }
            self.api_type = next(
                (api_t for keyword, api_t in api_type_map.items() if keyword in model_lower),
                'openai'
            )
        else:
            self.api_type = api_type
        
        if self.api_type == "gemini":
            self.endpoint = f"{self.api_base_url}/v1beta/models/{self.model}:generateContent"
        elif self.api_type == "claude":
            self.endpoint = f"{self.api_base_url}/v1/messages"
        else:
            self.endpoint = f"{self.api_base_url}/v1/chat/completions"
        
        print(f"🤖 Initializing LLM generator:")
        print(f"  - Model: {self.model}")
        print(f"  - API type: {self.api_type}")
        print(f"  - Endpoint: {self.endpoint}")
        
        if not self.api_key:
            print("⚠️ Warning: API key is not set, please provide api_key")
    
    def _call_api(self, prompt: str) -> Dict[str, Any]:
        
        headers = {
            'Content-Type': 'application/json'
        }
        
            
        if self.api_type == "claude":
            payload = {
                "model": self.model,
                "max_tokens": 200,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            headers['x-api-key'] = self.api_key
            headers['anthropic-version'] = '2023-06-01'
            url = self.endpoint
            
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a code summarization expert. Generate concise, accurate summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0,
                "top_p": 0.8,
                "max_tokens": 200
            }
            
            headers['Authorization'] = f'Bearer {self.api_key}'
            url = self.endpoint
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if "400" in str(e) and "response_format" in payload and attempt == 0:
                    print(f"  ℹ️ Detected HTTP 400, response_format may be unsupported, retrying without it...")
                    payload.pop("response_format", None)
                    continue
                
                if attempt < self.max_retries - 1:
                    print(f"  ⚠️ API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    try:
                        if 'response' in locals():
                            error_detail = response.json()
                            print(f"  ❌ API error details: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        pass
                    raise Exception(f"API request failed after {self.max_retries} retries: {e}")
    
    def _extract_summary(self, response: Dict[str, Any]) -> str:
        
        try:
            text = self._extract_text_from_response(response)
            
            return self._parse_summary_text(text)
        
        except Exception as e:
            return self._fallback_extract(response, e)
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        
        if self.api_type == "claude":
            return response['content'][0]['text']
        
        else:
            message = response['choices'][0]['message']
            if 'content' in message and message['content']:
                return message['content']
            else:
                return message.get('content', '')
    
    def _parse_summary_text(self, text: str) -> str:
        
        text = text.strip()
        
        try:
            result = json.loads(text)
            return self._extract_from_json(result)
        except json.JSONDecodeError:
            return text.strip('"').strip("'").strip()
    
    def _extract_from_json(self, result: Any) -> str:
        
        if isinstance(result, dict):
            for key in ['summary', 'text', 'content', 'output']:
                if key in result and result[key]:
                    return str(result[key]).strip()
            return str(result)
        
        elif isinstance(result, list) and len(result) > 0:
            first_item = result[0]
            if isinstance(first_item, dict):
                return self._extract_from_json(first_item)
            else:
                return str(first_item)
        
        else:
            return str(result)
    
    def _fallback_extract(self, response: Dict[str, Any], original_error: Exception) -> str:
        
        try:      
            
            if self.api_type == "claude":
                content = response.get('content', [])
                if content and 'text' in content[0]:
                    return content[0]['text'].strip()
            
            else:
                choices = response.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    if 'content' in message:
                        return message['content'].strip()
            
            raise original_error
            
        except Exception as inner_e:
            print(f"  ❌ Invalid response format:")
            print(f"     Original error: {original_error}")
            print(f"     Fallback error: {inner_e}")
            print(f"     Response type: {type(response)}")
            print(f"     Response content: {json.dumps(response, indent=2, ensure_ascii=False)[:500]}...")
            raise Exception(f"Cannot extract summary from response: {original_error}")
    
    def _build_example_based_prompt(
        self,
        source_code: str,
        example_code: str,
        example_summary: str
    ) -> str:
        
        def truncate_code(code: str, max_len: int) -> str:
            code_stripped = code.strip()
            return code_stripped[:max_len] + ("..." if len(code_stripped) > max_len else "")
        
        code_snippet_ref = truncate_code(example_code, self.max_example_length)
        code_snippet_target = truncate_code(source_code, self.max_source_length)
        
        prompt = f"""Task: Generate a concise function summary for the TARGET code.

REFERENCE EXAMPLE:
Code: {code_snippet_ref}
Summary: {example_summary.strip()}

TARGET CODE:
{code_snippet_target}

Instructions:
1. Read the TARGET code carefully and compare TARGET with REFERENCE code:
   - If they perform SIMILAR operations: Adapt the reference summary by changing only the specific details (variable names, types, objects)
   - If they perform DIFFERENT operations: Write a new summary following the reference style (sentence structure, verb choice)

2. Requirements:
   - Length: 6-12 words (match reference length closely)
   - Format: "<verb> <object/details> [additional context]"
   - Use EXACT technical terms from the target code (class names, method names)
   - Use lowercase, end with a period
   - Do NOT add information not present in the code

Summary:"""
        
        return prompt
    
    def generate(
        self,
        source_code: str,
        example_code: str,
        example_summary: str
    ) -> str:
        
        prompt = self._build_example_based_prompt(source_code, example_code, example_summary)
        response = self._call_api(prompt)
        summary = self._extract_summary(response)
        return summary
    
    def generate_batch(
        self,
        source_codes: List[str],
        example_codes: List[str],
        example_summaries: List[str],
        show_progress: bool = True,
        max_workers: int = 10,
        request_delay: float = 0.0
    ) -> List[str]:
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        total = len(source_codes)
        summaries = [""] * total
        
        def generate_single(idx, source_code, example_code, example_summary):
            
            try:
                if request_delay > 0:
                    time.sleep(request_delay)
                
                summary = self.generate(source_code, example_code, example_summary)
                return idx, summary, None
            except Exception as e:
                return idx, "", str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_single, i, src, ex_code, ex_sum): i
                for i, (src, ex_code, ex_sum) in enumerate(
                    zip(source_codes, example_codes, example_summaries)
                )
            }
            
            completed = 0
            for future in as_completed(futures):
                idx, summary, error = future.result()
                summaries[idx] = summary
                completed += 1
                
                if error:
                    print(f"  ❌ Failed to generate summary #{idx + 1}: {error}")
                
                if show_progress and completed % 10 == 0:
                    print(f"  Generation progress: {completed}/{total}")
        
        if show_progress and total % 10 != 0:
            print(f"  Generation progress: {total}/{total}")
        
        return summaries

