import json
import time
import requests
from typing import Dict, List, Tuple, Optional





PROMPT = """You are an expert code documentation evaluator. Your task is to assess whether the given summary accurately and completely describes the provided code.

**Code:**
```
{query_code}
```

**Summary:**
{candidate_summary}

**Evaluation Criteria:**
1. **Semantic Consistency**: Does the summary accurately describe the functionality and behavior of the code?
2. **Information Completeness**: Does the summary cover the key aspects of the code (purpose, main operations, important details)?

**Instructions:**
- If the summary is semantically consistent AND informationally complete, respond with: Y
- If there are any inconsistencies, missing critical information, or inaccuracies, respond with: N
- Only output Y or N, nothing else.
"""




RECOMMENDED_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 5,
    "description": "YN，"
}


class LLMPostProcessor:

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 5,
        max_workers: int = 5,
        retry_attempts: int = 3
    ):
        """
        LLM

        Args:
            api_base_url: APIURL
            api_key: API
            model_name:
            temperature:
            max_tokens: token
            max_workers:
            retry_attempts:
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts

        self.prompt_template = PROMPT

    def call_llm_api(self, prompt: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    return content
                else:
                    print(f"      ⚠️  API error (status {response.status_code}): {response.text}")

            except Exception as e:
                print(f"      ⚠️  API call exception (attempt {attempt+1}/{self.retry_attempts}): {e}")

            if attempt < self.retry_attempts - 1:
                time.sleep(1)

        return None

    def parse_decision(self, response: Optional[str]) -> bool:
        if response is None:
            return False


        response = response.strip().upper()


        if response and response[0] in ['Y', 'N']:
            return response[0] == 'Y'


        if 'YES' in response or 'Y' in response:
            return True
        if 'NO' in response or 'N' in response:
            return False


        print(f"      ⚠️  Cannot parse LLM response: '{response}', defaulting to no replacement")
        return False

    def evaluate_single(
        self,
        query_code: str,
        candidate_summary: str
    ) -> Tuple[bool, Optional[str]]:
        """


        Args:
            query_code:
            candidate_summary: （）

        Returns:
            (, LLM)
        """

        prompt = self.prompt_template.format(
            query_code=query_code,
            candidate_summary=candidate_summary
        )


        response = self.call_llm_api(prompt)


        should_replace = self.parse_decision(response)

        return should_replace, response

    def evaluate_batch(
        self,
        query_codes: List[str],
        candidate_summaries: List[str],
        verbose: bool = True
    ) -> List[Dict]:
        """


        Args:
            query_codes:
            candidate_summaries:
            verbose:

        Returns:
            ， {index, should_replace, llm_response}
        """
        results = []
        total = len(query_codes)

        if verbose:
            print(f"\n🤖 Start LLM post-processing evaluation (total: {total} samples)...")

        for i, (code, summary) in enumerate(zip(query_codes, candidate_summaries)):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total}")

            should_replace, llm_response = self.evaluate_single(code, summary)

            results.append({
                'index': i,
                'should_replace': should_replace,
                'llm_response': llm_response,
                'query_code': code,
                'candidate_summary': summary
            })


            time.sleep(0.1)

        if verbose:
            replace_count = sum(1 for r in results if r['should_replace'])
            print(f"  ✅ Done: {replace_count}/{total} samples marked for replacement")

        return results


def apply_llm_postprocessing(
    detailed_results: List[Dict],
    generated_summaries: List[str],
    processor: LLMPostProcessor,
    verbose: bool = True
) -> Tuple[List[str], int, List[int]]:
    """
    LLM

    Args:
        detailed_results:
        generated_summaries:
        processor: LLM
        verbose:

    Returns:
        (, , )
    """

    query_codes = []
    candidate_summaries = []
    valid_indices = []

    for i, result in enumerate(detailed_results):
        if generated_summaries[i] is None or generated_summaries[i] == "":
            continue

        query_code = result['query_code']
        selected_idx = result.get('selected_example_idx', 0)
        candidate_summary = result['top_n_summaries'][selected_idx]

        query_codes.append(query_code)
        candidate_summaries.append(candidate_summary)
        valid_indices.append(i)


    evaluation_results = processor.evaluate_batch(
        query_codes,
        candidate_summaries,
        verbose=verbose
    )


    updated_summaries = generated_summaries.copy()
    replaced_count = 0
    replaced_indices = []

    for eval_result in evaluation_results:
        eval_idx = eval_result['index']
        original_idx = valid_indices[eval_idx]

        if eval_result['should_replace']:

            candidate_summary = eval_result['candidate_summary']
            updated_summaries[original_idx] = candidate_summary


            detailed_results[original_idx]['post_processed'] = True
            detailed_results[original_idx]['llm_decision'] = 'Y'
            detailed_results[original_idx]['llm_response'] = eval_result['llm_response']
            detailed_results[original_idx]['replacement_reason'] = "LLM"

            replaced_count += 1
            replaced_indices.append(original_idx)

            if verbose:
                print(f"  ✅ Sample {original_idx}: replacement approved by LLM")
        else:

            detailed_results[original_idx]['post_processed'] = False
            detailed_results[original_idx]['llm_decision'] = 'N'
            detailed_results[original_idx]['llm_response'] = eval_result['llm_response']

    return updated_summaries, replaced_count, replaced_indices


if __name__ == "__main__":

    print("=" * 80)
    print("LLM post-processing module test")
    print("=" * 80)


    test_config = {
        "api_base_url": "https://api.openai.com",
        "api_key": "your-api-key",
        "model_name": "gpt-4o-mini"
    }

    print("\nConfiguration:")
    print(f"  API URL: {test_config['api_base_url']}")
    print(f"  Model: {test_config['model_name']}")
    print(f"  Temperature: {RECOMMENDED_CONFIG['temperature']}")
    print(f"  Max Tokens: {RECOMMENDED_CONFIG['max_tokens']}")

    print("\nUsage:")
    print("  1. Enable ENABLE_POST_PROCESSING = True in run_contrastive_retrieval.py")
    print("  2. Set POST_PROCESSING_MODE = 'llm'")
    print("  3. Configure LLM API parameters")
    print("  4. Run the generation pipeline")

    print("\n" + "=" * 80)




