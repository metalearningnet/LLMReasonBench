import re
import time
import json
import random
import openai
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import List, Dict, Any, Optional, Tuple, Set
from config import (
    REASON_TOKEN, CHOICE_MAP, DEFAULT_DATA_DIR, LLM_API_BASE, LLM_API_MODEL,
    STEPS, COT_TOKEN_NAMES, MEMORY_TOKEN, MEMORY_TOKEN_NAME,
    load_config, load_datasets_config, logger, dataset_names
)

@dataclass
class GeneratorArguments:
    dataset: str = field(
        default="truthfulqa",
        metadata={
            "choices": dataset_names,
            "help": "Dataset to generate CoT data for"
        }
    )
    mode: str = field(
        default="train",
        metadata={
            "choices": ["train", "test"],
            "help": "Whether to generate train or test data"
        }
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key for LLM service"}
    )
    api_base: Optional[str] = field(
        default=None,
        metadata={"help": "API endpoint URL"}
    )
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Model name. For OpenAI: 'gpt-4o', etc. For vLLM: HuggingFace model path"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for generated datasets"}
    )
    max_retries: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum retries for API calls"}
    )
    num_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of examples to generate"}
    )
    validate: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to validate CoT steps"}
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={"help": "Test configuration without making API calls"}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for generation"}
    )
    backend: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["openai", "vllm"],
            "help": "Backend to use for LLM generation. 'openai' for API calls, 'vllm' for local inference. If not specified, uses config value."
        }
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for vLLM generation (only used with backend='vllm')"}
    )
    gpu_memory_utilization: Optional[float] = field(
        default=None,
        metadata={"help": "GPU memory utilization for vLLM (0.0-1.0, only used with backend='vllm')"}
    )
    tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Tensor parallelism size for vLLM (number of GPUs, only used with backend='vllm')"}
    )
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config file"}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable debug logging for troubleshooting"}
    )

class BaseLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        generator_config = config.get('generator', {})
        llm_config = generator_config.get('llm_api', {})
        
        self.frequency_penalty = generator_config.get('frequency_penalty', 0.0)
        self.presence_penalty = generator_config.get('presence_penalty', 0.0)
        self.stop_sequences = generator_config.get('stop_sequences', [])
        self.temperature = generator_config.get('temperature', 0.3)
        self.max_tokens = generator_config.get('max_tokens', 1024)
        self.max_retries = generator_config.get('max_retries', 20)
        self.batch_size = generator_config.get('batch_size', 1)
        self.top_p = generator_config.get('top_p', 1.0)
        
        self.retry_max_wait = llm_config.get('retry_max_wait', 60)
        self.retry_min_wait = llm_config.get('retry_min_wait', 1)

        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
    
    def get_response(self, prompt: str, max_tokens: int = None) -> str:
        raise NotImplementedError
    
    def get_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        raise NotImplementedError
    
    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_output_tokens": self.total_output_tokens
        }
    
    def print_summary(self):
        summary = self.get_cost_summary()
        logger.info(f"Summary:")
        logger.info(f"  Total Requests: {summary['total_requests']}")
        logger.info(f"  Output Tokens: {summary['total_output_tokens']:,}")

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        llm_config = config.get('generator', {}).get('llm_api', {})
        
        self.api_key = llm_config.get('api_key', '')
        self.api_base = llm_config.get('api_base', LLM_API_BASE)
        self.model = llm_config.get('model', LLM_API_MODEL)
        self.timeout = llm_config.get('timeout', 60)
        
        self._configure_client()
        
        if not config.get('dry_run', False):
            self._test_connection()
    
    def _configure_client(self):
        if not self.api_key:
            logger.warning("No API key provided. Some LLM services may not require an API key.")
        
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": 0
        }
        
        if self.api_base != LLM_API_BASE:
            client_kwargs["base_url"] = self.api_base
        
        self.client = openai.OpenAI(**client_kwargs)
    
    def _test_connection(self):
        test_messages = [{"role": "user", "content": "Hello, are you working?"}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                max_tokens=5
            )
            
            if response:
                logger.info(f"✓ Connected to OpenAI API at {self.api_base}")
                logger.info(f"✓ Using model: {self.model}")
                return True
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            logger.info("Please check your configuration:")
            logger.info(f"  - API Base: {self.api_base}")
            logger.info(f"  - API Key: {'Set' if self.api_key else 'Not set'}")
            logger.info(f"  - Model: {self.model}")
            return False
    
    def _make_request(self, messages: List[Dict], max_tokens: int = None) -> Optional[Dict]:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        
        try:
            response = self.client.chat.completions.create(**params)
            self.total_requests += 1
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                output_tokens = usage.completion_tokens
                self.total_output_tokens += output_tokens
            
            return response
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_response(self, prompt: str, max_tokens: int = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(self.max_retries):
            try:
                response = self._make_request(messages, max_tokens)
                if response and response.choices:
                    return response.choices[0].message.content
                else:
                    logger.warning(f"Empty response, attempt {attempt + 1}/{self.max_retries}")
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                wait_time = min(self.retry_min_wait * (2 ** attempt), self.retry_max_wait)
                wait_time *= (0.5 + random.random())
                
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {e}")
                raise
        
        raise Exception(f"Failed to get response after {self.max_retries} attempts")
    
    def get_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        responses = []
        for prompt in tqdm(prompts, desc="Generating with OpenAI API"):
            try:
                response = self.get_response(prompt, max_tokens)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                responses.append("")
        return responses

class VLLMClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vllm_config = config.get('generator', {}).get('vllm', {})
        
        self.model_path = vllm_config.get('model')
        self.gpu_memory_utilization = vllm_config.get('gpu_memory_utilization', 0.9)
        self.max_model_len = vllm_config.get('max_model_len', 1024)
        self.tensor_parallel_size = vllm_config.get('tensor_parallel_size', 1)
        self.use_chat_template = vllm_config.get('use_chat_template', True)
        self.system_message = vllm_config.get('system_message', "You are a helpful assistant.")
        self.seed = vllm_config.get('seed', 42)
        
        if not self.model_path:
            raise ValueError("vLLM requires model_path to be specified in config")
        
        self._initialize_vllm()
    
    def _initialize_vllm(self):
        try:
            logger.info(f"Initializing vLLM with model: {self.model_path}")
            
            self.llm = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True
            )
            
            self.tokenizer = self.llm.get_tokenizer()
            
            logger.info(f"✓ vLLM initialized successfully with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
                return prompt
        return prompt
    
    def get_response(self, prompt: str, max_tokens: int = None) -> str:
        return self.get_responses([prompt], max_tokens)[0]
    
    def get_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        try:
            formatted_prompts = [self._format_prompt(p) for p in prompts]
            
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens,
                stop=self.stop_sequences,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed
            )
            
            all_responses = []
            for i in range(0, len(formatted_prompts), self.batch_size):
                batch_prompts = formatted_prompts[i:i + self.batch_size]
                
                outputs = self.llm.generate(batch_prompts, sampling_params)
                
                for output in outputs:
                    generated_text = output.outputs[0].text.strip()
                    all_responses.append(generated_text)
                    completion_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
                    
                    self.total_output_tokens += completion_tokens
                    self.total_requests += 1
                
                logger.debug(f"Generated batch {i//self.batch_size + 1}/{(len(formatted_prompts)-1)//self.batch_size + 1}")
            
            return all_responses
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise

class LLMClientFactory:
    @staticmethod
    def create_client(config: Dict[str, Any], backend: str = None) -> BaseLLMClient:
        backend = backend or config.get('generator', {}).get('backend', 'openai')
        
        if backend == 'openai':
            return OpenAIClient(config)
        elif backend == 'vllm':
            return VLLMClient(config)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'openai' or 'vllm'.")

class CoTGenerator:
    def __init__(self, config: Dict[str, Any], backend: str = None):
        self.config = config
        generator_config = config.get('generator', {})
        
        self.backend = backend or generator_config.get('backend', 'openai')
        self.llm_client = LLMClientFactory.create_client(config, self.backend)
        self.should_validate_cot_steps = generator_config.get('validate_cot_steps', True)
        self.require_memory_before_reason = generator_config.get('require_memory_before_reason', False)
        self.min_memory_steps = generator_config.get('min_memory_steps', 1)
        self.min_reason_steps = generator_config.get('min_reason_steps', 1)
        self.max_validation_attempts = generator_config.get('max_cot_validation_attempts', 3)
        self.max_generation_attempts = generator_config.get('max_generation_attempts', 3)
        self.min_cot_steps = generator_config.get('min_cot_steps', 2)
        self.max_cot_steps = generator_config.get('max_cot_steps', 30)
        self.require_both_memory_reason = generator_config.get('require_both_memory_reason', False)
        self.log_raw_responses = config.get('log_raw_responses', False)
        self.dataset_configs = config.get('datasets', {})
        self.debug = config.get('debug', False)
        
        logger.info(f"Initialized CoTGenerator with backend: {self.backend}")
        if self.debug:
            logger.info(f"Debug mode enabled")
            logger.info(f"Max generation attempts: {self.max_generation_attempts}")
            logger.info(f"Min CoT steps: {self.min_cot_steps}, Max CoT steps: {self.max_cot_steps}")
    
    def format_example(self, question: str, options: List[str] = None, cot_content: str = "") -> str:
        example = f"Question: {question}"
        if options:
            example += "\nOptions: "
            for i, opt in enumerate(options):
                example += f"{CHOICE_MAP[i]}. {opt}\n"
        
        if cot_content:
            example += f"Answer: {cot_content}\n\n"
        else:
            example += "\nAnswer: "
        
        return example

    def planning_prompt(self, question: str, answer: str) -> str:
        token_descriptions = [
            f"<{name}>: {STEPS[name]['description']}"
            for name in COT_TOKEN_NAMES
        ]
        
        guidelines_sections = []
        for name in COT_TOKEN_NAMES:
            guidelines = STEPS[name]['guidelines']
            guidelines_sections.append(f"<{name}>:")
            guidelines_sections.extend(f"- {guideline}" for guideline in guidelines)
            guidelines_sections.append("")
        
        if guidelines_sections and guidelines_sections[-1] == "":
            guidelines_sections.pop()
        
        token_types_str = "\n".join(token_descriptions)
        guidelines_str = "\n".join(guidelines_sections)
        
        return f"""Generate a step-by-step reasoning plan to solve the following question:

{question}

TOKEN TYPES:
{token_types_str}

CRITICAL RULES:
- Each step must begin with a token followed by ': '
- No step numbers
- Never write tokens without angle brackets (e.g., use "{MEMORY_TOKEN}:" not "{MEMORY_TOKEN_NAME}:")
- Never add explanatory text after the token (e.g., use "{MEMORY_TOKEN}:" not "{MEMORY_TOKEN} step:")
- Never prefix tokens with "Step:" (e.g., use "{MEMORY_TOKEN}:" not "Step: {MEMORY_TOKEN}:")
- Each step should contain only one type of content
- Do not include any commentary about the task. Only write the steps in the specified format.

GUIDELINES BY TOKEN TYPE:
{guidelines_str}

Now, generate the CoT steps for the question above:

Begin immediately with the first token:
"""
    
    def ner_agent_prompt(self, questions: List[str]) -> str:
        """
        Generate the NER (Named Entity Recognition) agent prompt for factual knowledge.
        
        Args:
            questions: List of knowledge queries
            
        Returns:
            Formatted NER prompt
        """
        questions_text = "\n".join(f"- {q}" for q in questions)
        return f"""You are a factual knowledge provider. For each query below, provide the factual knowledge that can be verified through evidence or observation.

Queries:
{questions_text}

Provide your response as a JSON dictionary where:
- Keys are the exact query strings from above
- Values are concise factual knowledge statements

Format your response EXACTLY as:
```json
{{
  "query 1": "factual knowledge for query 1",
  "query 2": "factual knowledge for query 2",
  ...
}}
```

Important:
- Only include the JSON in your response.
- Do not add any additional text or explanation.
- Ensure the JSON is properly formatted.
"""
    
    def extract_knowledge_based(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        patterns = [
            r"\*\*Knowledge based\*\*:\s*(.*?)(?=\*\*Content\*\*:|$)",
            r'Knowledge based:\s*(.*?)(?=Content:|$)',
            r'Knowledge query:\s*(.*?)(?=Content:|$)',
            r'Query:\s*(.*?)(?=Content:|$)'
        ]
        
        all_queries = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                queries = []
                for match in matches:
                    clean_match = match.strip()
                    clean_match = re.sub(r'\*\*Content\*\*.*$', '', clean_match, flags=re.DOTALL | re.IGNORECASE)
                    clean_match = clean_match.strip()
                    if clean_match:
                        queries.append(clean_match)
                
                if queries:
                    all_queries.extend(queries)
                    if self.debug:
                        logger.debug(f"Found {len(queries)} queries with pattern: {pattern[:30]}...")
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in all_queries:
            if query not in seen and len(query) > 5:
                seen.add(query)
                unique_queries.append(query)
        
        if self.debug and unique_queries:
            logger.debug(f"Extracted {len(unique_queries)} unique knowledge queries")
        
        return unique_queries
    
    def extract_labeled_content(self, input_string: str) -> List[str]:
        if not input_string or not input_string.strip():
            return []
        
        input_string = input_string.strip()
        
        token_names_pattern = '|'.join(re.escape(name) for name in COT_TOKEN_NAMES)
        pattern = rf'<({token_names_pattern})>:\s*([^<]+?)(?=\s*<({token_names_pattern})>|$)'
        
        steps = []
        for match in re.finditer(pattern, input_string, re.IGNORECASE | re.DOTALL):
            token_name = match.group(1).lower()
            content = match.group(2).strip()
            
            if not content:
                continue
            
            content = re.sub(r'\s+', ' ', content).strip()
            
            if self._is_garbage_content(content):
                continue
            
            if content and content[0].islower():
                content = content[0].upper() + content[1:]
            
            if content and not any(content.endswith(p) for p in ['.', '!', '?', ':']):
                if len(content.split()) > 2:
                    content = content + '.'
            
            steps.append(f"<{token_name}>: {content}")
        
        if not steps:
            pattern2 = rf'<({token_names_pattern})>:\s*([^\n]+)'
            for match in re.finditer(pattern2, input_string, re.IGNORECASE):
                token_name = match.group(1).lower()
                content = match.group(2).strip()
                
                if not content:
                    continue
                
                content = re.sub(r'\s+', ' ', content).strip()
                
                if self._is_garbage_content(content):
                    continue
                
                if content and content[0].islower():
                    content = content[0].upper() + content[1:]
                
                if content and not any(content.endswith(p) for p in ['.', '!', '?', ':']):
                    if len(content.split()) > 2:
                        content = content + '.'
                
                steps.append(f"<{token_name}>: {content}")
        
        seen = set()
        unique_steps = []
        for step in steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)
        
        return unique_steps
    
    def clean_json_string(self, json_str: str) -> Dict:
        if not json_str or not json_str.strip():
            return {}
        
        json_str = json_str.strip()
        
        if self.debug:
            logger.debug(f"Cleaning JSON string (first 200 chars): {json_str[:200]}")
        
        # Remove code blocks
        json_str = re.sub(r'```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```', '', json_str)
        
        # Remove non-JSON content before and after
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        
        # Fix common JSON issues
        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
        
        # Remove empty key-value pairs
        json_str = re.sub(r'"[^"]*"\s*:\s*""\s*,?', '', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Remove extra whitespace
        json_str = re.sub(r'\s+', ' ', json_str)
        
        # Validate braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        if open_braces != close_braces:
            if self.debug:
                logger.debug(f"Unbalanced braces: {open_braces} open, {close_braces} close")
            # Try to fix by adding missing braces
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            else:
                json_str = '{' * (close_braces - open_braces) + json_str
        
        try:
            result = json.loads(json_str)
            if self.debug:
                logger.debug(f"Successfully parsed JSON with {len(result)} keys")
            return result
        except json.JSONDecodeError as e:
            if self.debug:
                logger.debug(f"JSON decode error: {e}")
                logger.debug(f"Problematic JSON string: {json_str[:500]}")
            
            try:
                # Try to parse as single string and convert to dict
                if json_str.startswith('"') and json_str.endswith('"'):
                    json_str = json_str[1:-1]
                    # Try to parse the inner string
                    inner_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                    if inner_match:
                        return json.loads(inner_match.group(0))
                
                # Try to extract key-value pairs manually
                key_value_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', json_str)
                if key_value_pairs:
                    result = {}
                    for key, value in key_value_pairs:
                        result[key] = value
                    logger.warning(f"Recovered JSON with {len(result)} key-value pairs after error")
                    return result
            except Exception as e2:
                logger.debug(f"Recovery attempt failed: {e2}")
            
            return {}
    
    def validate_cot_steps(self, cot_steps: List[str]) -> bool:
        if not cot_steps:
            if self.debug:
                logger.debug("Validation failed: No CoT steps generated")
            return False
        
        if not self.should_validate_cot_steps:
            return True
        
        if len(cot_steps) > self.max_cot_steps:
            if self.debug:
                logger.debug(f"Validation failed: Extremely many steps ({len(cot_steps)} > {self.max_cot_steps})")
            return False
        
        if len(cot_steps) < self.min_cot_steps:
            if self.debug:
                logger.debug(f"Validation failed: Too few steps ({len(cot_steps)} < 2)")
            return False
        
        token_step_counts = {token_name: 0 for token_name in COT_TOKEN_NAMES}
        valid_tokens_pattern = '|'.join(re.escape(name) for name in COT_TOKEN_NAMES)
        
        content_set = set()
        unique_content_count = 0
        
        for i, step in enumerate(cot_steps):
            if ': ' not in step:
                if self.debug:
                    logger.debug(f"Validation failed: Step without ': ' separator at {i}: {step[:50]}...")
                return False
            
            tag_part, content = step.split(': ', 1)
            tag_part = tag_part.strip()
            content = content.strip()
            
            tag_match = re.search(rf'\<({valid_tokens_pattern})\>', tag_part, re.IGNORECASE)
            if not tag_match:
                if self.debug:
                    logger.debug(f"Validation failed: Step with invalid tag format at {i}: {tag_part[:50]}...")
                return False
            
            token_name = tag_match.group(1).lower()
            
            if not content:
                if self.debug:
                    logger.debug(f"Validation failed: Empty content for <{token_name}> at step {i}")
                return False
            
            if self._is_garbage_content(content):
                if self.debug:
                    logger.debug(f"Validation failed: Garbage/incomplete content at step {i}: {content[:50]}...")
                return False
            
            content_normalized = content.lower().strip()
            if content_normalized and content_normalized not in content_set:
                content_set.add(content_normalized)
                unique_content_count += 1
            
            token_step_counts[token_name] += 1
        
        if unique_content_count < 2:
            if self.debug:
                logger.debug(f"Validation failed: Not enough unique content ({unique_content_count} < 2)")
            return False
        
        total_steps = len(cot_steps)
        if total_steps > 3:
            max_single_type_ratio = 0.85
            
            for token_name, count in token_step_counts.items():
                if count / total_steps > max_single_type_ratio:
                    if self.debug:
                        logger.debug(f"Validation failed: Too many <{token_name}> steps ({count}/{total_steps} > {max_single_type_ratio*100}%)")
                    return False
        
        if self.debug:
            token_summary = ', '.join([f"{count} <{token}>" for token, count in token_step_counts.items() if count > 0])
            logger.debug(f"Validation passed: {len(cot_steps)} steps total ({token_summary}), {unique_content_count} unique content items")
        
        return True

    def _is_garbage_content(self, content: str) -> bool:
        if not content:
            return True
        
        content_lower = content.lower().strip()
        
        garbage_patterns = [
            r'^\s*s\s*[:.]',
            r'^\s*ing\.',
            r'^\s*steps?\.',
            r'^\s*wait,',
            r'^\s*but\s+',
            r'^\s*so\s+',
            r'^\s*well,',
            r'^\s*</?think>',
            r'^\s*\*+\s*$',
            r'^\s*\.\.+',
            r'^\s*$',
            r'^\s*then\s+the\s*$',
            r'^\s*and\s+then\s*$',
            r'^\s*extract\s+',
            r'^\s*state\s+',
            r'^\s*list\s+',
            r'^\s*first\s+',
            r'^\s*next\s+',
            r'^\s*i\s+',
            r'^\s*we\s+'
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Very short content that's not meaningful
        if len(content_lower) < 4:
            return True
        
        # Content that's just a single word or two without context
        words = content_lower.split()
        if len(words) <= 2:
            # Allow short facts
            if not any(word.isdigit() for word in words):
                # If no numbers, probably incomplete
                return True
        
        # Check for cut-off endings
        cutoff_endings = [' a', ' an', ' the', ' to', ' of', ' for', ' with', ' without', ' and', ' or', ' but', ' then', ' so']
        for ending in cutoff_endings:
            if content_lower.endswith(ending):
                return True
        
        return False

    def get_cot_steps(self, question: str, answer: str) -> List[str]:
        if self.debug:
            logger.debug(f"Generating CoT for question: {question[:100]}...")
        
        planning_prompt = self.planning_prompt(question, answer)
        
        if self.log_raw_responses:
            logger.info(f"Planning prompt: {planning_prompt}")
        
        try:
            plan_response = self.llm_client.get_response(planning_prompt)
            
            if not plan_response or plan_response.strip() == "":
                if self.debug:
                    logger.debug(f"Empty plan response for question: {question[:50]}...")
                return []
            
            if self.log_raw_responses:
                logger.info(f"Plan response (first 1000 chars): {plan_response[:1000]}...")
            
            if self.debug:
                logger.debug(f"Plan response length: {len(plan_response)} characters")
        except Exception as e:
            logger.error(f"Error getting planning response: {e}")
            return []
        
        # Extract knowledge-based queries
        kb_queries = self.extract_knowledge_based(plan_response)
        
        if kb_queries:
            if self.debug:
                logger.debug(f"Found {len(kb_queries)} knowledge queries")
            
            # Get factual knowledge
            try:
                ner_prompt = self.ner_agent_prompt(kb_queries)
                ner_response = self.llm_client.get_response(ner_prompt)
                
                if self.log_raw_responses:
                    logger.info(f"NER response (first 1000 chars): {ner_response[:1000] if ner_response else 'EMPTY'}")
                
                try:
                    kb_dict = self.clean_json_string(ner_response)
                    if kb_dict:
                        if self.debug:
                            logger.debug(f"Successfully parsed {len(kb_dict)} knowledge entries")
                        
                        # Replace queries with knowledge in planning
                        for query, knowledge in kb_dict.items():
                            if query in plan_response:
                                plan_response = plan_response.replace(query, f"{query} {MEMORY_TOKEN}{knowledge}")
                                if self.debug:
                                    logger.debug(f"Replaced query '{query[:50]}...' with knowledge")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse NER response: {e}")
            except Exception as e:
                logger.error(f"Error getting NER response: {e}")
        
        cot_steps = self.extract_labeled_content(plan_response)
        
        if self.debug:
            logger.debug(f"Extracted {len(cot_steps)} CoT steps")
        
        if not cot_steps:
            if self.debug:
                logger.debug("No CoT steps extracted")
            return []
        
        if not self.validate_cot_steps(cot_steps):
            if self.debug:
                logger.debug("CoT steps failed validation")
            
            if not self.should_validate_cot_steps:
                return cot_steps
            else:
                return []
        
        return cot_steps
    
    def get_cot_steps_with_retry(self, question: str, answer: str, max_attempts: int = None) -> List[str]:
        max_attempts = max_attempts or self.max_generation_attempts
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    if self.debug:
                        logger.debug(f"Retry attempt {attempt + 1}/{max_attempts} for question: {question[:50]}...")
                
                cot_steps = self.get_cot_steps(question, answer)
                
                if cot_steps:
                    if self.debug:
                        logger.debug(f"Successfully generated {len(cot_steps)} CoT steps on attempt {attempt + 1}")
                    return cot_steps
                else:
                    if self.debug:
                        logger.debug(f"Empty CoT steps on attempt {attempt + 1}/{max_attempts}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to generate CoT after {max_attempts} attempts")
                    return []
        
        return []
    
    def batch_generate_cot_steps(self, questions: List[str], answers: List[str]) -> List[List[str]]:
        if len(questions) != len(answers):
            raise ValueError(f"Number of questions ({len(questions)}) doesn't match number of answers ({len(answers)})")
        
        all_planning_prompts = []
        question_indices = []
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            planning_prompt = self.planning_prompt(question, answer)
            all_planning_prompts.append(planning_prompt)
            question_indices.append(i)
        
        logger.info(f"Batch generating planning responses for {len(all_planning_prompts)} questions")
        
        try:
            plan_responses = self.llm_client.get_responses(all_planning_prompts)
            
            if self.debug:
                successful = sum(1 for r in plan_responses if r and r.strip())
                logger.debug(f"Successfully generated {successful}/{len(plan_responses)} planning responses")
                for i, (prompt, response) in enumerate(zip(all_planning_prompts[:3], plan_responses[:3])):
                    logger.debug(f"Example {i} - Response preview: {response[:500] if response else 'EMPTY'}...")
        except Exception as e:
            logger.error(f"Failed to generate planning responses: {e}")
            return [[] for _ in questions]
        
        # Extract knowledge-based queries and prepare NER prompts
        kb_queries_batch = []
        query_to_question_map = []
        plan_responses_with_kb = plan_responses.copy()
        
        for i, (plan_response, question_idx) in enumerate(zip(plan_responses, question_indices)):
            if not plan_response or plan_response.strip() == "":
                if self.debug and i < 5:
                    logger.debug(f"Empty plan response for question index {question_idx}")
                continue
            
            kb_queries = self.extract_knowledge_based(plan_response)
            if kb_queries:
                for query in kb_queries:
                    kb_queries_batch.append(query)
                    query_to_question_map.append((question_idx, query, i))
        
        if kb_queries_batch:
            logger.info(f"Batch generating NER responses for {len(kb_queries_batch)} knowledge queries")
            try:
                ner_prompt = self.ner_agent_prompt(kb_queries_batch)
                ner_response = self.llm_client.get_response(ner_prompt)
                
                if self.debug:
                    logger.debug(f"NER response length: {len(ner_response) if ner_response else 0} characters")
                
                try:
                    kb_dict = self.clean_json_string(ner_response)
                    update_count = 0
                    for question_idx, query, plan_idx in query_to_question_map:
                        if query in kb_dict:
                            knowledge = kb_dict[query]
                            if query in plan_responses_with_kb[plan_idx]:
                                plan_responses_with_kb[plan_idx] = plan_responses_with_kb[plan_idx].replace(
                                    query, f"{query} {MEMORY_TOKEN}{knowledge}"
                                )
                                update_count += 1
                    
                    if self.debug:
                        logger.debug(f"Updated {update_count} planning responses with knowledge")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse NER response: {e}")
                    if self.debug:
                        logger.debug(f"Raw NER response (first 500 chars): {ner_response[:500] if ner_response else 'EMPTY'}")
            except Exception as e:
                logger.error(f"Failed to generate NER responses: {e}")
        
        all_cot_steps = []
        validation_stats = {"valid": 0, "invalid": 0, "empty": 0}
        
        for i, plan_response in enumerate(plan_responses_with_kb):
            if not plan_response or plan_response.strip() == "":
                all_cot_steps.append([])
                validation_stats["empty"] += 1
                if self.debug and i < 5:
                    logger.debug(f"Example {i}: Empty response")
                continue
                
            cot_steps = self.extract_labeled_content(plan_response)
            
            if self.debug and i < 3 and cot_steps:
                logger.debug(f"Example {i} - Extracted {len(cot_steps)} steps")
            
            if not cot_steps:
                if self.debug and i < 5:
                    logger.debug(f"Example {i}: No steps extracted from response")
                all_cot_steps.append([])
                validation_stats["empty"] += 1
                continue
            
            if len(cot_steps) < self.min_cot_steps:
                if self.debug and i < 5:
                    logger.debug(f"Example {i}: Too few steps ({len(cot_steps)} < {self.min_cot_steps})")
                all_cot_steps.append([])
                validation_stats["invalid"] += 1
                continue
            
            if self.should_validate_cot_steps:
                is_valid = self.validate_cot_steps(cot_steps)
                if is_valid:
                    all_cot_steps.append(cot_steps)
                    validation_stats["valid"] += 1
                else:
                    if self.debug and i < 5:
                        logger.debug(f"Example {i}: Steps failed validation")
                    
                    all_cot_steps.append(cot_steps)
                    validation_stats["invalid"] += 1
            else:
                all_cot_steps.append(cot_steps)
                validation_stats["valid"] += 1
        
        logger.info(f"CoT generation stats: {validation_stats['valid']} valid, {validation_stats['invalid']} invalid, {validation_stats['empty']} empty")
        
        return all_cot_steps
    
    def is_high_quality_cot(self, cot_steps: List[str]) -> bool:
        if not cot_steps:
            return False
        
        if len(cot_steps) < self.min_cot_steps:
            return False
        
        total_content_length = 0
        has_memory = False
        has_reason = False
        
        for step in cot_steps:
            if ': ' in step:
                tag_part, content = step.split(': ', 1)
                total_content_length += len(content.strip())
                
                if MEMORY_TOKEN in tag_part.lower():
                    has_memory = True
                elif REASON_TOKEN in tag_part.lower():
                    has_reason = True
        
        avg_content_length = total_content_length / len(cot_steps)
        if avg_content_length < 20:
            return False
        
        if self.require_both_memory_reason and not (has_memory and has_reason):
            return False
        
        return True
    
    def generate_with_quality_check(self, question: str, answer: str, max_attempts: int = None) -> List[str]:
        max_attempts = max_attempts or self.max_generation_attempts
        
        for attempt in range(max_attempts):
            cot_steps = self.get_cot_steps_with_retry(question, answer, max_attempts=1)
            
            if self.is_high_quality_cot(cot_steps):
                if self.debug:
                    logger.debug(f"Generated high-quality CoT with {len(cot_steps)} steps on attempt {attempt + 1}")
                return cot_steps
            
            if attempt < max_attempts - 1:
                if self.debug:
                    logger.debug(f"CoT quality insufficient, retrying... (attempt {attempt + 1}/{max_attempts})")
        
        if self.debug:
            logger.debug(f"Failed to generate high-quality CoT after {max_attempts} attempts")
        return []

class DatasetGenerator:
    ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"
    ANSWER_TYPE_BOOLEAN = "boolean"
    ANSWER_TYPE_NUMERIC = "numeric"
    ANSWER_TYPE_OPEN_ENDED = "open_ended"
    
    def __init__(self, cot_generator, config: Dict[str, Any], dataset_name: str):
        datasets_config = load_datasets_config()

        self.config = config
        self.dataset_name = dataset_name
        self.cot_generator = cot_generator
        self.generator_config = config['generator']
        self.dataset_config = datasets_config[dataset_name]
        
        self.output_dir = Path(self.generator_config.get('output_dir', DEFAULT_DATA_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.incremental_save = self.generator_config.get('incremental_save', True)
        self.filter_output = self.generator_config.get('filter_output', True)
        self.incremental_save_interval = self.generator_config.get('incremental_save_interval', 1000)
        
        self.max_retries = self.cot_generator.llm_client.max_retries
        self.batch_size = self.generator_config.get('batch_size', self.cot_generator.llm_client.batch_size)
        
        self._initialize_configuration()
        
        logger.info(f"Dataset {dataset_name}: answer_type={self._answer_type}")
        logger.info(f"Valid answers: {self._valid_answers}")
        logger.info(f"Field mapping: {self._field_mapping}")
        logger.info(f"LaTeX cleaning: {'Enabled' if self._clean_latex_enabled else 'Disabled'}")
    
    def _initialize_configuration(self):
        self._answer_type = self.get_answer_type()
        self._valid_answers = self.get_valid_answers()
        self._choice_labels = self.get_choice_labels()
        self._field_mapping = self.get_field_mapping()
        
        self._clean_latex_enabled = self.dataset_config.get('clean_latex', False)
        self._additional_config = self.dataset_config.get('additional_config', {})
    
    def get_answer_type(self) -> str:
        return self.dataset_config['answer_type']
    
    def get_valid_answers(self) -> Optional[Set[str]]:
        if self.get_answer_type() != self.ANSWER_TYPE_MULTIPLE_CHOICE:
            return None
        
        valid_answers = self.dataset_config.get('valid_answers')
        if valid_answers is None:
            return None
        
        if isinstance(valid_answers, list):
            return set(str(v).upper() for v in valid_answers)
        elif isinstance(valid_answers, str):
            if '-' in valid_answers:
                # Range like "A-D"
                start, end = valid_answers.split('-')
                if len(start) == 1 and len(end) == 1:
                    return set(chr(c) for c in range(ord(start.upper()), ord(end.upper()) + 1))
            return set(v.strip().upper() for v in valid_answers.split(','))
        
        return None
    
    def get_choice_labels(self) -> List[str]:
        choice_labels = self.dataset_config.get('choice_labels')
        if choice_labels:
            if isinstance(choice_labels, list):
                return [str(label).upper() for label in choice_labels]
            elif isinstance(choice_labels, str):
                return [label.strip().upper() for label in choice_labels.split(',')]
        
        valid_answers = self.get_valid_answers()
        if valid_answers:
            valid_sorted = sorted(valid_answers, key=lambda x: x.upper())
            return [label.upper() for label in valid_sorted]
        
        return ['A', 'B', 'C', 'D', 'E']
    
    def get_field_mapping(self) -> Dict[str, str]:
        config_mapping = self.dataset_config.get('field_mapping', {})
        
        default_mapping = {
            'question': 'question',
            'answer': 'answer',
            'options': 'options',
            'choices': 'choices'
        }
        
        merged = default_mapping.copy()
        merged.update(config_mapping)
        
        return merged
    
    def extract_answer_from_example(self, example: Dict) -> Optional[Any]:
        return None
    
    def parse_options(self, options: Any) -> Optional[List[str]]:
        return None
    
    def get_answer_keywords(self) -> List[str]:
        return ['answer', 'option', 'choice', 'correct', 'solution', 'result', 'final']
    
    def get_answer_prefixes(self) -> List[str]:
        return ['Answer:', 'Solution:', 'Response:', 'The answer is:', 'Result:']
    
    def _clean_latex(self, text: str) -> str:
        if not text:
            return text
        
        text = text.replace('\\(', '').replace('\\)', '')
        text = text.replace('\\[', '').replace('\\]', '')
        
        text = re.sub(r'\$(.*?)\$', r'\1', text)
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
        
        text = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'(\1)/(\2)', text, flags=re.DOTALL)
        text = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', text)
        text = re.sub(r'\\sqrt\[([^\]]+)\]{(.+?)}', r'(\2)^(1/\1)', text, flags=re.DOTALL)
        
        text = re.sub(r'\\text{(.+?)}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\\mathrm{(.+?)}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\\mathbf{(.+?)}', r'\1', text, flags=re.DOTALL)
        
        greek_map = {
            r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\Gamma': 'Γ',
            r'\\delta': 'δ', r'\\Delta': 'Δ', r'\\epsilon': 'ε', r'\\varepsilon': 'ε',
            r'\\zeta': 'ζ', r'\\eta': 'η', r'\\theta': 'θ', r'\\Theta': 'Θ',
            r'\\iota': 'ι', r'\\kappa': 'κ', r'\\lambda': 'λ', r'\\Lambda': 'Λ',
            r'\\mu': 'μ', r'\\nu': 'ν', r'\\xi': 'ξ', r'\\Xi': 'Ξ',
            r'\\pi': 'π', r'\\Pi': 'Π', r'\\rho': 'ρ', r'\\sigma': 'σ',
            r'\\Sigma': 'Σ', r'\\tau': 'τ', r'\\upsilon': 'υ', r'\\Upsilon': 'Υ',
            r'\\phi': 'φ', r'\\Phi': 'Φ', r'\\chi': 'χ', r'\\psi': 'ψ',
            r'\\Psi': 'Ψ', r'\\omega': 'ω', r'\\Omega': 'Ω'
        }
        
        for latex, unicode in greek_map.items():
            text = text.replace(latex, unicode)
        
        operator_map = {
            r'\\times': '×', r'\\cdot': '·', r'\\div': '÷',
            r'\\pm': '±', r'\\mp': '∓', r'\\leq': '≤', r'\\le': '≤',
            r'\\geq': '≥', r'\\ge': '≥', r'\\neq': '≠', r'\\ne': '≠',
            r'\\approx': '≈', r'\\sim': '∼', r'\\propto': '∝',
            r'\\infty': '∞', r'\\partial': '∂', r'\\nabla': '∇'
        }
        
        for latex, unicode in operator_map.items():
            text = re.sub(latex, unicode, text)
        
        text = re.sub(r'\^{([^}]+)}', r'^\1', text)
        text = re.sub(r'_{([^}]+)}', r'_\1', text)
        
        text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_dataset(self, split: str):
        source = self.dataset_config.get('source')
        if not source:
            logger.error(f"No source specified for dataset {self.dataset_name}")
            raise ValueError(f"Dataset source not configured for {self.dataset_name}")
        
        split_mapping = self.dataset_config.get('split_mapping', {})
        config_name = self.dataset_config.get('config_name')
        actual_split = split_mapping.get(split, split)
        
        try:
            from datasets import load_dataset
            logger.info(f"Loading dataset: {source}, split: {actual_split}")
            
            if config_name:
                dataset = load_dataset(source, config_name)
            else:
                dataset = load_dataset(source)
            
            if actual_split not in dataset:
                available_splits = list(dataset.keys())
                logger.error(f"Split '{actual_split}' not found. Available splits: {available_splits}")
                raise ValueError(f"Split '{actual_split}' not found in dataset")
            
            logger.info(f"Successfully loaded {len(dataset[actual_split])} examples from {source}/{actual_split}")
            return dataset[actual_split]
            
        except ImportError:
            logger.error("The 'datasets' library is not installed. Please install it with: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset {source} split {actual_split}: {e}")
            raise
    
    def get_output_filename(self, split: str) -> Path:
        if self.dataset_name:
            base_name = f"{self.dataset_name}_{split}"
        else:
            dataset_name = self.__class__.__name__.replace('Generator', '')
            base_name = f"{dataset_name}_{split}"
        
        base_name = base_name.lower()
        
        return self.output_dir / f"{base_name}.json"
    
    def save_results(self, results: List[Dict], output_file: Path):
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved {len(results)} examples to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise
    
    def filter_results(self, results: List[Dict], split: str) -> List[Dict]:
        pattern = re.compile(r"^\s*\<(.*?)\>\s*:")
        cleaned_data = []
        
        for item in results:
            if not item:
                continue
            
            cot_steps = item.get('cot_steps', [])
            
            if split == "test" and not cot_steps:
                cleaned_data.append(item)
                continue
            
            filtered_steps = [
                step for step in cot_steps
                if step and ': ' in step and step.split(': ')[1].strip()
            ]
            
            if not filtered_steps:
                # Skip examples with no valid CoT steps
                if split != "test":
                    continue
                else:
                    cleaned_data.append(item)
                    continue
            
            processed_steps = []
            for entry in filtered_steps:
                match = pattern.search(entry)
                if match:
                    processed_steps.append(entry)
                else:
                    processed_steps.append(entry)
            
            cleaned_data.append({
                "question": item.get('question', ''),
                "answer": item.get('answer', ''),
                "cot_steps": processed_steps,
                "split": item.get('split', split)
            })
        
        logger.info(f"Filtered {len(results)} -> {len(cleaned_data)} examples")
        return cleaned_data
    
    def format_question_and_answer(self, example: Dict) -> Tuple[str, str]:
        return self._format_question_and_answer_internal(example)
    
    def _format_question_and_answer_internal(self, example: Dict) -> Tuple[str, str]:
        try:
            raw_question = self._get_field(example, 'question')
            if not raw_question:
                logger.warning(f"No question found in example: {example.keys()}")
                raise ValueError("No question found")
            
            raw_answer = None
            
            if hasattr(self, 'extract_answer_from_example'):
                try:
                    raw_answer = self.extract_answer_from_example(example)
                except Exception as e:
                    logger.debug(f"Custom answer extraction failed: {e}")
            
            if raw_answer is None:
                raw_answer = self._get_field(example, 'answer')
            
            if raw_answer is None:
                logger.warning(f"No answer found in example: {example.keys()}")
                raise ValueError("No answer found")
            
            raw_options = self._get_field(example, 'options')
            if raw_options is None:
                raw_options = self._get_field(example, 'choices')
            
            question = self._format_question_with_options(raw_question, raw_options)
            
            if self._clean_latex_enabled:
                question = self._clean_latex(question)
            
            answer = self._normalize_answer_internal(raw_answer, raw_options)
            
            return question, answer
        except Exception as e:
            logger.warning(f"Error formatting example: {e}")
            raise
    
    def _get_field(self, example: Dict, field_name: str) -> Any:
        mapped_field = self._field_mapping.get(field_name)
        if mapped_field and mapped_field in example:
            return example[mapped_field]
        
        if field_name in example:
            return example[field_name]
        
        alternatives = self._get_field_alternatives(field_name)
        for alt in alternatives:
            if alt in example:
                return example[alt]
        
        return None
    
    def _get_field_alternatives(self, field_name: str) -> List[str]:
        if field_name == 'question':
            return ['query', 'problem', 'prompt', 'text', 'sentence', 'content', 'stem']
        elif field_name == 'answer':
            return ['correct', 'correct_answer', 'solution', 'response',
                   'label', 'target', 'gold_answer', 'answer_key']
        elif field_name in ['options', 'choices']:
            return ['choices', 'options', 'candidates', 'alternatives']
        return []
    
    def _format_question_with_options(self, question: str, options: Any) -> str:
        question = str(question).strip()
        
        question = self._clean_question(question)
        
        if not options:
            return question
        
        parsed_options = None
        if hasattr(self, 'parse_options'):
            parsed_options = self.parse_options(options)
        
        if parsed_options is None:
            parsed_options = self._parse_options_internal(options)
        
        if not parsed_options:
            return question
        
        formatted_options = self._format_options(parsed_options)
        
        return f"{question}\n\n{formatted_options}"
    
    def _clean_question(self, question: str) -> str:
        question = str(question).strip()
        
        prefixes = ['Question:', 'Problem:', 'Q:', 'Prompt:']
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
                break
        
        if self._clean_latex_enabled:
            question = self._clean_latex(question)
        
        if not question.endswith(('?', '!', '.')):
            if any(word in question.lower() for word in ['what', 'why', 'how', 'when', 'where', 'which']):
                question += '?'
            else:
                question += '.'
        
        return question
    
    def _parse_options_internal(self, options: Any) -> List[str]:
        if not options:
            return []
        
        if isinstance(options, list):
            return [str(opt).strip() for opt in options]
        
        if isinstance(options, str):
            options = options.strip()
            
            if options.startswith('[') and options.endswith(']'):
                try:
                    parsed = json.loads(options)
                    if isinstance(parsed, list):
                        return [str(opt).strip() for opt in parsed]
                except:
                    pass
            
            if ',' in options:
                import shlex
                try:
                    lexer = shlex.shlex(options, posix=True)
                    lexer.whitespace = ','
                    lexer.whitespace_split = True
                    parts = list(lexer)
                    return [part.strip().strip("'\"") for part in parts]
                except:
                    return [opt.strip().strip("'\"") for opt in options.split(',')]
            
            return [options]
        
        return [str(options)]
    
    def _format_options(self, options: List[str]) -> str:
        if not options:
            return ""
        
        formatted = []
        for i, option in enumerate(options):
            if i >= len(self._choice_labels):
                break
            label = self._choice_labels[i]
            
            option_text = str(option).strip()
            option_text = re.sub(r'^[A-Z][).:]\s*', '', option_text)
            option_text = re.sub(r'^\([A-Z]\)\s*', '', option_text)
            
            if self._clean_latex_enabled:
                option_text = self._clean_latex(option_text)
            
            formatted.append(f"{label}. {option_text}")
        
        return "\n".join(formatted)
    
    def _normalize_answer_internal(self, raw_answer: Any, options: Any = None) -> str:
        answer_type = self._answer_type
        
        if answer_type == self.ANSWER_TYPE_MULTIPLE_CHOICE:
            return self._normalize_multiple_choice_answer(raw_answer, options)
        elif answer_type == self.ANSWER_TYPE_BOOLEAN:
            return self._normalize_boolean_answer(raw_answer)
        elif answer_type == self.ANSWER_TYPE_NUMERIC:
            return self._normalize_numeric_answer(raw_answer)
        else:
            return self._normalize_open_ended_answer(raw_answer)
    
    def _normalize_multiple_choice_answer(self, raw_answer: Any, options: Any) -> str:
        if not self._valid_answers:
            logger.warning("No valid answers defined for multiple choice dataset")
            return self._fallback_normalize_answer(raw_answer, options)
        
        answer_str = str(raw_answer).strip().upper()
        
        valid_letters = ''.join(sorted(self._valid_answers))
        letter_match = re.search(rf'^[{valid_letters}]$', answer_str)
        if letter_match:
            return letter_match.group(0)
        
        patterns = self._get_multiple_choice_patterns()
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in self._valid_answers:
                    return letter
        
        if answer_str.isdigit():
            try:
                idx = int(answer_str) - 1
                valid_list = sorted(self._valid_answers)
                if 0 <= idx < len(valid_list):
                    return valid_list[idx]
            except:
                pass
        
        if options:
            parsed_options = self._parse_options_internal(options)
            for i, option in enumerate(parsed_options):
                option_lower = option.lower()
                answer_lower = answer_str.lower()
                if answer_lower in option_lower or option_lower in answer_lower:
                    valid_list = sorted(self._valid_answers)
                    if i < len(valid_list):
                        return valid_list[i]
        
        valid_list = sorted(self._valid_answers)
        return valid_list[0] if valid_list else 'A'
    
    def _get_multiple_choice_patterns(self) -> List[str]:
        valid_letters = ''.join(sorted(self._valid_answers)) if self._valid_answers else 'ABCDE'
        
        return [
            rf'^OPTION\s*[{valid_letters}]',
            rf'^CHOICE\s*[{valid_letters}]',
            rf'^ANSWER\s*[:\-]?\s*[{valid_letters}]',
            rf'^THE ANSWER IS\s*[:\-]?\s*[{valid_letters}]'
        ]
    
    def _fallback_normalize_answer(self, raw_answer: Any, options: Any) -> str:
        answer_str = str(raw_answer).strip().upper()
        
        patterns = self._get_multiple_choice_patterns()
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                return match.group(0)[-1].upper()
        
        if re.match(r'^[A-E]$', answer_str):
            return answer_str
        
        if answer_str.isdigit():
            try:
                idx = int(answer_str) - 1
                if 0 <= idx < len(self._choice_labels):
                    return self._choice_labels[idx]
            except:
                pass
        
        return self._choice_labels[0] if self._choice_labels else 'A'
    
    def _normalize_boolean_answer(self, raw_answer: Any) -> str:
        if isinstance(raw_answer, bool):
            return "true" if raw_answer else "false"
        
        answer_str = str(raw_answer).strip().lower()
        
        true_values = {'true', 'yes', 'correct', 'right', '1', 't', 'y'}
        false_values = {'false', 'no', 'incorrect', 'wrong', '0', 'f', 'n'}
        
        if answer_str in true_values:
            return "true"
        elif answer_str in false_values:
            return "false"
        
        patterns = [
            r'the answer is\s*[:\-]?\s*(true|false|yes|no)',
            r'answer\s*[:\-]?\s*(true|false|yes|no)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                extracted = match.group(1).lower()
                if extracted in ['true', 'yes']:
                    return "true"
                elif extracted in ['false', 'no']:
                    return "false"
        
        return "true"
    
    def _normalize_numeric_answer(self, raw_answer: Any) -> str:
        answer_str = str(raw_answer).strip()
        
        if self._clean_latex_enabled:
            answer_str = self._clean_latex(answer_str)
        
        # Extract from GSM8K format
        if "####" in answer_str:
            answer_str = answer_str.split("####")[-1].strip()
        
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str.replace(',', ''))
        if numbers:
            return numbers[-1]
        
        patterns = [
            r'the answer is\s*[:\-]?\s*([\d\.]+)',
            r'answer\s*[:\-]?\s*([\d\.]+)',
            r'result\s*[:\-]?\s*([\d\.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return answer_str
    
    def _normalize_open_ended_answer(self, raw_answer: Any) -> str:
        answer_str = str(raw_answer).strip()
        
        if self._clean_latex_enabled:
            answer_str = self._clean_latex(answer_str)
        
        prefixes = self.get_answer_prefixes()
        for prefix in prefixes:
            if answer_str.lower().startswith(prefix.lower()):
                answer_str = answer_str[len(prefix):].strip()
                break
        
        return answer_str
    
    def process_example(self, example: Dict, split: str) -> Dict:
        question, answer = self._format_question_and_answer_internal(example)
        
        cot_steps = []
        if split != "test":
            for retry in range(self.max_retries):
                try:
                    cot_steps = self.cot_generator.get_cot_steps_with_retry(question, answer)
                    if cot_steps:
                        break
                    logger.warning(f"Empty CoT steps for question: {question[:50]}... Attempt {retry + 1}/{self.max_retries}")
                except Exception as e:
                    logger.error(f"Error generating CoT for question: {e}")
                    if retry == self.max_retries - 1:
                        logger.error(f"Failed after {self.max_retries} retries")
        
        if cot_steps and hasattr(self, '_postprocess_cot_steps'):
            cot_steps = self._postprocess_cot_steps(cot_steps, answer)
        
        return {
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,
            "split": split
        }
    
    def process_batch(self, examples: List[Dict], split: str) -> List[Dict]:
        if split == "test":
            results = []
            for example in examples:
                question, answer = self._format_question_and_answer_internal(example)
                results.append({
                    "question": question,
                    "answer": answer,
                    "cot_steps": [],
                    "split": split
                })
            return results
        
        questions = []
        answers = []
        example_objs = []
        
        for example in examples:
            try:
                question, answer = self._format_question_and_answer_internal(example)
                questions.append(question)
                answers.append(answer)
                example_objs.append((question, answer, example))
            except Exception as e:
                logger.warning(f"Failed to process example: {example}")
                continue
        
        logger.info(f"Batch generating CoT steps for {len(questions)} examples")
        
        try:
            all_cot_steps = self.cot_generator.batch_generate_cot_steps(questions, answers)
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [self.process_example(example, split) for example in examples]
        
        results = []
        for (question, answer, example), cot_steps in zip(example_objs, all_cot_steps):
            if cot_steps and hasattr(self, '_postprocess_cot_steps'):
                cot_steps = self._postprocess_cot_steps(cot_steps, answer)
            
            results.append({
                "question": question,
                "answer": answer,
                "cot_steps": cot_steps,
                "split": split
            })
        
        return results
    
    def _should_use_batch_processing(self, split: str, backend: str) -> bool:
        return (backend == 'vllm' and self.batch_size > 1 and split != "test")
    
    def _generate_for_split(self, data, split: str, num_examples: Optional[int] = None) -> List[Dict]:
        backend = self.cot_generator.config.get('generator', {}).get('backend', 'openai')
        output_file = self.get_output_filename(split)
        results = []
        
        data_list = list(data)
        if num_examples and len(data_list) > num_examples:
            indices = random.sample(range(len(data_list)), num_examples)
            data_list = [data_list[i] for i in indices]
            logger.info(f"Sampled {num_examples} examples for {split} split")
        
        if split == "test":
            logger.info(f"Generating test data without CoT steps")
            for i, example in enumerate(tqdm(data_list, desc=f"Processing {self.dataset_name} {split}")):
                try:
                    question, answer = self._format_question_and_answer_internal(example)
                    results.append({
                        "question": question,
                        "answer": answer,
                        "cot_steps": [],
                        "split": split
                    })
                    
                    if self.incremental_save and (i + 1) % self.incremental_save_interval == 0:
                        self.save_results(results, output_file)
                except Exception as e:
                    logger.warning(f"Failed to process test example {i}: {e}")
                    continue
        else:
            use_batch = self._should_use_batch_processing(split, backend)
            
            if use_batch:
                logger.info(f"Using batch processing with batch_size={self.batch_size}")
                for i in range(0, len(data_list), self.batch_size):
                    batch_data = data_list[i:i + self.batch_size]
                    batch_num = i // self.batch_size + 1
                    total_batches = (len(data_list) + self.batch_size - 1) // self.batch_size
                    logger.info(f"Processing batch {batch_num}/{total_batches}")
                    
                    try:
                        batch_results = self.process_batch(batch_data, split)
                        results.extend(batch_results)
                        
                        if self.incremental_save and (i + 1) % self.incremental_save_interval == 0:
                            self.save_results(results, output_file)
                    except Exception as e:
                        logger.error(f"Failed to process batch starting at {i}: {e}")
                        for example in batch_data:
                            try:
                                processed = self.process_example(example, split)
                                results.append(processed)
                            except Exception as e2:
                                logger.warning(f"Failed to process individual example: {e2}")
                                continue
            else:
                logger.info(f"Processing {split} examples individually")
                for i, example in enumerate(tqdm(data_list, desc=f"Processing {self.dataset_name} {split}")):
                    try:
                        processed = self.process_example(example, split)
                        results.append(processed)
                        
                        if self.incremental_save and (i + 1) % self.incremental_save_interval == 0:
                            self.save_results(results, output_file)
                    except Exception as e:
                        logger.warning(f"Failed to process example {i}: {e}")
                        continue
        
        self.save_results(results, output_file)
        
        if self.filter_output and split != "test":
            logger.info(f"Filtering {split} results")
            results = self.filter_results(results, split)
            output_file = self.get_output_filename(split)
            self.save_results(results, output_file)
        
        logger.info(f"Generated {len(results)} examples for {self.dataset_name} {split}")
        return results
    
    def generate(self, split: str, num_examples: Optional[int] = None) -> List[Dict]:
        try:
            data = self.get_dataset(split)
        except Exception as e:
            logger.error(f"Failed to load {self.dataset_name} dataset: {e}")
            return []
        
        return self._generate_for_split(data, split, num_examples)

def main(args):
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.warning(f"Config file not found: {e}")
        config = {'generator': {'llm_api': {}, 'vllm': {}}}
    
    config['debug'] = args.debug if args.debug is not None else False
    generator_config = config.setdefault('generator', {})
    
    if args.backend is not None:
        backend = args.backend
        generator_config['backend'] = backend
        logger.info(f"Using backend from command line: {backend}")
    elif 'backend' in generator_config:
        backend = generator_config['backend']
        logger.info(f"Using backend from config: {backend}")
    else:
        backend = "openai"
        generator_config['backend'] = backend
        logger.info(f"Using default backend: {backend}")
    
    if backend == 'openai':
        llm_config = generator_config.setdefault('llm_api', {})
        
        if args.api_key:
            llm_config['api_key'] = args.api_key
        
        if args.api_base:
            llm_config['api_base'] = args.api_base
        
        if args.model:
            llm_config['model'] = args.model
        
        if 'model' not in llm_config or not llm_config['model']:
            llm_config['model'] = LLM_API_MODEL
        
        if args.batch_size:
            generator_config['batch_size'] = args.batch_size
            llm_config['batch_size'] = args.batch_size
        elif 'batch_size' not in generator_config:
            generator_config['batch_size'] = 1
            llm_config['batch_size'] = 1
        else:
            llm_config['batch_size'] = generator_config['batch_size']
    elif backend == 'vllm':
        vllm_config = generator_config.setdefault('vllm', {})
        
        if args.model:
            vllm_config['model'] = args.model
        elif 'model' not in vllm_config or not vllm_config['model']:
            logger.error("vLLM backend requires a model path. Use --model to specify a HuggingFace model path.")
            return
        
        if args.batch_size:
            generator_config['batch_size'] = args.batch_size
            vllm_config['batch_size'] = args.batch_size
        elif 'batch_size' not in generator_config:
            generator_config['batch_size'] = 8
            vllm_config['batch_size'] = 8
        else:
            vllm_config['batch_size'] = generator_config['batch_size']
        
        if args.gpu_memory_utilization:
            vllm_config['gpu_memory_utilization'] = args.gpu_memory_utilization
        elif 'gpu_memory_utilization' not in vllm_config:
            vllm_config['gpu_memory_utilization'] = 0.9
        
        if args.tensor_parallel_size:
            vllm_config['tensor_parallel_size'] = args.tensor_parallel_size
        elif 'tensor_parallel_size' not in vllm_config:
            vllm_config['tensor_parallel_size'] = 1
    else:
        logger.error(f"Unknown backend: {backend}. Must be 'openai' or 'vllm'.")
        return

    if args.temperature is not None:
        generator_config['temperature'] = args.temperature
    elif 'temperature' not in generator_config:
        generator_config['temperature'] = 0.3

    if backend == 'openai':
        llm_config['temperature'] = generator_config['temperature']
    else:
        vllm_config['temperature'] = generator_config['temperature']
    
    if args.output_dir:
        generator_config['output_dir'] = args.output_dir
    
    if args.max_retries:
        generator_config['max_retries'] = args.max_retries
    elif 'max_retries' not in generator_config:
        generator_config['max_retries'] = 20
    
    if backend == 'openai':
        llm_config['max_retries'] = generator_config['max_retries']
    else:
        vllm_config['max_retries'] = generator_config['max_retries']
    
    if args.validate is not None:
        generator_config['validate_cot_steps'] = args.validate
    
    if args.dry_run:
        config['dry_run'] = True
    
    try:
        cot_gen = CoTGenerator(config, backend)
        logger.info(f"✓ LLM service configured with backend: {backend}")
        
        if backend == 'openai':
            logger.info(f"  API Base: {llm_config.get('api_base')}")
            logger.info(f"  Model: {llm_config.get('model')}")
        else:
            logger.info(f"  Model Path: {vllm_config.get('model', 'Not specified')}")
            logger.info(f"  Batch Size: {vllm_config.get('batch_size')}")
            logger.info(f"  GPU Memory Utilization: {vllm_config.get('gpu_memory_utilization')}")
            logger.info(f"  Tensor Parallel Size: {vllm_config.get('tensor_parallel_size')}")
        
        logger.info(f"  Temperature: {generator_config.get('temperature')}")
    except Exception as e:
        logger.error(f"Error initializing LLM client: {e}")
        logger.info("Please check your configuration")
        return
    
    if args.dry_run:
        logger.info("✓ Dry run completed - configuration is valid")
        return
    
    from dataset import GENERATOR_MAP

    if args.dataset not in GENERATOR_MAP:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(GENERATOR_MAP.keys())}")
    
    if args.mode == "train":
        num_examples = args.num_examples or generator_config.get('num_train_examples')
    else:
        num_examples = args.num_examples or generator_config.get('num_test_examples')
    
    logger.info(f"{'='*60}")
    logger.info(f"Generating {args.mode} data for {args.dataset}")
    logger.info(f"Backend: {backend}")
    
    if backend == 'openai':
        logger.info(f"API Service: {llm_config.get('api_base')}")
        logger.info(f"Model: {llm_config.get('model')}")
    else:
        logger.info(f"Model: {vllm_config.get('model', 'Not specified')}")
        logger.info(f"Batch Size: {vllm_config.get('batch_size')}")
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Examples: {num_examples or 'all'}")
    logger.info(f"{'='*60}")
    
    generator = GENERATOR_MAP[args.dataset](cot_gen, config, args.dataset)
    results = generator.generate(args.mode, num_examples)
    
    if args.mode == 'train':
        cot_gen.llm_client.print_summary()
    
    logger.info(f"{'='*60}")
    logger.info(f"Generation Complete!")
    logger.info(f"Generated {len(results)} examples for {args.dataset} ({args.mode})")
    logger.info(f"Backend used: {backend}")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    parser = HfArgumentParser(GeneratorArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
