import re
import json
import torch
import random
import transformers
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dataset import DATASET_MAP
from huggingface_hub import login
from preprocess import DataConfig
from train import create_cot_tokens
from load_model import AutoCausalLM
from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass, field, asdict
from peft_model import PeftModelForCausalLMWrapper
from typing import Optional, Dict, List, Tuple, Any
from config import (
    COT_TOKEN_NAMES, DEFAULT_EVAL_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR, MD_PATH, MD_SRC, RESERVED_MODELS,
    load_config, load_datasets_config, update_dataclass_from_config, setup_directories, dataset_names, logger
)

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to trained model checkpoint. Required for evaluation."
        }
    )
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the output directory."}
    )
    max_length: Optional[int] = field(default=1000)
    save_result: Optional[bool] = field(default=True)
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    decoding_scheme: Optional[str] = field(default="default")
    generation_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to generation config file or JSON string"}
    )
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    num_beams: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=None)
    parameter_efficient_mode: Optional[str] = field(
        default='none',
        metadata={"choices": ["none", "prompt-tuning", "lora", "lora+prompt-tuning"]}
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face token required for llama family models."}
    )
    enable_cpu_offload: Optional[bool] = field(default=False)
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config file"}
    )

@dataclass
class DataArguments:
    dataset: str = field(
        default=None,
        metadata={
            "help": "Dataset name.",
            "choices": dataset_names
        }
    )
    seed: Optional[int] = field(default=42)
    batch_size: Optional[int] = field(default=1)
    num_test: Optional[int] = field(default=None)

class TokenizerFactory:
    @staticmethod
    def create(model_name: str, cache_dir: Optional[str]) -> transformers.PreTrainedTokenizer:
        model_name_lower = model_name.lower()
        
        if 'llama' in model_name_lower or 'alpaca' in model_name_lower:
            tokenizer_class = transformers.LlamaTokenizer
        else:
            tokenizer_class = transformers.AutoTokenizer
        
        tokenizer = tokenizer_class.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        tokenizer.padding_side = "left"
        logger.debug(f"Set padding_side to: {tokenizer.padding_side}")
        
        return tokenizer

class ModelPathResolver:
    @staticmethod
    def resolve(model_args: ModelArguments) -> Tuple[str, Optional[Path], Optional[Path], Path]:
        model_name = model_args.model_path
        checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
        input_embedding_file, output_embedding_file = ModelPathResolver._get_embedding_files(
            model_args, checkpoint_dir
        )
        return model_name, input_embedding_file, output_embedding_file, checkpoint_dir
    
    @staticmethod
    def _get_embedding_files(
        model_args: ModelArguments,
        checkpoint_dir: Path
    ) -> Tuple[Optional[Path], Optional[Path]]:
        if 'prompt-tuning' not in model_args.parameter_efficient_mode:
            return None, None
        
        possible_paths = [
            (checkpoint_dir / 'embeddings.pt', None),
            (checkpoint_dir / 'input_embeddings.pt', checkpoint_dir / 'output_embeddings.pt'),
        ]
        
        for input_path, output_path in possible_paths:
            if input_path.exists():
                if output_path is None or output_path.exists():
                    logger.debug(f"Found embedding files: {input_path}, {output_path}")
                    return input_path, output_path
        
        logger.debug("No embedding files found")
        return None, None

class ModelLoader:
    @staticmethod
    def load(
        model_args: ModelArguments,
        model_name: str,
        input_embedding_file: Optional[Path]=None,
        output_embedding_file: Optional[Path]=None,
        num_new_tokens: Optional[int]=None
    ) -> torch.nn.Module:
        logger.debug(f"Loading model {model_name}")
        
        if model_name == MD_PATH:
            if not MD_SRC.exists():
                logger.error(f"Failed to find {MD_SRC}")
            import sys
            sys.path.append(str(MD_SRC))
            from md import MD
            from utils import md_cfg
            model = MD.from_pretrained(ckpt_path=md_cfg.ckpt_path)
        else:
            load_kwargs = ModelLoader._prepare_load_kwargs(
                model_args,
                model_name,
                input_embedding_file,
                output_embedding_file,
                num_new_tokens
            )
            
            model = ModelLoader._load_base_model(load_kwargs)
        
            if 'lora' in model_args.parameter_efficient_mode:
                model = ModelLoader._apply_lora(model_args, model, num_new_tokens)
            
            model = ModelLoader._place_model_on_device(model, model_args)

        return model

    @staticmethod
    def _place_model_on_device(model: torch.nn.Module, model_args: ModelArguments) -> torch.nn.Module:
        try:
            if hasattr(model, 'device'):
                logger.info(f"Model device attribute: {model.device}")
            
            if model_args.load_in_8bit or model_args.load_in_4bit:
                logger.info("Using quantized model")
                return model
            
            if torch.cuda.is_available():
                model = model.to('cuda')
                
                all_on_gpu = True
                for name, param in model.named_parameters():
                    if param.device.type != 'cuda':
                        logger.error(f"Parameter {name} is NOT on GPU: {param.device}")
                        all_on_gpu = False
                
                if all_on_gpu:
                    logger.info("All model parameters are on GPU")
                else:
                    logger.error("Some parameters are NOT on GPU!")
                
                for name, buffer in model.named_buffers():
                    if buffer.device.type != 'cuda':
                        logger.warning(f"Buffer {name} is NOT on GPU: {buffer.device}")
            else:
                logger.info("Model on CPU")
        except Exception as e:
            logger.warning(f"Could not check model device: {e}")
        
        return model
    
    @staticmethod
    def _prepare_load_kwargs(
        model_args: ModelArguments,
        model_name: str,
        input_embedding_file: Optional[Path],
        output_embedding_file: Optional[Path],
        num_new_tokens: int
    ) -> Dict[str, Any]:
        load_kwargs = {
            "n_tokens": num_new_tokens,
            "input_embedding_file": str(input_embedding_file) if input_embedding_file else None,
            "output_embedding_file": str(output_embedding_file) if output_embedding_file else None,
            "pretrained_model_name_or_path": model_name,
            "parameter_efficient_mode": model_args.parameter_efficient_mode,
            "cache_dir": model_args.cache_dir,
            "offload_folder": "offload",
            "offload_state_dict": True
        }
        
        device_map = "auto"

        if model_args.load_in_8bit:
            load_kwargs.update({
                "dtype": torch.float16,
                "device_map": device_map,
                "load_in_8bit": True
            })
            logger.info("Loading model in 8-bit precision")
        elif model_args.load_in_4bit:
            load_kwargs.update({
                "dtype": torch.float16,
                "device_map": device_map,
                "load_in_4bit": True
            })
            logger.info("Loading model in 4-bit precision")
        else:
            load_kwargs.update({
                "dtype": torch.float32,
                "device_map": device_map
            })
            logger.info("Loading model with float32 precision")
        
        return load_kwargs
    
    @staticmethod
    def _load_base_model(load_kwargs: Dict[str, Any]) -> torch.nn.Module:
        try:
            model = AutoCausalLM.from_pretrained(**load_kwargs)
            logger.info(f"Base model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    @staticmethod
    def _apply_lora(
        model_args: ModelArguments,
        model: torch.nn.Module,
        num_new_tokens: int
    ) -> torch.nn.Module:
        try:
            _, _, _, checkpoint_dir = ModelPathResolver.resolve(model_args)
            model = PeftModelForCausalLMWrapper.from_pretrained(
                model,
                checkpoint_dir,
                load_embeddings=True,
                n_tokens=num_new_tokens
            )
            logger.info(f"LoRA adapter loaded successfully from: {checkpoint_dir}")
            return model
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise

class DatasetManager:
    @classmethod
    def get_data_class(cls, dataset_name: str) -> Any:
        if dataset_name not in DATASET_MAP:
            available = list(DATASET_MAP.keys())
            raise ValueError(f"Dataset '{dataset_name}' not implemented. Available: {available}")
        
        return DATASET_MAP[dataset_name]
    
    @staticmethod
    def create_dataset(
        data_class: Any,
        name: str,
        split: str,
        data_config: DataConfig
    ) -> torch.utils.data.Dataset:
        try:
            dataset = data_class(name, split, config=data_config)
            logger.info(f"Dataset created: {split} split with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    @staticmethod
    def sample_dataset(
        dataset: torch.utils.data.Dataset,
        num_samples: int,
        seed: int = 42
    ) -> torch.utils.data.Dataset:
        if len(dataset) <= num_samples:
            logger.debug(f"Dataset size {len(dataset)} <= requested samples {num_samples}, skipping sampling")
            return dataset
        
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_samples)
        
        try:
            return Subset(dataset, indices)
        except:
            logger.warning("Could not create Subset, attempting custom sampling")
            if hasattr(dataset, 'x') and hasattr(dataset, 'y'):
                sampled_x = [dataset.x[i] for i in indices]
                sampled_y = [dataset.y[i] for i in indices]
                dataset.x = sampled_x
                dataset.y = sampled_y
                logger.info(f"Sampled dataset to {len(sampled_x)} examples")
                return dataset
        
        raise ValueError("Dataset cannot be sampled")

class BatchEvaluator:
    @staticmethod
    def evaluate(
        model: torch.nn.Module,
        model_name: str,
        tokenizer: transformers.PreTrainedTokenizer,
        x_text: List[str],
        y_text: List[str],
        dataset: Any,
        max_length: int,
        decoding_scheme: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, List[Dict], Dict[str, int], Dict[str, int]]:
        use_quantization = hasattr(model, 'is_quantized') or (
            hasattr(model, 'config') and
            hasattr(model.config, 'quantization_config')
        )

        if use_quantization:
            logger.debug("Using quantization-aware device handling")
            encoding = BatchEvaluator._encode_inputs_no_device(
                tokenizer,
                x_text,
                max_length
            )
        else:
            try:
                model_device = next(model.parameters()).device
                logger.debug(f"Model device: {model_device}")
                encoding = BatchEvaluator._encode_inputs(
                    tokenizer,
                    x_text,
                    max_length,
                    model_device
                )
            except StopIteration:
                logger.warning("Could not get model device, using default")
                encoding = BatchEvaluator._encode_inputs_no_device(
                    tokenizer,
                    x_text,
                    max_length
                )
        
        generated_ids = BatchEvaluator._generate_text(
            model,
            model_name,
            encoding,
            max_length,
            decoding_scheme,
            generation_kwargs
        )
        
        generated_texts = BatchEvaluator._decode_generated_text(
            tokenizer,
            generated_ids,
            encoding
        )
        
        cleaned_texts = BatchEvaluator._clean_generated_texts(
            generated_texts,
            dataset
        )
        
        return BatchEvaluator._process_results(
            cleaned_texts,
            x_text,
            y_text,
            dataset
        )
    
    @staticmethod
    def _clean_generated_texts(
        generated_texts: List[str],
        dataset: Any
    ) -> List[str]:
        cleaned_texts = []
        dataset_type = dataset.__class__.__name__ if hasattr(dataset, '__class__') else 'unknown'
        
        for text in generated_texts:
            if not text or not text.strip():
                cleaned_texts.append("")
                continue
            
            text = text.strip()
            
            text = BatchEvaluator._apply_general_cleaning(text)
            
            text = BatchEvaluator._apply_dataset_specific_cleaning(text, dataset_type)
            
            text = BatchEvaluator._extract_likely_answer(text, dataset_type)
            
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    @staticmethod
    def _apply_general_cleaning(text: str) -> str:
        if not text:
            return text
        
        conversational_patterns = [
            (r'Okay,\s*let\'s see\s*', ''),
            (r'Let me think\s*', ''),
            (r'Hmm,\s*', ''),
            (r'Wait,\s*', ''),
            (r'Well,\s*', ''),
            (r'So,\s*', ''),
            (r'Now,\s*', ''),
            (r'First,\s*', ''),
            (r'Next,\s*', ''),
            (r'Then,\s*', ''),
            (r'Finally,\s*', ''),
            (r'In summary,\s*', ''),
            (r'To summarize,\s*', ''),
        ]
        
        for pattern, replacement in conversational_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        
        if COT_TOKEN_NAMES:
            for token in COT_TOKEN_NAMES:
                patterns_to_fix = [
                    (rf'<{re.escape(token)}>\s*:\s*', f'<{token}>: '),
                    (rf'<{re.escape(token)}>:\s+', f'<{token}>: '),
                    (rf'<{re.escape(token)}>:(?!\s)', f'<{token}>: '),
                ]
                
                for pattern, replacement in patterns_to_fix:
                    text = re.sub(pattern, replacement, text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_cot_line = False
            if COT_TOKEN_NAMES:
                for token in COT_TOKEN_NAMES:
                    if line.startswith(f'<{token}>:'):
                        is_cot_line = True
                        if ': ' in line:
                            parts = line.split(': ', 1)
                            if len(parts) == 2:
                                tag, content = parts
                                content = re.sub(r'\s+', ' ', content).strip()
                                line = f'{tag}: {content}'
                        break
            
            if not is_cot_line:
                line = re.sub(r'\s+', ' ', line).strip()
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _apply_dataset_specific_cleaning(text: str, dataset_type: str) -> str:
        if not text:
            return text
        
        has_math_indicators = any(indicator in text.lower() for indicator in
                                 ['$', '\\boxed', 'sqrt', 'frac', '^', 'pi', 'equation'])
        
        has_multiple_choice = any(pattern in text for pattern in
                                 ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 'I)', 'J)', 'K)'
                                  'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.'])
        
        has_code = any(pattern in text for pattern in
                      ['def ', 'class ', 'import ', 'print(', 'return ', 'function'])
        
        if has_math_indicators:
            text = BatchEvaluator._clean_math_content(text)
        
        if has_multiple_choice:
            text = BatchEvaluator._clean_multiple_choice_content(text)
        
        if has_code:
            text = BatchEvaluator._clean_code_content(text)
        
        return text
    
    @staticmethod
    def _clean_math_content(text: str) -> str:
        if not text:
            return text
        
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_match = re.search(boxed_pattern, text)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        math_patterns = [
            r'\$\$([^$]+)\$\$',
            r'\$([^$]+)\$',
            r'\\\[([^\]]+)\\\]',
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                last_math = matches[-1].strip()
                last_math = re.sub(r'\s+', ' ', last_math)
                return last_math
        
        math_answer_patterns = [
            r'=\s*([\d\.\+\-\*/\(\)\s]+)(?=[\s\.\n]|$)',
            r'is\s*([\d\.\+\-\*/\(\)\s]+)(?=[\s\.\n]|$)',
        ]
        
        for pattern in math_answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'\s+', '', answer)
                return answer
        
        return text
    
    @staticmethod
    def _clean_multiple_choice_content(text: str) -> str:
        if not text:
            return text
        
        choice_patterns = [
            r'[\(\[]?\s*([A-G])\s*[\)\]]',
            r'Answer:\s*([A-G])',
            r'Option\s*([A-G])',
            r'Choice\s*([A-G])',
        ]
        
        for pattern in choice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[A-G][\.\)]\s*', line):
                return line[0].upper()
        
        return text
    
    @staticmethod
    def _clean_code_content(text: str) -> str:
        if not text:
            return text
        
        code_block_pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[-1].strip()
        
        lines = text.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            if any(pattern in line for pattern in
                  ['def ', 'return ', 'print(', 'import ', 'from ', 'class ']):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines[-3:])
        
        return text
    
    @staticmethod
    def _extract_likely_answer(text: str, dataset_type: str) -> str:
        if not text:
            return text
        
        extraction_strategies = [
            BatchEvaluator._extract_via_structured_patterns,
            BatchEvaluator._extract_via_keywords,
            BatchEvaluator._extract_via_numeric_patterns,
            BatchEvaluator._extract_via_final_line,
        ]
        
        for strategy in extraction_strategies:
            extracted = strategy(text)
            if extracted and extracted.strip():
                return extracted.strip()
        
        if len(text) > 200:
            return text[-100:].strip()
        
        return text.strip()
    
    @staticmethod
    def _extract_via_structured_patterns(text: str) -> str:
        patterns = [
            (r'Answer:\s*([^\n\.]+)', 1),
            (r'The answer is:\s*([^\n\.]+)', 1),
            (r'Final answer:\s*([^\n\.]+)', 1),
            (r'Solution:\s*([^\n\.]+)', 1),
            (r'Result:\s*([^\n\.]+)', 1),
            (r'Therefore,\s*([^\n\.]+)', 1),
            (r'Thus,\s*([^\n\.]+)', 1),
            (r'Hence,\s*([^\n\.]+)', 1),
            (r'So,\s*([^\n\.]+)', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(group).strip()
                answer = re.sub(r'[\.\,\;\:\!\?]*$', '', answer)
                return answer
        
        return ""
    
    @staticmethod
    def _extract_via_keywords(text: str) -> str:
        lines = text.split('\n')
        answer_lines = []
        
        answer_keywords = ['answer', 'result', 'solution', 'final', 'therefore',
                          'thus', 'hence', 'equals', 'is', 'are', 'gives']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in answer_keywords):
                for keyword in answer_keywords:
                    if keyword in line_lower:
                        parts = re.split(keyword, line_lower, maxsplit=1, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            answer = parts[1].strip()
                            answer = re.sub(r'^[:\s]*', '', answer)
                            if answer:
                                answer_lines.append(answer)
                        break
        
        if answer_lines:
            return answer_lines[-1]
        
        return ""
    
    @staticmethod
    def _extract_via_numeric_patterns(text: str) -> str:
        numeric_patterns = [
            r'=\s*([\d\.]+)',
            r'is\s*([\d\.]+)',
            r'are\s*([\d\.]+)',
            r'equals\s*([\d\.]+)',
            r'gives\s*([\d\.]+)',
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1]
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            last_num = numbers[-1]
            num_pos = text.rfind(last_num)
            if num_pos >= 0:
                context_start = max(0, num_pos - 30)
                context_end = min(len(text), num_pos + len(last_num) + 30)
                context = text[context_start:context_end].lower()
                
                if any(indicator in context for indicator in
                      ['answer', 'result', 'solution', '=', 'is', 'are', 'equals']):
                    return last_num
        
        return ""
    
    @staticmethod
    def _extract_via_final_line(text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        last_line = lines[-1]
        
        if len(last_line.split()) <= 10:
            last_line = re.sub(r'^(Answer|Result|Solution|Final):?\s*', '',
                              last_line, flags=re.IGNORECASE)
            return last_line
        
        return ""
    
    @staticmethod
    def _encode_inputs_no_device(
        tokenizer: transformers.PreTrainedTokenizer,
        x_text: List[str],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        return tokenizer(
            x_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    @staticmethod
    def _encode_inputs(
        tokenizer: transformers.PreTrainedTokenizer,
        x_text: List[str],
        max_length: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        encoding = tokenizer(
            x_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {k: v.to(device) for k, v in encoding.items()}
    
    @staticmethod
    def _generate_text(
        model: torch.nn.Module,
        model_name: str,
        encoding: Dict[str, Any],
        max_length: int,
        decoding_scheme: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        max_new_tokens = max_length - encoding['input_ids'].shape[1]
        
        default_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else None,
            "eos_token_id": model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else None,
        }
        
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            model_config_dict = model.generation_config.to_dict()
            model_config_dict.pop('max_length', None)
            model_config_dict.pop('max_new_tokens', None)
            default_kwargs.update(model_config_dict)
        
        if generation_kwargs:
            default_kwargs.update(generation_kwargs)
        
        if decoding_scheme != "default":
            if decoding_scheme == "greedy":
                default_kwargs.update({
                    "do_sample": False,
                    "temperature": 1.0,
                    "num_beams": 1
                })
            elif decoding_scheme == "beam":
                default_kwargs.update({
                    "num_beams": 4,
                    "early_stopping": True,
                    "do_sample": False
                })
            elif decoding_scheme == "sampling":
                default_kwargs.update({
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9
                })
        
        generation_kwargs = {k: v for k, v in default_kwargs.items() if v is not None}
        
        if model_name not in RESERVED_MODELS:
            with torch.no_grad():
                return model.generate(
                    **encoding,
                    **generation_kwargs
                )
        else:
            with torch.no_grad():
                return model.generate(input_ids=encoding['input_ids'])
    
    @staticmethod
    def _decode_generated_text(
        tokenizer: transformers.PreTrainedTokenizer,
        generated_ids: torch.Tensor,
        encoding: Dict[str, torch.Tensor]
    ) -> List[str]:
        try:
            return tokenizer.batch_decode(
                generated_ids[:, encoding['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"Error decoding generated IDs: {e}")
            return [""] * len(generated_ids)
    
    @staticmethod
    def _process_results(
        generated_texts: List[str],
        x_text: List[str],
        y_text: List[str],
        dataset: Any
    ) -> Tuple[int, int, List[Dict], Dict[str, int], Dict[str, int]]:
        batch_correct = 0
        batch_outputs = []
        
        for text, x, y in zip(generated_texts, x_text, y_text):
            text, x, y = str(text), str(x), str(y)
            
            try:
                if dataset.is_correct(text, y):
                    batch_correct += 1
                    result = 'correct'
                else:
                    result = 'wrong'
            except Exception as e:
                logger.warning(f"Error checking correctness: {e}")
                result = 'error'
            
            batch_outputs.append({
                'input': x,
                'target': y,
                'generated_text': text,
                'result': result
            })
        
        return batch_correct, len(x_text), batch_outputs

class ResultSaver:
    @staticmethod
    def save(
        model_args: ModelArguments,
        data_args: DataArguments,
        all_outputs: List[Dict],
        total_correct: int,
        total_examples: int,
        config: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        results_dir = ResultSaver._create_results_dir(model_args, data_args)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_identifier = ResultSaver._get_model_identifier(model_args)
        
        json_path = ResultSaver._save_json_results(
            results_dir, timestamp, model_identifier,
            model_args, data_args, all_outputs,
            total_correct, total_examples, accuracy, config
        )
        
        csv_path = ResultSaver._save_csv_summary(
            results_dir, timestamp, model_identifier, all_outputs
        )
        
        ResultSaver._print_summary(
            model_args, data_args, accuracy,
            total_correct, total_examples, json_path, csv_path
        )
        
        return json_path, csv_path
    
    @staticmethod
    def _create_results_dir(
        model_args: ModelArguments,
        data_args: DataArguments
    ) -> Path:
        results_dir = Path(model_args.output_dir) / data_args.dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    @staticmethod
    def _get_model_identifier(model_args: ModelArguments) -> str:
        if Path(model_args.model_path).exists():
            return Path(model_args.model_path).name
        else:
            return model_args.model_path.replace('/', '_')
    
    @staticmethod
    def _save_json_results(
        results_dir: Path,
        timestamp: str,
        model_identifier: str,
        model_args: ModelArguments,
        data_args: DataArguments,
        all_outputs: List[Dict],
        total_correct: int,
        total_examples: int,
        accuracy: float,
        config: Dict[str, Any]
    ) -> Path:
        json_path = results_dir / f"{model_identifier}_{timestamp}_results.json"
        
        results = {
            "model": model_args.model_path,
            "dataset": data_args.dataset,
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct_predictions": total_correct,
            "timestamp": timestamp,
            "config": {
                "model_args": asdict(model_args),
                "data_args": asdict(data_args),
                **config
            },
            "predictions": all_outputs
        }
        
        def convert_path_to_str(obj):
            if isinstance(obj, dict):
                return {k: convert_path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_path_to_str(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        results = convert_path_to_str(results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.debug(f"Saved detailed results to: {json_path}")
        return json_path
    
    @staticmethod
    def _save_csv_summary(
        results_dir: Path,
        timestamp: str,
        model_identifier: str,
        all_outputs: List[Dict]
    ) -> Optional[Path]:
        try:
            csv_path = results_dir / f"{model_identifier}_{timestamp}_summary.csv"
            
            summary_df = pd.DataFrame([
                {
                    'input': item['input'],
                    'target': item['target'],
                    'generated_text': item['generated_text'],
                    'result': item['result']
                }
                for item in all_outputs
            ])
            
            summary_df.to_csv(csv_path, index=False)
            logger.debug(f"Saved summary to: {csv_path}")
            return csv_path
        except Exception as e:
            logger.warning(f"Could not save CSV summary: {e}")
            return None
    
    @staticmethod
    def _print_summary(
        model_args: ModelArguments,
        data_args: DataArguments,
        accuracy: float,
        total_correct: int,
        total_examples: int,
        json_path: Path,
        csv_path: Optional[Path]
    ):
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model path: {model_args.model_path}")
        logger.info(f"Dataset: {data_args.dataset}")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Correct predictions: {total_correct}")
        logger.info(f"Incorrect predictions: {total_examples - total_correct}")
        logger.info(f"Results saved to: {json_path}")
        if csv_path:
            logger.info(f"Summary saved to: {csv_path}")
        logger.info("=" * 60)

def evaluate() -> float:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    if model_args.model_path is None:
        logger.error("Error: --model-path is required for evaluation")
        raise ValueError("model_path is required")
    
    if data_args.dataset is None:
        logger.error("Error: --dataset is required for evaluation")
        raise ValueError("dataset is required")
    
    try:
        config = load_config(model_args.config if hasattr(model_args, 'config') else None)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Configuration file not found: {e}")
        config = {
            'eval': {}
        }
        logger.info("Using default configuration")
    
    model_args = update_dataclass_from_config(model_args, config, ['common', 'eval'])
    data_args = update_dataclass_from_config(data_args, config, ['common', 'eval'])
    
    setup_directories(config)
    
    if model_args.output_dir is None:
        model_args.output_dir = DEFAULT_EVAL_OUTPUT_DIR
    
    if model_args.hf_hub_token:
        try:
            login(token=model_args.hf_hub_token)
            logger.info("Logged in to Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face Hub: {e}")
    
    model_name, input_embedding_file, output_embedding_file, checkpoint_dir = ModelPathResolver.resolve(model_args)
    
    if model_name not in RESERVED_MODELS:
        tokenizer = TokenizerFactory.create(model_args.model_path, model_args.cache_dir)
        logger.debug(f"Resolved model paths: base={model_name}, checkpoint={checkpoint_dir}")
        
        cot_tokens = create_cot_tokens(model_args, tokenizer)
        num_new_tokens = len(cot_tokens)
        if cot_tokens:
            logger.debug(f"Loaded {len(cot_tokens)} indicator tokens")

        model = ModelLoader.load(
            model_args,
            model_name,
            input_embedding_file,
            output_embedding_file,
            num_new_tokens
        )
    else:
        model = ModelLoader.load(
            model_args,
            model_name
        )
        tokenizer = model.tokenizer
    
    model.eval()
    data_class = DatasetManager.get_data_class(data_args.dataset)
    logger.debug(f"Using dataset class: {data_class.__name__}")
    
    data_config = DataConfig()
    datasets_config = load_datasets_config()
    data_config.dataset = datasets_config[data_args.dataset]

    dataset = DatasetManager.create_dataset(
        data_class,
        data_args.dataset,
        "test",
        data_config
    )
    
    if data_args.num_test and len(dataset) > data_args.num_test:
        dataset = DatasetManager.sample_dataset(
            dataset, data_args.num_test, data_args.seed
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Created dataloader with batch size {data_args.batch_size}")
    
    total_correct = 0
    total_examples = 0
    all_outputs = []
    generation_kwargs = {}
    if model_args.temperature is not None:
        generation_kwargs["temperature"] = model_args.temperature
    if model_args.top_p is not None:
        generation_kwargs["top_p"] = model_args.top_p
    if model_args.top_k is not None:
        generation_kwargs["top_k"] = model_args.top_k
    if model_args.num_beams is not None:
        generation_kwargs["num_beams"] = model_args.num_beams
    if model_args.do_sample is not None:
        generation_kwargs["do_sample"] = model_args.do_sample
    
    if model_args.generation_config:
        try:
            if Path(model_args.generation_config).exists():
                with open(model_args.generation_config, 'r') as f:
                    file_config = json.load(f)
                    generation_kwargs.update(file_config)
            else:
                file_config = json.loads(model_args.generation_config)
                generation_kwargs.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load generation config: {e}")
        
    logger.info(f"Evaluating {len(dataset)} examples in batches of {data_args.batch_size}...")
    
    with tqdm(dataloader, desc="Evaluation", unit="batch") as progress_bar:
        for batch_idx, batch in enumerate(progress_bar):
            x_text, y_text = batch['x'], batch['y']
            
            try:
                batch_correct, batch_size, batch_outputs = BatchEvaluator.evaluate(
                    model=model,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    x_text=x_text,
                    y_text=y_text,
                    dataset=dataset,
                    max_length=model_args.max_length,
                    decoding_scheme=model_args.decoding_scheme,
                    generation_kwargs=generation_kwargs
                )
            except Exception as e:
                logger.error(f"Error evaluating batch {batch_idx}: {e}")
                continue
            
            total_correct += batch_correct
            total_examples += batch_size
            all_outputs.extend(batch_outputs)
            
            current_accuracy = total_correct / total_examples if total_examples > 0 else 0
            progress_bar.set_postfix({
                'accuracy': f'{current_accuracy:.4f}',
                'correct': f'{total_correct}/{total_examples}'
            })
    
    final_accuracy = total_correct / total_examples if total_examples > 0 else 0
    
    if model_args.save_result:
        ResultSaver.save(
            model_args, data_args, all_outputs,
            total_correct, total_examples,
            config
        )
    
    return final_accuracy

if __name__ == "__main__":
    evaluate()
