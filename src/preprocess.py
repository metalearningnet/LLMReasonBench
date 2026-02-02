import re
import os
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Tuple, Set
from config import DEFAULT_DATA_DIR, LOG_PRED, INVALID_ANS, logger

class AnswerNormalizer(ABC):
    @abstractmethod
    def normalize(self, answer: Any) -> Optional[str]:
        pass

class MultipleChoiceNormalizer(AnswerNormalizer):
    def __init__(self, valid_answers: Optional[Set[str]] = None):
        self.valid_answers = valid_answers or {'a', 'b', 'c', 'd', 'e'}
        logger.debug(f"Initialized MultipleChoiceNormalizer with valid answers: {sorted(self.valid_answers)}")
    
    def normalize(self, answer: Any) -> Optional[str]:
        if answer is None:
            logger.debug("Answer is None, returning None")
            return None
        
        answer_str = str(answer).strip().lower()
        logger.debug(f"Normalizing answer: '{answer}' -> '{answer_str}'")
        
        answer_str = re.sub(r'[().,\-:;]', '', answer_str)
        
        if len(answer_str) == 1 and answer_str in self.valid_answers:
            logger.debug(f"Answer is already valid single letter: {answer_str}")
            return answer_str
        
        patterns = [
            (r'^\s*([a-z])\s*$', re.IGNORECASE),
            (r'option\s*([a-z])', re.IGNORECASE),
            (r'choice\s*([a-z])', re.IGNORECASE),
            (r'answer\s*[:\-]?\s*([a-z])', re.IGNORECASE),
            (r'the answer is\s*[:\-]?\s*([a-z])', re.IGNORECASE)
        ]
        
        for pattern, flags in patterns:
            match = re.search(pattern, answer_str, flags)
            if match:
                letter = match.group(1).lower()
                if letter in self.valid_answers:
                    logger.debug(f"Pattern '{pattern}' matched letter: {letter}")
                    return letter
        
        if answer_str.isdigit():
            try:
                idx = int(answer_str) - 1
                if 0 <= idx < len(self.valid_answers):
                    normalized = chr(ord('a') + idx)
                    logger.debug(f"Converted numeric index {answer_str} to letter: {normalized}")
                    return normalized
            except ValueError:
                pass
        
        logger.warning(f"Could not normalize answer: '{answer}'")
        return None

class BooleanAnswerNormalizer(AnswerNormalizer):
    def normalize(self, answer: Any) -> Optional[str]:
        if answer is None:
            logger.debug("Answer is None, returning None")
            return None
        
        logger.debug(f"Normalizing boolean answer: '{answer}'")
        
        if isinstance(answer, bool):
            result = 'true' if answer else 'false'
            logger.debug(f"Boolean input {answer} -> {result}")
            return result
        
        answer_str = str(answer).strip().lower()
        
        true_values = {'true', 'yes', 'correct', 'right', '1', 't', 'y'}
        false_values = {'false', 'no', 'incorrect', 'wrong', '0', 'f', 'n'}
        
        if answer_str in true_values:
            logger.debug(f"Answer '{answer_str}' recognized as true")
            return 'true'
        elif answer_str in false_values:
            logger.debug(f"Answer '{answer_str}' recognized as false")
            return 'false'
        
        patterns = [
            (r'the answer is:\s*(true|false|yes|no)', re.IGNORECASE),
            (r'answer:\s*(true|false|yes|no)', re.IGNORECASE),
            (r'\b(true|false|yes|no)\b', re.IGNORECASE)
        ]
        
        for pattern, flags in patterns:
            match = re.search(pattern, answer_str, flags)
            if match:
                extracted = match.group(1).lower()
                if extracted in ['true', 'yes']:
                    logger.debug(f"Pattern '{pattern}' extracted 'true' from '{answer_str}'")
                    return 'true'
                elif extracted in ['false', 'no']:
                    logger.debug(f"Pattern '{pattern}' extracted 'false' from '{answer_str}'")
                    return 'false'
        
        if answer_str:
            first_char = answer_str[0]
            if first_char in ['t', 'y']:
                logger.debug(f"First character '{first_char}' indicates true")
                return 'true'
            elif first_char in ['f', 'n']:
                logger.debug(f"First character '{first_char}' indicates false")
                return 'false'
        
        logger.warning(f"Could not normalize boolean answer: '{answer}'")
        return None

class NumericAnswerNormalizer(AnswerNormalizer):
    def normalize(self, answer: Any) -> Optional[str]:
        if answer is None or not isinstance(answer, (str, int, float)):
            logger.debug(f"Invalid answer type or None: {type(answer)}")
            return None
        
        answer_str = str(answer).strip()
        logger.debug(f"Normalizing numeric answer: '{answer_str}'")
        
        # Handle GSM8K #### format
        if "####" in answer_str:
            answer_str = answer_str.split("####")[-1].strip()
            logger.debug(f"Extracted from #### format: '{answer_str}'")
        
        prefixes = [
            "the answer is:",
            "the answer is",
            "answer:",
            "answer is:",
            "answer is",
            "final answer:",
            "final answer is:"
        ]
        
        for prefix in prefixes:
            if answer_str.lower().startswith(prefix):
                answer_str = answer_str[len(prefix):].strip()
                logger.debug(f"Removed prefix '{prefix}': '{answer_str}'")
                break
        
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str.replace(',', ''))
        if numbers:
            result = numbers[-1]
            logger.debug(f"Extracted number: {result} from '{answer_str}'")
            return result
        
        logger.warning(f"Could not find numeric value in: '{answer_str}'")
        return None

class AnswerExtractor(ABC):
    def __init__(self, invalid_ans: str = INVALID_ANS):
        self.invalid_ans = invalid_ans
        logger.debug(f"Initialized AnswerExtractor with invalid_ans: '{invalid_ans}'")
    
    @abstractmethod
    def extract(self, completion: str) -> str:
        pass

class MultipleChoiceExtractor(AnswerExtractor):
    def __init__(self,
                 valid_answers: Optional[Set[str]] = None,
                 invalid_ans: str = INVALID_ANS):
        super().__init__(invalid_ans)
        self.valid_answers = valid_answers or {'a', 'b', 'c', 'd', 'e'}
        logger.debug(f"Initialized MultipleChoiceExtractor with valid answers: {sorted(self.valid_answers)}")
    
    def extract(self, completion: str) -> str:
        if not completion or not completion.strip():
            logger.debug("Empty completion, returning invalid answer")
            return self.invalid_ans
        
        completion = completion.strip()
        logger.debug(f"Extracting multiple-choice answer from: '{completion[:100]}...'")
        
        patterns = [
            (r"the answer is\s*[:]?\s*([a-z])", re.IGNORECASE),
            (r"answer\s*[:]?\s*([a-z])", re.IGNORECASE),
            (r"final answer\s*[:]?\s*([a-z])", re.IGNORECASE),
            (r"correct answer\s*[:]?\s*([a-z])", re.IGNORECASE)
        ]
        
        for pattern, flags in patterns:
            try:
                match = re.search(pattern, completion, flags)
                if match:
                    letter = match.group(1).lower()
                    if letter in self.valid_answers:
                        logger.debug(f"Pattern '{pattern}' matched letter: {letter}")
                        return letter
            except Exception as e:
                logger.warning(f"Pattern '{pattern}' failed with error: {e}")
                continue
        
        completion_lower = completion.lower()
        for char in completion_lower:
            if char in self.valid_answers:
                idx = completion_lower.find(char)
                if idx >= 0:
                    prev_char = completion_lower[idx-1] if idx > 0 else ' '
                    next_char = completion_lower[idx+1] if idx < len(completion_lower)-1 else ' '
                    
                    if not prev_char.isalpha() and not next_char.isalpha():
                        logger.debug(f"Found isolated letter '{char}' at position {idx}")
                        return char
        
        logger.debug(f"No valid answer found in completion: '{completion[:100]}...'")
        return self.invalid_ans

class BooleanAnswerExtractor(AnswerExtractor):
    def extract(self, completion: str) -> str:
        if not completion or not completion.strip():
            logger.debug("Empty completion, returning invalid answer")
            return self.invalid_ans
        
        completion = completion.strip()
        completion_lower = completion.lower()
        logger.debug(f"Extracting boolean answer from: '{completion[:100]}...'")
        
        letter_match = re.search(r'\b([ab])\b', completion_lower)
        if letter_match:
            letter = letter_match.group(1)
            logger.debug(f"Found letter '{letter}' in completion")
            if letter == 'a':
                return 'true'
            elif letter == 'b':
                return 'false'
        
        if completion_lower.startswith('f') and len(completion_lower) <= 5:
            logger.debug("Found 'F' which likely means False")
            return 'false'
        
        if completion_lower.startswith('t') and len(completion_lower) <= 5:
            logger.debug("Found 'T' which likely means True")
            return 'true'
        
        true_values = {'true', 'yes', 'correct', 'right', '1', 't', 'y'}
        false_values = {'false', 'no', 'incorrect', 'wrong', '0', 'f', 'n'}
        
        if completion_lower in true_values:
            logger.debug(f"Answer '{completion_lower}' recognized as true")
            return 'true'
        elif completion_lower in false_values:
            logger.debug(f"Answer '{completion_lower}' recognized as false")
            return 'false'
        
        patterns = [
            (r'the answer is:\s*(true|false|yes|no)', re.IGNORECASE),
            (r'answer:\s*(true|false|yes|no)', re.IGNORECASE),
            (r'\b(true|false|yes|no)\b', re.IGNORECASE)
        ]
        
        for pattern, flags in patterns:
            match = re.search(pattern, completion_lower, flags)
            if match:
                extracted = match.group(1).lower()
                if extracted in ['true', 'yes']:
                    logger.debug(f"Pattern '{pattern}' extracted 'true' from '{completion_lower}'")
                    return 'true'
                elif extracted in ['false', 'no']:
                    logger.debug(f"Pattern '{pattern}' extracted 'false' from '{completion_lower}'")
                    return 'false'
        
        if re.search(r'\*\*', completion):
            logger.debug(f"Found '**' in completion, can't extract boolean")
            return self.invalid_ans
        
        if completion_lower:
            first_char = completion_lower[0]
            if first_char in ['t', 'y']:
                logger.debug(f"First character '{first_char}' indicates true")
                return 'true'
            elif first_char in ['f', 'n']:
                logger.debug(f"First character '{first_char}' indicates false")
                return 'false'
        
        logger.warning(f"Could not normalize boolean answer: '{completion}'")
        return self.invalid_ans

class EnhancedNumericExtractor(AnswerExtractor):
    def extract(self, completion: str) -> str:
        if not completion:
            return self.invalid_ans
        
        completion = completion.strip()
        logger.debug(f"Enhanced numeric extraction from: '{completion[:100]}...'")
        
        explicit_patterns = [
            (r'the answer is\s*[:]?\s*([^\.\n]+)', re.IGNORECASE),
            (r'final answer\s*[:]?\s*([^\.\n]+)', re.IGNORECASE),
            (r'answer\s*[:]?\s*([^\.\n]+)', re.IGNORECASE),
            (r'result\s*[:]?\s*([^\.\n]+)', re.IGNORECASE),
            (r'solution\s*[:]?\s*([^\.\n]+)', re.IGNORECASE)
        ]
        
        for pattern, flags in explicit_patterns:
            matches = list(re.finditer(pattern, completion, flags))
            if matches:
                last_match = matches[-1]
                answer_text = last_match.group(1).strip()
                logger.debug(f"Found explicit answer pattern: '{pattern}' -> '{answer_text}'")
                
                numbers = self._extract_numbers_from_text(answer_text)
                if numbers:
                    logger.debug(f"Extracted number from answer text: {numbers[-1]}")
                    return numbers[-1]
        
        boxed_result = self._extract_boxed_answer(completion)
        if boxed_result != self.invalid_ans:
            return boxed_result
        
        gsm8k_result = self._extract_gsm8k_format(completion)
        if gsm8k_result != self.invalid_ans:
            return gsm8k_result
        
        last_line_result = self._extract_from_last_line(completion)
        if last_line_result != self.invalid_ans:
            return last_line_result
        
        all_numbers = re.findall(r'-?\d+(?:\.\d+)?', completion.replace(',', ''))
        if all_numbers:
            last_number = all_numbers[-1]
            logger.debug(f"Using last number found in completion: {last_number}")
            return last_number
        
        logger.warning(f"No numeric answer found in completion: '{completion[:100]}...'")
        return self.invalid_ans
    
    def _extract_numbers_from_text(self, text: str) -> List[str]:
        text = text.replace(',', '')
        
        fractions = re.findall(r'-?\d+\s*/\s*\d+', text)
        if fractions:
            return fractions
        
        return re.findall(r'-?\d+(?:\.\d+)?', text)
    
    def _extract_boxed_answer(self, completion: str) -> str:
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'\\boxed{([^}]+)}',
            r'boxed{([^}]+)}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                numbers = self._extract_numbers_from_text(answer)
                if numbers:
                    return numbers[-1]
        
        return self.invalid_ans
    
    def _extract_gsm8k_format(self, completion: str) -> str:
        match = re.search(r'####\s*([\d\.,\s]+(?:\.\d+)?)', completion)
        if match:
            answer = match.group(1).strip()
            answer = answer.replace(',', '').replace(' ', '')
            if answer:
                return answer
        return self.invalid_ans
    
    def _extract_from_last_line(self, completion: str) -> str:
        lines = completion.split('\n')
        
        for line in reversed(lines[-3:]):
            line = line.strip()
            if not line:
                continue
            
            if len(line) < 50:
                numbers = self._extract_numbers_from_text(line)
                if numbers:
                    return numbers[-1]
            
            patterns = [
                r'=\s*([\d\.]+)',
                r'is\s+([\d\.]+)',
                r'=\s*\$?([\d\.]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    number = match.group(1).strip()
                    if number:
                        return number
        
        return self.invalid_ans

@dataclass
class DataConfig:
    invalid_ans: str = INVALID_ANS
    answer_normalizer: Optional[AnswerNormalizer] = None
    answer_extractor: Optional[AnswerExtractor] = None
    valid_answers: Optional[Set[str]] = None
    is_multiple_choice: bool = False
    dataset: dict = None

class BaseData(Dataset, ABC):
    ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"
    ANSWER_TYPE_BOOLEAN = "boolean"
    ANSWER_TYPE_NUMERIC = "numeric"
    ANSWER_TYPE_OPEN_ENDED = "open_ended"
    
    def __init__(self, name: str, split: str, config: DataConfig):
        super().__init__()
        self.name = name
        self.split = split
        self.config = config
        self.dataset_config = config.dataset
        
        logger.debug(f"Initializing {self.__class__.__name__} for split: {split}")
        
        self._instruction = self.get_instruction()
        self._answer_type = self.get_answer_type()
        self._normalizer = self._get_normalizer()
        self._extractor = self._get_extractor()
        
        logger.debug(f"Normalizer: {self._normalizer.__class__.__name__}")
        logger.debug(f"Extractor: {self._extractor.__class__.__name__}")
        
        self.data = self.load_data(split)
        logger.debug(f"Loaded {len(self.data)} raw examples")
        
        self.x, self.y = self._prepare_all_data()
        
        self._validate_data()
        
        logger.debug(f"Dataset ready: {len(self.x)} examples")
    
    def _get_normalizer(self) -> AnswerNormalizer:
        if self.config.answer_normalizer:
            logger.debug("Using provided answer_normalizer from config")
            return self.config.answer_normalizer
        
        answer_type = self.get_answer_type()
        logger.debug(f"Selecting normalizer for answer type: {answer_type}")
        
        if answer_type == self.ANSWER_TYPE_NUMERIC:
            logger.debug("Selecting NumericAnswerNormalizer")
            return NumericAnswerNormalizer()
        elif answer_type == self.ANSWER_TYPE_BOOLEAN:
            logger.debug("Selecting BooleanAnswerNormalizer")
            return BooleanAnswerNormalizer()
        elif answer_type == self.ANSWER_TYPE_MULTIPLE_CHOICE:
            valid_answers = self.get_valid_answers()
            if valid_answers:
                logger.debug(f"Selecting MultipleChoiceNormalizer with valid answers: {valid_answers}")
                return MultipleChoiceNormalizer(valid_answers)
            else:
                logger.warning(f"Multiple choice dataset has no valid answers, using default")
                return MultipleChoiceNormalizer()
        else:
            logger.warning(f"Unknown answer type, defaulting to MultipleChoiceNormalizer")
            return MultipleChoiceNormalizer()
    
    def _get_extractor(self) -> AnswerExtractor:
        if self.config.answer_extractor:
            logger.debug(f"Using config-provided extractor: {self.config.answer_extractor.__class__.__name__}")
            return self.config.answer_extractor
        
        answer_type = self.get_answer_type()
        logger.info(f"Dataset {self.__class__.__name__} has answer type: {answer_type}")
        
        if answer_type == self.ANSWER_TYPE_NUMERIC:
            logger.info("Selecting EnhancedNumericExtractor for numeric dataset")
            return EnhancedNumericExtractor(self.config.invalid_ans)
        elif answer_type == self.ANSWER_TYPE_BOOLEAN:
            logger.info("Selecting BooleanAnswerExtractor for boolean dataset")
            return BooleanAnswerExtractor(invalid_ans=self.config.invalid_ans)
        elif answer_type == self.ANSWER_TYPE_MULTIPLE_CHOICE:
            valid_answers = self.get_valid_answers()
            if valid_answers:
                logger.info(f"Selecting MultipleChoiceExtractor with valid answers: {valid_answers}")
                return MultipleChoiceExtractor(valid_answers, self.config.invalid_ans)
            else:
                logger.warning(f"Multiple choice dataset {self.__class__.__name__} has no valid answers specified")
                return MultipleChoiceExtractor(invalid_ans=self.config.invalid_ans)
        else:
            logger.warning(f"Unknown answer type '{answer_type}', defaulting to MultipleChoiceExtractor")
            return MultipleChoiceExtractor(invalid_ans=self.config.invalid_ans)
    
    @abstractmethod
    def load_data(self, split: str) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    @abstractmethod
    def get_instruction(self) -> str:
        raise NotImplementedError
    
    def get_answer_type(self) -> str:
        return self.dataset_config['answer_type']
    
    def get_valid_answers(self) -> Optional[Set[str]]:
        valid_answers = self.dataset_config.get('valid_answers')
        if valid_answers:
            return set(str(v).lower() for v in self.dataset_config['valid_answers'])
    
    def parse_q_a(self, example: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[List[str]], Optional[str]]:
        question = example.get("question", "").strip()
        raw_answer = example.get("answer", "")
        
        if not question:
            logger.warning(f"Skipping example: missing question. Example keys: {list(example.keys())}")
            return None, None, None, None
        
        if not raw_answer:
            logger.warning(f"Skipping example: missing answer. Question: {question[:50]}...")
            return None, None, None, None
        
        logger.debug(f"Parsing example: Q='{question[:50]}...', A='{raw_answer[:50]}...'")
        
        answer = self._normalizer.normalize(raw_answer)
        if answer is None:
            logger.warning(f"Could not normalize answer: '{raw_answer[:50]}...'")
            return None, None, None, None
        
        logger.debug(f"Normalized answer: '{raw_answer[:50]}...' -> '{answer}'")
        
        if self._answer_type == self.ANSWER_TYPE_MULTIPLE_CHOICE:
            options = example.get("options") or example.get("choices")
            if options and answer:
                if not self._is_answer_valid_for_options(answer, options):
                    logger.warning(f"Answer '{answer}' not valid for {len(options)} options")
                    return None, None, None, None
        
        cot_steps = example.get("cot_steps", [])
        if not isinstance(cot_steps, list):
            logger.warning(f"cot_steps is not a list: {type(cot_steps)}")
            cot_steps = []
        
        cot_steps = [step.strip() for step in cot_steps if step and step.strip()]
        
        if not cot_steps:
            cot_steps = [self._get_default_cot_prompt()]
            logger.debug("Generated default CoT steps")
        
        logger.debug(f"Parsed with {len(cot_steps)} CoT steps, answer: {answer}")
        return self._instruction, question, cot_steps, answer
    
    def _is_answer_valid_for_options(self, answer: str, options: List[str]) -> bool:
        if not answer or not options:
            return False
        
        try:
            answer_idx = ord(answer.lower()) - ord('a')
            return 0 <= answer_idx < len(options)
        except:
            return False
    
    def _get_default_cot_prompt(self) -> str:
        if self._answer_type == self.ANSWER_TYPE_BOOLEAN:
            return "Let me reason through this yes/no question step by step."
        elif self._answer_type == self.ANSWER_TYPE_NUMERIC:
            return "Let me solve this math problem step by step."
        else:
            return "Let me think through this question step by step."
    
    def _prepare_input(self, instruction: str, question: str, example: Dict[str, Any]) -> str:
        options = example.get("options") or example.get("choices")
        if options:
            question = self._format_question_with_options(question, options)
        
        input_text = f"{instruction}\n\n{question}\n\n"
        logger.debug(f"Prepared input text ({len(input_text)} chars)")
        return input_text
    
    def _format_question_with_options(self, question: str, options: List[str]) -> str:
        if not options:
            return question
        
        formatted = f"{question}\n\n"
        
        valid_answers = self.get_valid_answers()
        if valid_answers:
            max_options = min(len(options), len(valid_answers))
        else:
            max_options = len(options)
        
        for i in range(max_options):
            label = chr(ord('A') + i)
            formatted += f"{label}. {options[i]}\n"
        
        if valid_answers and len(options) > len(valid_answers):
            logger.warning(f"Question has {len(options)} options, truncated to {len(valid_answers)}")
        
        return formatted
    
    def _prepare_solution(self, cot_steps: List[str], answer: str) -> str:
        logger.debug(f"Preparing solution with {len(cot_steps)} CoT steps, answer: {answer}")
        
        if cot_steps:
            if len(cot_steps) == 1:
                cot_text = cot_steps[0]
            else:
                cot_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cot_steps)])
        else:
            cot_text = ""
        
        formatted_answer = f"The answer is: {answer}"
        
        if self._answer_type == self.ANSWER_TYPE_NUMERIC:
            if not any(word in formatted_answer.lower() for word in ['therefore', 'thus', 'so']):
                formatted_answer = f"Therefore, {formatted_answer.lower()}"
        
        if cot_text:
            solution = f"{cot_text}\n\n{formatted_answer}"
        else:
            solution = formatted_answer
        
        logger.debug(f"Solution prepared ({len(solution)} chars)")
        return solution
    
    def _prepare_all_data(self) -> Tuple[List[str], List[str]]:
        x_list, y_list = [], []
        skipped = 0
        
        logger.debug(f"Preparing {len(self.data)} examples...")
        
        for idx, example in enumerate(self.data):
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"  Processed {idx}/{len(self.data)} examples...")
            
            inst, q, cot_steps, ans = self.parse_q_a(example)
            
            if any(v is None for v in [inst, q, cot_steps, ans]):
                skipped += 1
                if skipped <= 5:
                    logger.debug(f"  Skipped example {idx}: parse_q_a returned None")
                continue
            
            input_text = self._prepare_input(inst, q, example)
            solution = self._prepare_solution(cot_steps, ans)
            
            x_list.append(input_text)
            y_list.append(solution)
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} examples during preparation ({skipped/len(self.data)*100:.1f}%)")
        
        logger.debug(f"Prepared {len(x_list)} valid examples")
        return x_list, y_list
    
    def _validate_data(self) -> None:
        if len(self.x) != len(self.y):
            error_msg = f"X and Y length mismatch: {len(self.x)} != {len(self.y)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(self.x) == 0:
            error_msg = "No data was loaded after preparation"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Data validation passed: {len(self.x)} examples ready")
    
    def _normalize_numeric_string(self, text: str) -> Optional[str]:
        if not text or text == self.config.invalid_ans:
            return None
        
        text = text.replace(',', '').replace(' ', '').strip().lower()
        
        text = re.sub(r'[^\d\.\-]', '', text)
        
        if not text:
            return None
        
        if '/' in text:
            try:
                parts = text.split('/')
                if len(parts) == 2:
                    num, denom = float(parts[0]), float(parts[1])
                    if denom != 0:
                        return str(num / denom)
            except:
                return None
        
        if text.endswith('%'):
            try:
                value = float(text[:-1]) / 100
                return str(value)
            except:
                return None
        
        return text
    
    def _compare_numeric_answers(self, pred: str, gt: str) -> bool:
        pred_norm = self._normalize_numeric_string(pred)
        gt_norm = self._normalize_numeric_string(gt)
        
        if not pred_norm or not gt_norm:
            return False
        
        try:
            pred_float = float(pred_norm)
            gt_float = float(gt_norm)
            
            tolerance = 1e-6
            
            if abs(pred_float - gt_float) < tolerance:
                return True
            
            if round(pred_float) == round(gt_float):
                return True
                
            return False
        except (ValueError, TypeError):
            return pred_norm == gt_norm
    
    def extract_answer(self, completion: str) -> str:
        logger.debug(f"Extracting answer from completion ({len(completion)} chars)")
        result = self._extractor.extract(completion)
        
        if result == self.config.invalid_ans:
            logger.debug(f"Could not extract valid answer from: '{completion[:100]}...'")
        else:
            logger.debug(f"Extracted answer: '{result}'")
        
        return result
    
    def is_correct(self, model_completion: str, ground_truth: str) -> bool:
        logger.debug(f"Checking correctness: completion vs ground_truth")
        
        pred_answer = self.extract_answer(model_completion)
        gt_answer = self.extract_answer(ground_truth)
        
        if gt_answer == self.config.invalid_ans:
            for line in ground_truth.split('\n'):
                extracted = self._extractor.extract(line)
                if extracted != self.config.invalid_ans:
                    gt_answer = extracted
                    logger.debug(f"Extracted ground truth from line: '{gt_answer}'")
                    break
        
        if gt_answer == self.config.invalid_ans:
            logger.error(f"Invalid ground truth after extraction: {ground_truth[:100]}...")
            return False
        
        if self._answer_type == self.ANSWER_TYPE_NUMERIC:
            return self._compare_numeric_answers(pred_answer, gt_answer)
        
        pred_answer = pred_answer.strip().lower()
        gt_answer = gt_answer.strip().lower()
        
        if self._answer_type == self.ANSWER_TYPE_BOOLEAN:
            if pred_answer in ['yes', 'y']:
                pred_answer = 'true'
            elif pred_answer in ['no', 'n']:
                pred_answer = 'false'
            
            if gt_answer in ['yes', 'y']:
                gt_answer = 'true'
            elif gt_answer in ['no', 'n']:
                gt_answer = 'false'
        
        is_correct = pred_answer == gt_answer
        
        if LOG_PRED:
            status = "✓" if is_correct else "✗"
            logger.info(f"{status} PREDICTION: pred='{pred_answer}', gt='{gt_answer}', correct={is_correct}")
            if not is_correct:
                logger.debug(f"  Model output: '{model_completion[:200]}...'")
        
        return is_correct
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        logger.debug(f"Getting item {idx}/{len(self)}")
        return {
            'x': self.x[idx],
            'y': self.y[idx]
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split={self.split}, examples={len(self)}, type={self._answer_type})"
    
    def print_sample(self, n: int = 3) -> None:
        logger.info(f"Sample of {min(n, len(self))} examples:")
        for i in range(min(n, len(self))):
            item = self[i]
            logger.info(f"\n{'='*60}")
            logger.info(f"Example {i}:")
            logger.info(f"Input (x) [{len(item['x'])} chars]:")
            logger.info(f"  {item['x'][:200]}...")
            logger.info(f"Target (y) [{len(item['y'])} chars]:")
            logger.info(f"  {item['y'][:200]}...")

class JsonBasedData(BaseData):
    def __init__(self, name: str, split: str, config: DataConfig):
        self.json_path_pattern = str(DEFAULT_DATA_DIR / f"{name}_{split}.json")
        logger.debug(f"JSON path pattern: {self.json_path_pattern}")
        super().__init__(name, split, config)
    
    def load_data(self, split: str) -> List[Dict[str, Any]]:
        json_path = str(self.json_path_pattern).format(split=split)
        
        if not os.path.exists(json_path):
            error_msg = f"JSON file not found: {json_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.debug(f"Loading data from {json_path}")
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                error_msg = f"Expected list in JSON, got {type(data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Successfully loaded {len(data)} examples from {json_path}")
            
            if data and len(data) > 0:
                logger.debug(f"Sample data keys: {list(data[0].keys())}")
            
            return data
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON: {e}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to load JSON: {e}"
            logger.error(error_msg)
            raise
