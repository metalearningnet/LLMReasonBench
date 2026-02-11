import json
import copy
import torch
import string
import difflib
import logging
import transformers
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from datasets import Dataset as HFDataset, load_dataset
from trl import GRPOConfig, GRPOTrainer, DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from config import DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR, logger, load_rl_config, load_config as load_base_config
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM
)

logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

@dataclass
class RLTrainingArguments(transformers.TrainingArguments):
    beta: float = field(default=0.1)
    dpo_loss_type: str = field(default="sigmoid")
    
    num_rollouts: int = field(default=4)
    top_p: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=None)
    
    max_prompt_length: int = field(default=512)
    max_completion_length: int = field(default=1024)
    
    dataset_name: str = field(default=None)
    dataset_split: str = field(default="train")
    eval_split: str = field(default="validation")
    
    dpo_field_mappings: Optional[Dict[str, str]] = field(default=None)
    grpo_field_mappings: Optional[Dict[str, Any]] = field(default=None)
    
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    training_mode: str = field(default="dpo")
    
    model_init_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    ref_model_init_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    num_train: Optional[int] = field(default=None)
    checkpoint_dir: str = field(default=None)

class RLConfigManager:
    @staticmethod
    def load_merged_config(
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if base_config is None:
            base_config = load_base_config()
        
        rl_config = load_rl_config()
        merged_config = base_config.copy()
        merged_config["rl"] = rl_config
        return merged_config
    
    @staticmethod
    def create_rl_training_args(
        config: Dict[str, Any],
        training_mode: Optional[str] = None
    ) -> RLTrainingArguments:
        if training_mode is None:
            training_mode = config.get("rl", {}).get("training_mode", "dpo")
        
        rl_config = config.get("rl", {})
        train_config = config.get("train", {})
        common_config = config.get("common", {})
        generator_config = config.get("generator", {})
        base_dataset_name = common_config.get("dataset")
        
        args_dict = {
            "fp16": train_config.get("fp16", True),
            "bf16": train_config.get("bf16", False),
            "eval_steps": train_config.get("eval_steps", 50),
            "save_steps": train_config.get("save_steps", 100),
            "warmup_steps": train_config.get("warmup_steps", 100),
            "logging_steps": train_config.get("logging_steps", 1000),
            "learning_rate": train_config.get("learning_rate", 1e-5),
            "num_train_epochs": train_config.get("num_train_epochs", 1),
            "save_total_limit": train_config.get("save_total_limit", 1),
            "max_prompt_length": train_config.get("max_prompt_length", 512),
            "output_dir": train_config.get("output_dir", DEFAULT_OUTPUT_DIR),
            "checkpoint_dir": train_config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR),
            "max_completion_length": train_config.get("max_completion_length", 1024),
            "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps", 4),
            "load_in_8bit": common_config.get("load_in_8bit", False),
            "load_in_4bit": common_config.get("load_in_4bit", False),
            "temperature": generator_config.get("temperature", 0.3),
            "top_p": generator_config.get("top_p", 1.0),
            "num_train": config.get("num_train", None),
            "seed": common_config.get("seed", 42),
            "training_mode": training_mode,
            "ref_model_init_kwargs": {},
            "model_init_kwargs": {}
        }
        
        if training_mode == "dpo":
            dpo_config = rl_config.get("dpo", {})
            dataset_name = base_dataset_name if base_dataset_name else dpo_config["dataset"]
            field_mappings = dpo_config.get("field_mappings", {})
            args_dict.update({
                "dataset_name": dataset_name,
                "beta": dpo_config.get("beta", 0.1),
                "dpo_loss_type": dpo_config.get("loss_type", "sigmoid"),
                "per_device_train_batch_size": max(1, common_config.get("batch_size", 1)),
                "per_device_eval_batch_size": 1,
                "dpo_field_mappings": field_mappings
            })
        elif training_mode == "grpo":
            grpo_config = rl_config.get("grpo", {})
            dataset_name = base_dataset_name if base_dataset_name else grpo_config["dataset"]
            field_mappings = grpo_config.get("field_mappings", {})
            if not field_mappings:
                raise ValueError("GRPO config must contain 'field_mappings'")
            
            answer_fields = field_mappings.get("answer_fields", {})
            score_fields = field_mappings.get("score_fields", {})
            
            if not answer_fields or not score_fields:
                raise ValueError("GRPO field_mappings must contain 'answer_fields' and 'score_fields'")
            
            field_mappings = {
                "answer_fields": answer_fields,
                "score_fields": score_fields
            }
            base_batch_size = max(1, common_config.get("batch_size", 1))
            num_rollouts = grpo_config.get("num_rollouts", 4)
            if num_rollouts != answer_fields.get("count", 4):
                logger.warning(f"num_rollouts={num_rollouts}, answer count={answer_fields.get('count', 4)}. ")
            
            if base_batch_size % num_rollouts != 0:
                adjusted_batch_size = ((base_batch_size + num_rollouts - 1) // num_rollouts) * num_rollouts
                logger.warning(f"Adjusting batch_size from {base_batch_size} to {adjusted_batch_size} to be divisible by num_rollouts={num_rollouts}")
                batch_size = adjusted_batch_size
            else:
                batch_size = base_batch_size
            
            args_dict.update({
                "dataset_name": dataset_name,
                "beta": grpo_config.get("beta", 0.1),
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": 1,
                "grpo_field_mappings": field_mappings,
                "num_rollouts": num_rollouts
            })
        
        return RLTrainingArguments(**args_dict)

class BaseRLTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        args: RLTrainingArguments,
        train_dataset: Optional[HFDataset] = None,
        eval_dataset: Optional[HFDataset] = None
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        logger.info(f"Initialized {self.__class__.__name__} with {len(train_dataset) if train_dataset else 0} training examples")
    
    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        return {}
    
    def save_model(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        
        is_peft_model = False
        peft_config = None
        
        try:
            if isinstance(self.model, PeftModel):
                is_peft_model = True
                peft_config = self.model.peft_config
                logger.info("Detected PeftModel instance")
        except ImportError:
            logger.warning("PEFT library not available")
        
        if not is_peft_model and hasattr(self.model, 'peft_config'):
            peft_config = self.model.peft_config
            if peft_config is not None:
                is_peft_model = True
                logger.info("Detected model with peft_config attribute")
        
        if is_peft_model and peft_config:
            logger.info("Saving LoRA adapters separately")
            
            try:
                if isinstance(self.model, PeftModel):
                    self.model.save_pretrained(output_dir)
                    logger.info(f"PEFT adapters saved to {output_dir}")
                else:
                    logger.warning("Model has peft_config but isn't PeftModel instance")
                    logger.warning("Attempting manual adapter saving...")
                    
                    adapter_state_dict = get_peft_model_state_dict(self.model)
                    torch.save(adapter_state_dict, Path(output_dir) / "adapter_model.bin")
                    
                    config_dict = peft_config.to_dict()
                    with open(Path(output_dir) / "adapter_config.json", "w") as f:
                        json.dump(config_dict, f, indent=4)
                    
                    logger.info(f"Manual adapter saving completed")
            except Exception as e:
                logger.error(f"Error saving PEFT adapters: {e}")
                logger.warning("Falling back to standard model saving")
                self.model.save_pretrained(output_dir)
        else:
            logger.info("Saving standard model (full weights)")
            self.model.save_pretrained(output_dir)
        
        args_dict = asdict(self.args)
        del args_dict['checkpoint_dir']
        with open(Path(output_dir) / "training_args.json", "w") as f:
            json.dump(args_dict, f, indent=4, default=str)
        
        logger.info(f"Model saved to {output_dir}")

def preprocess_dpo_dataset(dataset, rl_training_args, logger):
    field_mappings = rl_training_args.dpo_field_mappings
    if field_mappings is None:
        raise ValueError("DPO field mappings must be provided")
    
    logger.info(f"DPO field mappings: {field_mappings}")
    
    required_columns = list(field_mappings.values())
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        raise ValueError(
            f"DPO dataset missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )
    
    rename_dict = {}
    for target_name, source_name in field_mappings.items():
        if source_name != target_name:
            rename_dict[source_name] = target_name
    
    if rename_dict:
        dataset = dataset.rename_columns(rename_dict)
        logger.info(f"Renamed dataset columns: {rename_dict}")
    
    def clean_example(example):
        cleaned = {}
        for key in ["prompt", "chosen", "rejected"]:
            value = example.get(key, "")
            if value is None:
                value = ""
            cleaned[key] = str(value).strip()
        return cleaned
    
    dataset = dataset.map(clean_example, batched=False, desc="Cleaning DPO dataset")
    
    initial_size = len(dataset)
    dataset = dataset.filter(
        lambda x: (
            x["prompt"] and
            x["chosen"] and
            x["rejected"] and
            x["chosen"] != x["rejected"]
        )
    )
    filtered_size = len(dataset)
    
    if filtered_size < initial_size:
        logger.warning(
            f"Filtered out {initial_size - filtered_size} examples with empty or identical responses"
        )
    
    if rl_training_args.num_train is not None and rl_training_args.num_train > 0:
        limit = min(rl_training_args.num_train, len(dataset))
        dataset = dataset.select(range(limit))
        logger.info(f"Limited dataset to {limit} examples (requested: {rl_training_args.num_train})")
    
    logger.info(f"Processed DPO dataset with {len(dataset)} examples")
    if len(dataset) > 0:
        logger.info(f"Sample prompt: {dataset[0]['prompt'][:100]}...")
        logger.info(f"Sample chosen (first 100 chars): {dataset[0]['chosen'][:100]}...")
        logger.info(f"Sample rejected (first 100 chars): {dataset[0]['rejected'][:100]}...")
    
    return dataset

class RLDPOTrainer(BaseRLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        ref_model = copy.deepcopy(self.model)
        for param in ref_model.parameters():
            param.requires_grad = False
        
        dpo_config = DPOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_steps=self.args.eval_steps,
            save_total_limit=self.args.save_total_limit,
            warmup_steps=self.args.warmup_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            seed=self.args.seed,
            beta=self.args.beta,
            loss_type=self.args.dpo_loss_type,
            model_init_kwargs=self.args.model_init_kwargs,
            ref_model_init_kwargs=self.args.ref_model_init_kwargs,
            max_length=self.args.max_prompt_length + self.args.max_completion_length,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_completion_length
        )
        
        self.trltrainer = DPOTrainer(
            model=self.model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        logger.info(f"RLDPOTrainer initialized with beta={self.args.beta}, loss_type={self.args.dpo_loss_type}")
    
    def train(self):
        try:
            train_result = self.trltrainer.train()
            logger.info("DPO training completed successfully")
            
            if self.args.checkpoint_dir:
                self.save_model(self.args.checkpoint_dir)
            
            return {
                "train_metrics": train_result.metrics,
                "output_dir": self.args.checkpoint_dir
            }
        except Exception as e:
            logger.error(f"Error during DPO training: {e}")
            raise

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    
    s = s.lower().strip()
    
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    
    return s

def calculate_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def preprocess_grpo_dataset(dataset, rl_training_args, logger):
    field_mappings = rl_training_args.grpo_field_mappings
    answer_fields = field_mappings.get("answer_fields", {})
    score_fields = field_mappings.get("score_fields", {})

    answer_prefix = answer_fields.get("prefix", "A")
    score_prefix = score_fields.get("prefix", "score_A")
    count = answer_fields.get("count", 4)
    
    num_train_limit = getattr(rl_training_args, "num_train", None)
    if num_train_limit is not None and num_train_limit > 0:
        logger.info(f"Limiting dataset to first {num_train_limit} valid examples.")
    
    processed = []
    dropped_zero_reward = 0
    dropped_empty = 0
    
    logger.info("Starting GRPO dataset preprocessing...")

    for i, example in enumerate(dataset):
        if num_train_limit is not None and len(processed) >= num_train_limit:
            logger.info(f"Reached num_train limit ({num_train_limit}). Stopping preprocessing.")
            break

        prompt = str(example.get("prompt", "")).strip()
        if not prompt:
            dropped_empty += 1
            continue

        answers = []
        scores = []

        for k in range(count):
            a_col = f"{answer_prefix}{k}"
            s_col = f"{score_prefix}{k}"
            
            if a_col in example and s_col in example:
                raw_ans = example[a_col]
                raw_score = example[s_col]
                
                if raw_ans is not None and str(raw_ans).strip():
                    try:
                        scr = float(raw_score)
                    except (ValueError, TypeError):
                        scr = 0.0
                    
                    answers.append(str(raw_ans).strip())
                    scores.append(scr)

        if not scores or max(scores) <= 0.0:
            dropped_zero_reward += 1
            continue

        processed.append({
            "prompt": prompt,
            "answers": answers,
            "scores": scores
        })

    logger.info(f"Dataset Preprocessing Complete:")
    logger.info(f"  - Final Size: {len(processed)}")
    logger.info(f"  - Dropped (Empty/No Prompt): {dropped_empty}")
    logger.info(f"  - Dropped (Max Score <= 0): {dropped_zero_reward}")
    
    return HFDataset.from_list(processed)

class RLGRPOTrainer(BaseRLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_to_reference = {}
        
        if self.train_dataset is not None:
            for ex in self.train_dataset:
                p = ex.get("prompt", "").strip()
                if p:
                    self.prompt_to_reference[p] = {
                        "answers": ex.get("answers", []),
                        "scores": ex.get("scores", [])
                    }
            logger.info(f"Built reference map for {len(self.prompt_to_reference)} unique prompts")
        else:
            logger.warning("train_dataset is None! Reference map is empty.")

        def soft_match_reward_func(prompts, completions, **kwargs):
            rewards = []
            
            for prompt, completion in zip(prompts, completions):
                prompt_key = prompt.strip()
                best_reward = 0.0
                
                if prompt_key in self.prompt_to_reference:
                    ref = self.prompt_to_reference[prompt_key]
                    
                    completion_norm = normalize_text(completion)
                    
                    for ref_ans, ref_score in zip(ref["answers"], ref["scores"]):
                        ref_score = float(ref_score)
                        
                        if ref_score <= 0:
                            continue

                        ref_ans_norm = normalize_text(ref_ans)
                        current_match_score = 0.0

                        if ref_ans_norm in completion_norm:
                            current_match_score = ref_score
                        else:
                            similarity = calculate_similarity(ref_ans_norm, completion_norm)
                            
                            if similarity > 0.6:
                                current_match_score = similarity * ref_score
                        
                        best_reward = max(best_reward, current_match_score)
                
                rewards.append(best_reward)
            
            return rewards
        
        grpo_config = GRPOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_completion_length,
            num_generations=self.args.num_rollouts, 
            beta=self.args.beta,
            seed=self.args.seed,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
        )

        self.trltrainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            reward_funcs=soft_match_reward_func,
            processing_class=self.tokenizer
        )

        logger.info(f"Initialized RLGRPOTrainer with {self.args.num_rollouts} rollouts")

    def train(self):
        logger.info("Starting Soft-Match GRPO training...")
        train_result = self.trltrainer.train()
        
        if self.args.checkpoint_dir:
            self.save_model(self.args.checkpoint_dir)
            
        return {
            "train_metrics": train_result.metrics,
            "output_dir": self.args.checkpoint_dir
        }

def create_rl_trainer_from_config(
    config: Optional[Dict[str, Any]] = None,
    model_args: Optional[Any] = None,
    training_args: Optional[Any] = None
) -> Union[RLDPOTrainer, RLGRPOTrainer]:
    training_mode = config["rl"]["training_mode"]
    config = RLConfigManager.load_merged_config(base_config=config)
    logger.info(f"Creating {training_mode.upper()} trainer from configuration")
    rl_training_args = RLConfigManager.create_rl_training_args(config, training_mode)
    
    if training_args:
        for field in asdict(training_args):
            if hasattr(rl_training_args, field):
                value = getattr(training_args, field)
                if value is not None:
                    setattr(rl_training_args, field, value)
    
    common_config = config.get("common", {})
    
    if model_args and hasattr(model_args, 'model'):
        model_path = model_args.model
    else:
        model_path = common_config["model"]
    
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    load_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch.float16 if rl_training_args.fp16 else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True
    }
    
    if rl_training_args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        logger.info("Loading model in 8-bit precision")
    elif rl_training_args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        logger.info("Loading model in 4-bit precision")
    
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    
    if common_config.get("parameter_efficient_mode", "none") != "none":
        lora_config = common_config.get("lora_config", {})
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get("dropout", 0.1),
            bias=lora_config.get("bias", "none") if training_mode != 'grpo' else "none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Applied LoRA: {trainable_params:,} trainable params ({100*trainable_params/total_params:.2f}%)")
    
    dataset_name = rl_training_args.dataset_name
    
    if training_mode == "dpo":
        logger.info(f"Loading DPO dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=rl_training_args.dataset_split)
        train_dataset = preprocess_dpo_dataset(dataset, rl_training_args, logger)
        
        trainer = RLDPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=rl_training_args,
            train_dataset=train_dataset
        )
    elif training_mode == "grpo":
        logger.info(f"Loading GRPO dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=rl_training_args.dataset_split)
        train_dataset = preprocess_grpo_dataset(dataset, rl_training_args, logger)
        
        trainer = RLGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=rl_training_args,
            train_dataset=train_dataset
        )
    else:
        logger.error(f"Unsupported training mode {training_mode}")
        raise ValueError(f"Unsupported training mode: {training_mode}")
    
    logger.info(f"{training_mode.upper()} trainer created successfully")
    return trainer
