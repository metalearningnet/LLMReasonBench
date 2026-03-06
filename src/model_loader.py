import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel
from sparse_model import SparseGPT2LMHeadModel, SparseLlamaForCausalLM
from typing import Optional, Union
from config import logger

class InputEmbedding(nn.Module):
    def __init__(
        self,
        original_embedding: nn.Embedding,
        n_new_tokens: int,
        initialize_tokens: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.original_embedding = original_embedding
        self.num_original_tokens = original_embedding.weight.size(0)
        self.n_new_tokens = n_new_tokens
        
        logger.info(f"Original vocab size: {self.num_original_tokens}")
        
        if n_new_tokens > 0:
            embedding_dim = original_embedding.weight.size(1)
            device = original_embedding.weight.device
            target_dtype = original_embedding.weight.dtype
            
            self.new_embedding = nn.Embedding(n_new_tokens, embedding_dim, dtype=target_dtype).to(device)
            
            if initialize_tokens is not None:
                logger.debug(f"Initializing {n_new_tokens} new tokens from provided tokens")
                with torch.no_grad():
                    new_embeddings = self.original_embedding(initialize_tokens)
                    self.new_embedding.weight.data.copy_(new_embeddings)
            else:
                logger.debug(f"Initializing {n_new_tokens} new tokens with mean embedding")
                with torch.no_grad():
                    mean_embedding = original_embedding.weight.mean(dim=0, keepdim=True)
                    self.new_embedding.weight.data.copy_(mean_embedding.repeat(n_new_tokens, 1))
        else:
            self.new_embedding = None
        
        self._update_combined_weight()
    
    def _update_combined_weight(self):
        with torch.no_grad():
            if self.n_new_tokens > 0 and self.new_embedding is not None:
                original_weight = self.original_embedding.weight.data
                new_weight = self.new_embedding.weight.data
                self.combined_weight = torch.cat([original_weight, new_weight], dim=0)
            else:
                self.combined_weight = self.original_embedding.weight.data
    
    @property
    def weight(self):
        return self.combined_weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.n_new_tokens == 0 or input_ids.max() < self.num_original_tokens:
            return self.original_embedding(input_ids)
        
        new_token_mask = input_ids >= self.num_original_tokens
        original_token_mask = ~new_token_mask
        
        if new_token_mask.any():
            new_indices = input_ids[new_token_mask] - self.num_original_tokens
            new_embeddings = self.new_embedding(new_indices)
        
        if original_token_mask.any():
            original_embeddings = self.original_embedding(input_ids[original_token_mask])
        
        embedding_dim = self.original_embedding.weight.size(1)
        batch_size, seq_len = input_ids.shape
        
        if original_token_mask.any():
            target_dtype = original_embeddings.dtype
        elif new_token_mask.any():
            target_dtype = new_embeddings.dtype
        else:
            target_dtype = torch.float16
        
        result = torch.zeros(
            (batch_size, seq_len, embedding_dim),
            dtype=target_dtype,
            device=input_ids.device
        )
        
        if new_token_mask.any():
            result[new_token_mask] = new_embeddings.to(target_dtype)
        if original_token_mask.any():
            result[original_token_mask] = original_embeddings.to(target_dtype)
        
        return result

class OutputEmbedding(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        n_new_tokens: int,
        initialize_tokens: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.original_linear = original_linear
        self.n_new_tokens = n_new_tokens
        
        if n_new_tokens > 0:
            hidden_dim = original_linear.weight.size(1)
            device = original_linear.weight.device
            target_dtype = original_linear.weight.dtype
            
            self.new_linear = nn.Linear(hidden_dim, n_new_tokens, dtype=target_dtype).to(device)
            
            if initialize_tokens is not None:
                logger.debug(f"Initializing output layer for {n_new_tokens} new tokens from provided tokens")
                with torch.no_grad():
                    new_weights = F.embedding(initialize_tokens, original_linear.weight.data)
                    self.new_linear.weight.data.copy_(new_weights)
            else:
                logger.debug(f"Initializing output layer for {n_new_tokens} new tokens with mean weights")
                with torch.no_grad():
                    mean_weight = original_linear.weight.mean(dim=0, keepdim=True)
                    self.new_linear.weight.data.copy_(mean_weight.repeat(n_new_tokens, 1))
        else:
            self.new_linear = None
        
        self._update_combined_weights()
    
    def _update_combined_weights(self):
        with torch.no_grad():
            if self.n_new_tokens > 0 and self.new_linear is not None:
                original_weight = self.original_linear.weight.data
                new_weight = self.new_linear.weight.data
                self.combined_weight = torch.cat([original_weight, new_weight], dim=0)
                
                if self.original_linear.bias is not None:
                    original_bias = self.original_linear.bias.data
                    if self.new_linear.bias is not None:
                        new_bias = self.new_linear.bias.data
                        self.combined_bias = torch.cat([original_bias, new_bias], dim=0)
                    else:
                        new_bias = torch.zeros(self.n_new_tokens, device=original_bias.device)
                        self.combined_bias = torch.cat([original_bias, new_bias], dim=0)
                else:
                    self.combined_bias = None
            else:
                self.combined_weight = self.original_linear.weight.data
                self.combined_bias = self.original_linear.bias.data if self.original_linear.bias is not None else None
    
    @property
    def weight(self):
        return self.combined_weight
    
    @property
    def bias(self):
        return self.combined_bias
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_logits = self.original_linear(inputs)
        
        if self.n_new_tokens > 0 and self.new_linear is not None:
            new_logits = self.new_linear(inputs)
            return torch.cat([original_logits, new_logits], dim=-1)
        
        return original_logits

def load_embeddings(
    model: nn.Module,
    input_embedding_file: str,
    output_embedding_file: Optional[str],
    n_tokens: int,
    orig_vocab_size: int
):
    def _safe_load_embedding(file_path: str, embedding_type: str):
        if not os.path.isfile(file_path):
            error_msg = f"{embedding_type} embedding file not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        safe_globals = [
            torch.nn.modules.sparse.Embedding,
            torch.nn.Linear,
            torch.nn.parameter.Parameter,
            torch._utils._rebuild_tensor_v2,
            torch.storage._load_from_bytes,
            dict,
            list,
            tuple
        ]
        
        try:
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(safe_globals)
            
            data = torch.load(file_path, weights_only=True)
            logger.debug(f"Loaded {embedding_type} embeddings with weights_only=True")
            return data
        except Exception as e:
            logger.warning(f"Safe loading failed for {embedding_type} embeddings: {e}")
            logger.warning("Attempting fallback loading (use only if you trust the source)")
            
            try:
                data = torch.load(file_path, weights_only=False)
                logger.warning(f"Loaded {embedding_type} embeddings with weights_only=False (backward compatibility mode)")
                return data
            except Exception as e2:
                error_msg = f"Failed to load {embedding_type} embeddings: {e2}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    logger.info(f"Loading input embeddings from: {input_embedding_file}")
    input_data = _safe_load_embedding(input_embedding_file, "input")
    
    input_weight = None
    if isinstance(input_data, torch.Tensor):
        input_weight = input_data
        logger.info(f"Loaded input embedding tensor with shape: {input_weight.shape}")
    elif hasattr(input_data, 'weight'):
        input_weight = input_data.weight.data
        logger.warning(f"Loaded input embedding module, extracting weight with shape: {input_weight.shape}")
        logger.warning("Consider resaving with new format for better security")
    else:
        error_msg = f"Unsupported input embedding format: {type(input_data)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if input_weight.dim() != 2:
        error_msg = f"Input embedding must be 2D tensor, got shape: {input_weight.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    n_loaded_tokens = input_weight.size(0)
    target_dtype = model.get_input_embeddings().weight.dtype
    
    if n_loaded_tokens == n_tokens + orig_vocab_size:
        logger.warning("Replacing entire input embedding layer with loaded weights")
        full_embedding = nn.Embedding.from_pretrained(input_weight.to(target_dtype))
        model.set_input_embeddings(full_embedding)
    elif n_loaded_tokens == n_tokens:
        logger.warning(f"Adding {n_tokens} new tokens to existing input embeddings")
        current_embeddings = model.get_input_embeddings()
        
        if not isinstance(current_embeddings, InputEmbedding):
            model.set_input_embeddings(InputEmbedding(current_embeddings, n_tokens))
        
        embedding_dim = input_weight.size(1)
        new_embedding = nn.Embedding(n_tokens, embedding_dim, dtype=target_dtype)
        new_embedding.weight.data.copy_(input_weight.to(target_dtype))
        model.get_input_embeddings().new_embedding = new_embedding
    else:
        error_msg = (
            f"Input embedding size mismatch: loaded {n_loaded_tokens} tokens, "
            f"expected {n_tokens} (new tokens) or {n_tokens + orig_vocab_size} (full vocab)"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Input embeddings loaded successfully")
    
    if output_embedding_file is not None:
        logger.info(f"Loading output embeddings from: {output_embedding_file}")
        output_data = _safe_load_embedding(output_embedding_file, "output")
        
        output_weight = None
        output_bias = None
        output_target_dtype = model.get_output_embeddings().weight.dtype
        
        if isinstance(output_data, dict) and 'weight' in output_data:
            output_weight = output_data['weight']
            output_bias = output_data.get('bias', None)
            logger.info(f"Loaded output weight tensor with shape: {output_weight.shape}")
            if output_bias is not None:
                logger.info(f"Loaded output bias tensor with shape: {output_bias.shape}")
        elif hasattr(output_data, 'weight'):
            output_weight = output_data.weight.data
            output_bias = output_data.bias.data if output_data.bias is not None else None
            logger.warning(f"Loaded output linear module, extracting weight with shape: {output_weight.shape}")
            logger.warning("Consider resaving with new format for better security")
        elif isinstance(output_data, torch.Tensor):
            output_weight = output_data
            output_bias = None
            logger.info(f"Loaded output weight tensor with shape: {output_weight.shape}")
        else:
            error_msg = f"Unsupported output embedding format: {type(output_data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if output_weight.dim() != 2:
            error_msg = f"Output weight must be 2D tensor, got shape: {output_weight.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        n_loaded_output_tokens = output_weight.size(0)
        
        if n_loaded_output_tokens == n_tokens + orig_vocab_size:
            logger.debug("Replacing entire output embedding layer with loaded weights")
            hidden_dim = output_weight.size(1)
            new_head = nn.Linear(hidden_dim, n_loaded_output_tokens, dtype=output_target_dtype)
            new_head.weight.data.copy_(output_weight.to(output_target_dtype))
            if output_bias is not None:
                new_head.bias.data.copy_(output_bias.to(output_target_dtype) if torch.is_tensor(output_bias) else output_bias)
            model.set_output_embeddings(new_head)
        elif n_loaded_output_tokens == n_tokens:
            logger.debug(f"Adding {n_tokens} new tokens to existing output embeddings")
            current_head = model.get_output_embeddings()
            
            if not isinstance(current_head, OutputEmbedding):
                model.set_output_embeddings(OutputEmbedding(current_head, n_tokens))
            
            hidden_dim = output_weight.size(1)
            new_linear = nn.Linear(hidden_dim, n_tokens, dtype=output_target_dtype)
            new_linear.weight.data.copy_(output_weight.to(output_target_dtype))
            if output_bias is not None:
                new_linear.bias.data.copy_(output_bias.to(output_target_dtype) if torch.is_tensor(output_bias) else output_bias)
            model.get_output_embeddings().new_linear = new_linear
        else:
            error_msg = (
                f"Output embedding size mismatch: loaded {n_loaded_output_tokens} tokens, "
                f"expected {n_tokens} (new tokens) or {n_tokens + orig_vocab_size} (full vocab)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Output embeddings loaded successfully")
    else:
        logger.debug("No output embeddings provided, using weight tying")
        if hasattr(model, 'tie_weights'):
            model.tie_weights()

def _create_sparse_model(model_name: str, **kwargs) -> nn.Module:
    model_name_lower = model_name.lower()
    
    if 'llama' in model_name_lower or 'alpaca' in model_name_lower:
        logger.info(f"Creating sparse model for: {model_name} (using SparseLlamaForCausalLM)")
        return SparseLlamaForCausalLM.from_pretrained(**kwargs)
    elif 'gpt2' in model_name_lower:
        logger.info(f"Creating sparse model for: {model_name} (using SparseGPT2LMHeadModel)")
        return SparseGPT2LMHeadModel.from_pretrained(**kwargs)
    else:
        error_msg = (
            f"Sparse model not implemented for: {model_name}. "
            "Supported models: llama, alpaca, gpt2"
        )
        logger.error(error_msg)
        raise NotImplementedError(error_msg)

class CausalLM(AutoModelForCausalLM):
    def __init__(
        self,
        n_tokens: int = 0,
        sparse: bool = False,
        parameter_efficient_mode: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.sparse = sparse
        self.parameter_efficient_mode = parameter_efficient_mode
    
    @classmethod
    def from_pretrained(
        cls,
        n_tokens: int = 0,
        input_embedding_file: Optional[str] = None,
        output_embedding_file: Optional[str] = None,
        sparse: bool = False,
        parameter_efficient_mode: str = 'none',
        initialize_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> 'CausalLM':
        if parameter_efficient_mode not in ["none", "lora", "lora-tag", "lora-tag-tuning"]:
            error_msg = (
                f"Invalid parameter_efficient_mode: {parameter_efficient_mode}. "
                "Valid options: none, lora, lora-tag, lora-tag-tuning"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Loading model with parameters: n_tokens={n_tokens}, "
                   f"sparse={sparse}, parameter_efficient_mode={parameter_efficient_mode}")
        
        if sparse:
            model = _create_sparse_model(kwargs['pretrained_model_name_or_path'], **kwargs)
        else:
            logger.debug(f"Loading standard model: {kwargs['pretrained_model_name_or_path']}")
            model = AutoModelForCausalLM.from_pretrained(**kwargs, trust_remote_code=True)
        
        model.n_tokens = n_tokens
        model.sparse = sparse
        model.parameter_efficient_mode = parameter_efficient_mode
        
        if n_tokens > 0:
            orig_vocab_size = model.get_input_embeddings().weight.size(0)
            logger.info(f"Original vocab size: {orig_vocab_size}, adding {n_tokens} new tokens")
            
            if initialize_tokens is not None:
                initialize_tokens = torch.as_tensor(
                    initialize_tokens,
                    dtype=torch.long,
                    device=model.device
                )
                logger.debug(f"Using provided initialization tokens of shape: {initialize_tokens.shape}")
            
            if parameter_efficient_mode != 'none':
                new_vocab_size = orig_vocab_size + n_tokens
                
                if hasattr(model.config, "text_config"):
                    logger.info(f"Updating text_config.vocab_size to {new_vocab_size}")
                    model.config.text_config.vocab_size = new_vocab_size
                
                if not hasattr(model.config, "vocab_size") or model.config.vocab_size != new_vocab_size:
                    logger.info(f"Forcing top-level vocab_size to {new_vocab_size}")
                    setattr(model.config, "vocab_size", new_vocab_size)
                
                logger.debug(f"Setting vocab size to: {new_vocab_size}")

                if hasattr(model, 'vocab_size'):
                    model.vocab_size = new_vocab_size
                    logger.debug(f"Set model.vocab_size to {new_vocab_size}")
                
                if input_embedding_file is not None:
                    logger.info("Loading embeddings from files")
                    load_embeddings(
                        model, input_embedding_file, output_embedding_file,
                        n_tokens, orig_vocab_size
                    )
                else:
                    logger.info("Creating new embeddings for parameter efficient mode")
                    model.set_input_embeddings(
                        InputEmbedding(
                            model.get_input_embeddings(),
                            n_tokens,
                            initialize_tokens
                        )
                    )
                    model.set_output_embeddings(
                        OutputEmbedding(
                            model.get_output_embeddings(),
                            n_tokens,
                            initialize_tokens
                        )
                    )
            elif initialize_tokens is not None:
                logger.info("Resizing token embeddings in standard mode")
                model.resize_token_embeddings(orig_vocab_size + n_tokens)
                new_vocab_size = model.get_input_embeddings().weight.size(0)
                
                if new_vocab_size != n_tokens + orig_vocab_size:
                    error_msg = (
                        f"Resized vocab size mismatch: {new_vocab_size} != {n_tokens + orig_vocab_size}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.debug("Initializing new token embeddings")
                with torch.no_grad():
                    new_input_embeddings = model.get_input_embeddings()(initialize_tokens)
                    model.get_input_embeddings().weight.data[-n_tokens:] = new_input_embeddings
                    
                    new_output_embeddings = F.embedding(
                        initialize_tokens,
                        model.get_output_embeddings().weight.data
                    )
                    model.get_output_embeddings().weight.data[-n_tokens:] = new_output_embeddings
        
        logger.info(f"Model loaded successfully: {model.__class__.__name__}")
        return model

def _save_pretrained_monkey_patch(
    self,
    save_directory: Union[str, os.PathLike],
    **kwargs
):
    if os.path.isfile(save_directory):
        error_msg = f"Provided path ({save_directory}) should be a directory, not a file"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    os.makedirs(save_directory, exist_ok=True)
    logger.info(f"Saving model to: {save_directory}")
    
    input_embeddings = self.get_input_embeddings()
    if hasattr(input_embeddings, 'new_embedding') and input_embeddings.new_embedding is not None:
        input_path = os.path.join(save_directory, "input_embeddings.pt")
        torch.save(input_embeddings.new_embedding.weight.data, input_path)
        logger.debug(f"Saved input embedding weights to: {input_path}")
    
    output_embeddings = self.get_output_embeddings()
    if hasattr(output_embeddings, 'new_linear') and output_embeddings.new_linear is not None:
        output_path = os.path.join(save_directory, "output_embeddings.pt")
        linear_state = {
            'weight': output_embeddings.new_linear.weight.data,
            'bias': output_embeddings.new_linear.bias.data if output_embeddings.new_linear.bias is not None else None
        }
        torch.save(linear_state, output_path)
        logger.debug(f"Saved output linear weights to: {output_path}")
    
    original_save_pretrained = PreTrainedModel.save_pretrained.__wrapped__ if hasattr(PreTrainedModel.save_pretrained, '__wrapped__') else None
    if original_save_pretrained:
        original_save_pretrained(self, save_directory, **kwargs)
    else:
        super(PreTrainedModel, self).save_pretrained(save_directory, **kwargs)

PreTrainedModel.save_pretrained = _save_pretrained_monkey_patch
