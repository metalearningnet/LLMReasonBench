import torch
import inspect
import warnings
from torch import nn
from config import logger
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union, Dict, Any
from transformers.generation.utils import GenerationMixin
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

class SparseGenerationMixin(GenerationMixin):
    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        encoder = self.get_encoder()

        irrelevant_prefix = [
            "decoder_",
            "cross_attn",
            "use_cache",
            "cross_attention_mask"
        ]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = (
            "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        )
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature
            }

        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        using_past_key_values = "past_key_values" in model_kwargs
        if using_past_key_values:
            warnings.warn(
                "past_key_values passed to encoder. "
                "This should only happen when reusing gist tokens."
            )
        encoder_outputs: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = encoder_outputs

        if using_past_key_values:
            del model_kwargs["past_key_values"]

        return model_kwargs

class SparseAttentionMixin:
    @staticmethod
    def prepare_decoder_attention_mask(
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int,
        model_type: str,
        sparse_mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Prepare decoder attention mask with optional sparse mask.
        
        For LLaMA: Creates causal mask with optional padding mask
        For GPT-2: Creates padding mask (causal mask is handled internally via bias)
        
        Args:
            attention_mask: Standard HF attention mask of shape (batch_size, seq_len)
            sparse_mask: Optional sparse attention mask of shape
                (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len, seq_len)
            input_shape: (batch_size, seq_len)
            dtype: Data type for the attention mask
            device: Device for the attention mask
            past_key_values_length: Length of past key values
            model_type: 'llama' or 'gpt2'
            
        Returns:
            Combined attention mask of shape (batch_size, 1, seq_len, seq_len + past_key_values_length)
            or None if no mask is needed
        """
        batch_size, seq_len = input_shape
        src_len = seq_len + past_key_values_length
        
        combined_mask = None
        
        # Handle LLaMA: Create causal mask with optional padding mask
        if model_type == "llama":
            combined_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=input_shape,
                inputs_embeds=None,
                past_key_values_length=past_key_values_length,
                dtype=dtype,
                device=device
            )
        # For GPT-2, we don't create causal mask here (handled internally)
        
        # Handle standard padding mask for GPT-2 (if provided)
        if attention_mask is not None and model_type != "llama":
            # GPT-2 only needs padding mask, causal is handled via bias
            if attention_mask.dim() != 2:
                raise ValueError(f"Standard mask should be 2D, got shape {attention_mask.shape}")
            
            if attention_mask.dtype not in [torch.bool, torch.long, torch.int]:
                attention_mask = attention_mask.to(torch.long)
            
            # Extend attention mask for past positions
            if past_key_values_length > 0:
                # For past tokens, assume they are valid (not padding)
                past_mask = torch.ones(
                    (batch_size, past_key_values_length),
                    device=device,
                    dtype=attention_mask.dtype
                )
                extended_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
            else:
                extended_attention_mask = attention_mask
            
            # Create 4D padding mask (batch_size, 1, tgt_len, src_len)
            padding_mask = _prepare_4d_attention_mask(
                extended_attention_mask,
                dtype=dtype,
                tgt_len=seq_len
            ).to(device)
            
            if combined_mask is None:
                combined_mask = padding_mask
            else:
                combined_mask = combined_mask + padding_mask
        
        if sparse_mask is not None:
            # Convert sparse_mask to boolean
            if sparse_mask.dtype != torch.bool:
                sparse_mask = sparse_mask.bool()
            
            # Validate sparse mask shape
            if sparse_mask.dim() == 3:
                # (batch_size, seq_len, seq_len)
                if sparse_mask.shape[-2:] != (seq_len, seq_len):
                    raise ValueError(
                        f"Sparse mask last two dimensions should be ({seq_len}, {seq_len}), "
                        f"got {sparse_mask.shape[-2:]}"
                    )
                sparse_mask = sparse_mask.unsqueeze(1) # Add head dimension
            elif sparse_mask.dim() == 4:
                # (batch_size, num_heads, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
                if sparse_mask.shape[-2:] != (seq_len, seq_len):
                    raise ValueError(
                        f"Sparse mask last two dimensions should be ({seq_len}, {seq_len}), "
                        f"got {sparse_mask.shape[-2:]}"
                    )
                # If sparse_mask has multiple heads but model expects 1, take first head
                if sparse_mask.shape[1] != 1:
                    # For compatibility with models expecting 1 head mask
                    sparse_mask = sparse_mask[:, :1, :, :]
            else:
                raise ValueError(
                    f"Sparse mask should be 3D or 4D, got shape {sparse_mask.shape}"
                )
            
            if combined_mask is None:
                combined_mask = torch.zeros(
                    (batch_size, 1, seq_len, src_len),
                    device=device,
                    dtype=dtype
                )
            
            # Extend sparse mask for past positions
            if past_key_values_length > 0:
                # Create extended sparse mask
                extended_sparse = torch.ones(
                    (batch_size, 1, seq_len, src_len),
                    device=device,
                    dtype=torch.bool
                )
                extended_sparse[..., past_key_values_length:] = sparse_mask
                sparse_mask = extended_sparse
            
            # Get safe negative value for the dtype
            if dtype in [torch.float16, torch.bfloat16]:
                large_negative = torch.tensor(-10000.0, device=device, dtype=dtype)
            else:
                large_negative = torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype)
            
            # Create float sparse mask: 0.0 where allowed, large_negative where blocked
            sparse_float_mask = torch.full(
                (batch_size, 1, seq_len, src_len),
                large_negative,
                device=device,
                dtype=dtype
            )
            
            sparse_float_mask = sparse_float_mask.masked_fill(
                sparse_mask,
                0.0
            )
            
            combined_mask = combined_mask + sparse_float_mask
        
        return combined_mask

class SparseLlamaModel(LlamaModel, SparseAttentionMixin):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.num_attention_heads = config.num_attention_heads
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sparse_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                device=input_ids.device if input_ids is not None else inputs_embeds.device,
                dtype=torch.long
            )
        
        # Prepare combined attention mask with sparse mask
        if attention_mask is not None or sparse_mask is not None:
            attention_mask = self.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                sparse_mask=sparse_mask,
                input_shape=(batch_size, seq_length),
                dtype=self.dtype,
                device=input_ids.device if input_ids is not None else inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                model_type='llama',
                num_attention_heads=self.num_attention_heads
            )
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class SparseLlamaForCausalLM(LlamaForCausalLM, SparseGenerationMixin):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = SparseLlamaModel(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sparse_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        # Forward through model with sparse_mask
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sparse_mask=sparse_mask
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # Handle position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids from attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            
            if past_key_values is not None:
                position_ids = position_ids[:, -1:] # Only last position for generation
        
        sparse_mask = kwargs.get("sparse_mask", None)
        
        model_inputs = {"input_ids": input_ids}
        
        if sparse_mask is not None:
            model_inputs["sparse_mask"] = sparse_mask
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", self.config.use_cache),
            "attention_mask": attention_mask
        })
        
        return model_inputs

class SparseGPT2Model(GPT2Model, SparseAttentionMixin):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.num_attention_heads = config.num_attention_heads
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sparse_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_shape[0]
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_hidden_shape = encoder_hidden_states.size()[:-1]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        
        # Prepare attention mask for GPT-2
        # GPT-2 handles causality internally, so we only prepare padding + sparse mask
        if attention_mask is not None or sparse_mask is not None:
            attention_mask = self.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                sparse_mask=sparse_mask,
                input_shape=input_shape,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_length,
                model_type='gpt2',
                num_attention_heads=self.num_attention_heads
            )
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.config.add_cross_attention) else None
        all_hidden_states = () if output_hidden_states else None
        
        for layer_idx, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
                
                for device_id, layers in self.device_map.items():
                    if layer_idx == layers[-1] and f"cuda:{device_id}" != self.last_device:
                        next_device = f"cuda:{device_id + 1}"
                        hidden_states = hidden_states.to(next_device)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)
                    return custom_forward
                
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[layer_idx],
                    encoder_hidden_states,
                    encoder_attention_mask
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[layer_idx],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
            
            hidden_states = outputs[0]
            
            if use_cache:
                presents = presents + (outputs[1],)
            
            if output_attentions:
                idx = 2 if use_cache else 1
                all_self_attentions = all_self_attentions + (outputs[idx],)
                
                if self.config.add_cross_attention:
                    idx = 3 if use_cache else 2
                    all_cross_attentions = all_cross_attentions + (outputs[idx],)
        
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states, presents, all_hidden_states,
                    all_self_attentions, all_cross_attentions
                ] if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions
        )

class SparseGPT2LMHeadModel(GPT2LMHeadModel, SparseGenerationMixin):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = SparseGPT2Model(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sparse_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        # Forward through transformer with sparse_mask
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sparse_mask=sparse_mask
        )
        
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": transformer_outputs.past_key_values if hasattr(transformer_outputs, 'past_key_values') else None,
            "hidden_states": transformer_outputs.hidden_states if hasattr(transformer_outputs, 'hidden_states') else None,
            "attentions": transformer_outputs.attentions if hasattr(transformer_outputs, 'attentions') else None,
            "cross_attentions": transformer_outputs.cross_attentions if hasattr(transformer_outputs, 'cross_attentions') else None
        }
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        token_type_ids = kwargs.get("token_type_ids", None)
        
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]
        
        sparse_mask = kwargs.get("sparse_mask", None)
        
        model_inputs = {"input_ids": input_ids}
        
        if sparse_mask is not None:
            model_inputs["sparse_mask"] = sparse_mask
        
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", self.config.use_cache),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })
        
        return model_inputs
