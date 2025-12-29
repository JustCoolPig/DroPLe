import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, LoraModel
from typing import Any, Optional, Tuple, Union
import numpy as np
import math
import time
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import copy
import sys

sys.path.append("../")
from utils.residual_entropy_loss import Demix, get_annealing_down_params
from utils.multimodal_alignment import MultimodalAlignmentGuidance
from utils.token_importance import (
    calculate_token_importance_score,
    extract_cls_attention,
    apply_importance_based_dropout,
)
# Create causal attention mask, which is crucial in autoregressive language models
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    # Create a tgt_len × tgt_len matrix filled with the minimum possible value
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # Create a lower triangular matrix with zeros on and below the diagonal, keeping minimum values in the upper triangle.
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Expand the 2D attention mask to 4D, suitable for multi-head attention mechanism.
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class DroPLe(nn.Module):

    def __init__(self, dset, classnames, custom_template, args, model, tokenizer, few_shot=False, indices=None):
        super(DroPLe, self).__init__()
        self.args = args
        self.dset = dset

        self.token_modification_method = args.token_modification_method
        self.noise_injection_sigma = args.noise_injection_sigma
        self.noise_max_prob = args.noise_max_prob
        self.noise_min_prob = args.noise_min_prob

        self.adapter = Adapter(512, 4)

        self.naive_decoding = args.naive_decoding
        self.debug = args.debug

        self.ema_decay = 0.999
        self.ema_params = None
        self.steps = 0

        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["layers.{}.self_attn.q_proj".format(i) for i in
                            range(args.v_lora_start, args.v_lora_end)] + ["layers.{}.self_attn.v_proj".format(i) for i
                                                                          in range(args.v_lora_start, args.v_lora_end)]
        )
        language_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["v_proj", "q_proj"]
        )

        self.clip_model = CLIPModel.from_pretrained(args.clip_basemodel_path, local_files_only=True)
        self.clip_model.requires_grad_(False)

        self.processor = CLIPProcessor.from_pretrained(args.clip_basemodel_path, local_files_only=True)

        self.text_inputs = {}
        self.prompt_offset_indices = {}
        self.eos_offset = {}

        self.few_shot = few_shot

        self.num_prior_tokens = args.num_prior_tokens
        self.num_llm_prompts = args.num_llm_prompts
        self.num_text_template = args.num_text_template

        self.num_text_ctx = args.num_text_ctx
        self.llm_prompt_depth = args.llm_prompt_depth
        self.prompt_depth = max(self.llm_prompt_depth, args.text_prompt_depth)
        self.text_prompt_depth = self.prompt_depth
        self.visual_prompt_depth = args.visual_prompt_depth

        self.decoder_skip_connection = args.decoder_skip_connection
        self.concat_fixed_prompts = args.concat_fixed_prompts
        self.temperature = nn.Parameter(torch.tensor([0.1]))
        if self.concat_fixed_prompts:
            self.num_special_tokens = 4 + self.num_llm_prompts
        else:
            self.num_special_tokens = self.num_llm_prompts
        self.prompt_type = args.prompt_type

        self.suffixes = ['dw']
        self.suffixes_length = len(self.suffixes)
        target_list = ['base', 'new'] if not self.few_shot else ['all']

        for target in target_list:
            self.text_inputs[target] = self.processor(
                ["a photo of a {}".format(c.replace("_", " ")) for c in classnames[target]],
                return_tensors="pt", padding=True
            )

            if self.prompt_type == 'prefix':
                self.text_inputs[target]['input_ids'] = torch.cat((torch.ones(
                    (len(classnames[target]), self.num_special_tokens),
                    dtype=torch.long) * self.processor.tokenizer.bos_token_id, self.text_inputs[target].input_ids),
                                                                  dim=1)
                self.text_inputs[target]['attention_mask'] = torch.cat((torch.ones(
                    (len(classnames[target]), self.num_special_tokens), dtype=torch.long),
                                                                        self.text_inputs[target].attention_mask), dim=1)
            elif self.prompt_type == "suffix":
                # Suffix
                # Each element in eoc_loc represents the position of the end-of-sequence (EOS) token in the corresponding sequence
                eos_loc = self.text_inputs[target]['input_ids'].argmax(dim=-1)
                # Used to identify which sequences have EOS tokens that are not at the last position
                idx = eos_loc != (self.text_inputs[target]['input_ids'].shape[1] - 1)

                # Ensure the attention mask of the last token is 1
                self.text_inputs[target]['attention_mask'][:, -1] = 1
                self.text_inputs[target]['input_ids'] = torch.cat((self.text_inputs[target].input_ids, torch.ones(
                    (len(classnames[target]), self.num_special_tokens),
                    dtype=torch.long) * self.processor.tokenizer.pad_token_id), dim=1)
                self.text_inputs[target]['attention_mask'] = torch.cat((self.text_inputs[target].attention_mask,
                                                                        torch.ones((len(classnames[target]),
                                                                                    self.num_special_tokens),
                                                                                   dtype=torch.long)), dim=1)

                # Reposition the EOC location
                eos_loc = self.text_inputs[target]['input_ids'].argmax(dim=-1)
                # By setting the attention mask at the EOS position to 0, we can prevent the model from continuing attention computation after EOS
                self.text_inputs[target]['attention_mask'][torch.arange(len(classnames[target]))[idx], eos_loc[idx]] = 0

                self.eos_offset[target] = (
                torch.arange(len(classnames[target])), eos_loc)  # Store the position information of EOS (End of Sequence) tokens for each target class sequence

        self.eos_token_id = self.clip_model.text_model.eos_token_id

        if self.naive_decoding:
            if args.freeze_vit:
                self.lora_model = nn.ModuleDict({'default': self.clip_model.vision_model})
                self.lora_model.requires_grad_(False)
            else:
                self.lora_model = nn.ModuleDict(
                    {'default': LoraModel(self.clip_model.vision_model, {'default': vision_peft_config}, 'default')})

        self.text_hidden_size = self.clip_model.text_model.config.hidden_size
        self.visual_proj_size = self.text_hidden_size

        source_target_list = ['source', 'target']
        text_embeddings = {}
        for target in source_target_list:
            text_embeddings[target] = torch.load(os.path.join(self.dset[target].data_dir, args.clip_text_embed_file))
        self.base_embeddings = nn.ParameterDict()
        self.new_embeddings = nn.ParameterDict()
        for target in source_target_list:
            self.base_embeddings[target] = nn.ParameterDict()
            self.new_embeddings[target] = nn.ParameterDict()
            for suffix in self.suffixes:
                self.base_embeddings[target][suffix] = nn.Parameter(text_embeddings[target][suffix]['base']['avg'], requires_grad=False)
                self.new_embeddings[target][suffix] = nn.Parameter(text_embeddings[target][suffix]['new']['avg'], requires_grad=False)

        if self.few_shot:
            if indices is not None:
                self.text_embeddings = nn.ParameterDict({
                    'all': torch.cat((self.base_embeddings, self.new_embeddings), dim=0)[indices]
                })
            else:
                self.text_embeddings = nn.ParameterDict({
                    'all': torch.cat((self.base_embeddings, self.new_embeddings), dim=0)
                })
        else:
            self.text_embeddings = nn.ParameterDict({
                'base': self.base_embeddings,
                'new': self.new_embeddings,
            })

        self.distillation_type = args.distillation_type
        self.base_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.token_bias = args.token_bias

        self.visual_prompting = args.visual_prompting
        if self.visual_prompting:
            self.visual_prompts = nn.Parameter(torch.empty(
                (self.visual_prompt_depth, args.num_vis_ctx, self.clip_model.vision_model.config.hidden_size)).normal_(
                0, 1))
            nn.init.kaiming_uniform_(self.visual_prompts, a=math.sqrt(5))

        self.text_prompts = nn.ParameterList()
        ctx_init = "a photo of a"
        n_ctx = self.num_text_ctx
        prompt = self.processor([ctx_init], return_tensors="pt")  # Text to tokens
        with torch.no_grad():
            embedding = self.clip_model.text_model.embeddings(input_ids=prompt.input_ids)  # token to embedding
        self.init_prompt = nn.Parameter(embedding[0, 1: 1 + n_ctx, :], requires_grad=True)
        self.text_prompts.extend(nn.ParameterList([self.init_prompt]))

        self.in_layer_prompts = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_text_ctx, 512).normal_(0, 1), requires_grad=True) for _ in
            range(self.text_prompt_depth - 1)])
        for i in range(len(self.in_layer_prompts)):
            nn.init.kaiming_uniform_(self.in_layer_prompts[i], a=math.sqrt(5))

        self.text_prompts.extend(self.in_layer_prompts)
        self.num_decoder_layers = args.num_decoder_layers

        # print("Loading past key values from {}".format(args.past_key_value_file))
        content_dict = {'source': {'base': {}, 'new': {}},
                        'target': {'base': {}, 'new': {}}
                        }
        for target in source_target_list:
            for suffix in self.suffixes:
                file_name = args.past_key_value_file.format(suffix)
                temp_dict = {}
                temp_dict[target] = torch.load(os.path.join(dset[target].data_dir, file_name))
                for split in ['base', 'new']:
                    if split not in content_dict[target]:
                        content_dict[target][split] = {}
                    content_dict[target][split][suffix] = temp_dict[target][split]

        self.base_class_key_values = nn.ParameterList()
        self.base_class_attn_mask = []
        self.new_class_key_values = nn.ParameterList()
        self.new_class_attn_mask = []
        if self.few_shot:
            self.past_key_values = nn.ParameterDict({'all': nn.ParameterDict()})
            self.attention_mask = {'all': {}}

            for suffix in self.suffixes:
                if indices is not None:
                    self.past_key_values['all'][suffix] = nn.ParameterList([
                        nn.Parameter(x['past_key_values'][-self.num_decoder_layers:, :, indices], requires_grad=False)
                        for x in content_dict['all'][suffix]
                    ])
                    self.attention_mask['all'][suffix] = [
                        x['attn_mask'][indices] for x in content_dict['all'][suffix]
                    ]
                else:
                    self.past_key_values['all'][suffix] = nn.ParameterList([
                        nn.Parameter(x['past_key_values'][-self.num_decoder_layers:], requires_grad=False)
                        for x in content_dict['all'][suffix]
                    ])
                    self.attention_mask['all'][suffix] = [
                        x['attn_mask'] for x in content_dict['all'][suffix]
                    ]
        else:
            self.past_key_values = {
            'source': nn.ParameterDict({
                'base': nn.ParameterDict(),
                'new': nn.ParameterDict()}),
            'target': nn.ParameterDict({
                'base': nn.ParameterDict(),
                'new': nn.ParameterDict()}),
            }
            self.attention_mask = {
                'source': {'base': {},'new': {}},
                'target': {'base': {},'new': {}}
            }
            for target in source_target_list:
                for split in ['base', 'new']:
                    for suffix in self.suffixes:
                        self.past_key_values[target][split][suffix] = nn.ParameterList([
                            nn.Parameter(x['past_key_values'][-self.num_decoder_layers:], requires_grad=False)
                            for x in content_dict[target][split][suffix]
                        ])
                        self.attention_mask[target][split][suffix] = [
                            x['attn_mask'] for x in content_dict[target][split][suffix]
                        ]

        self.class_token = nn.ParameterDict({
            suffix: nn.ParameterList([
                nn.Parameter(torch.empty((self.num_llm_prompts, model.config.hidden_size)).normal_(0, 1))
                for _ in range(1)
            ])
            for suffix in self.suffixes
        })

        for suffix in self.suffixes:
            for i in range(len(self.class_token[suffix])):
                nn.init.kaiming_uniform_(self.class_token[suffix][i], a=math.sqrt(5))

        self.class_proj = nn.Identity()

        self.class_norm = nn.ModuleDict({
            suffix: copy.deepcopy(model.model.norm)
            for suffix in self.suffixes
        })

        if args.lora_decoding:
            self.class_decoder = nn.ModuleDict({
                suffix: nn.ModuleList([
                    LoraModel(copy.deepcopy(model.model.layers[i]), {'default': language_peft_config}, 'default')
                    for i in range(-self.num_decoder_layers, 0)
                ])
                for suffix in self.suffixes
            })
            for suffix in self.suffixes:
                self.class_norm[suffix].requires_grad_(False)
        else:
            self.class_decoder = nn.ModuleDict({
                suffix: nn.ModuleList([
                    copy.deepcopy(model.model.layers[i])
                    for i in range(-self.num_decoder_layers, 0)
                ])
                for suffix in self.suffixes
            })

            for suffix in self.suffixes:
                self.class_decoder[suffix].requires_grad_(True)
                self.class_norm[suffix].requires_grad_(True)

        self.text_proj = nn.ModuleList(
            [nn.Linear(model.config.hidden_size, self.text_hidden_size, bias=False) for _ in
             range(self.llm_prompt_depth)]
        )

        self.llm_prompt_bias = nn.ParameterDict({
            suffix: nn.ParameterList([
                nn.Parameter(torch.empty(self.num_special_tokens, 512).normal_(0, 1))
                for _ in range(self.llm_prompt_depth)
            ])
            for suffix in self.suffixes
        })

        for suffix in self.suffixes:
            for i in range(len(self.llm_prompt_bias[suffix])):
                nn.init.kaiming_uniform_(self.llm_prompt_bias[suffix][i], a=math.sqrt(5))

        self.class_embed_weight = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.combine_weights = nn.Parameter(torch.ones(self.suffixes_length) / self.suffixes_length)
        self.combine_weights_CLIPtext = nn.Parameter(torch.ones(self.suffixes_length) / self.suffixes_length)

        if args.learn_class_embed_weight:
            self.class_embed_weight.requires_grad_(True)

        if args.prompt_learning:
            for suffix in self.suffixes:
                self.class_decoder[suffix].requires_grad_(False)
                self.class_norm[suffix].requires_grad_(False)

        if args.freeze_decoder_kv_proj:
            for suffix in self.suffixes:
                for decoder in self.class_decoder[suffix]:
                    decoder.self_attn.k_proj.requires_grad_(False)
                    decoder.self_attn.v_proj.requires_grad_(False)

        if args.freeze_decoder_q_proj:
            for suffix in self.suffixes:
                for decoder in self.class_decoder[suffix]:
                    decoder.self_attn.q_proj.requires_grad_(False)

        if args.freeze_decoder_o_proj:
            for suffix in self.suffixes:
                for decoder in self.class_decoder[suffix]:
                    decoder.self_attn.o_proj.requires_grad_(False)

        if args.freeze_decoder_attn:
            for suffix in self.suffixes:
                for decoder in self.class_decoder[suffix]:
                    decoder.self_attn.requires_grad_(False)

        if args.freeze_decoder_ffn:
            for suffix in self.suffixes:
                for decoder in self.class_decoder[suffix]:
                    decoder.mlp.requires_grad_(False)

        self.class_fn = self.decode_class
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1 / 0.01)]), requires_grad=True)
        self.dropout = nn.Dropout(args.prompt_dropout)
        self.image_dropout = nn.Dropout(args.img_dropout)
        self.lambda_dist = args.lambda_dist
        self.accumulated_features = []
        self.accumulated_labels = []
        self.current_accumulation_step = 0

        self.mag_module = MultimodalAlignmentGuidance(
            visual_dim=768,
            text_dim=512,
            shared_dim=512,
            num_tokens=64,
            temperature=args.mag_temperature
        )
        self.visual_features_by_layer = []
        self.text_features_by_layer = []
        self.visual_threshold_by_layer = []
        self.text_threshold_by_layer = []

    def decode_class(self, subset='base', bias=None):
        if subset =='base':
            target = 'source'
        else:
            target = 'target'
        pkv = self.past_key_values[target][subset]
        attention_mask = self.attention_mask[target][subset]
        if self.training:
            template_idx = torch.randint(self.num_text_template, (1,)).item()
            if self.token_bias:
                selected_embeddings = {}
                selected_attn_mask = {}
                for suffix in self.suffixes:
                    selected_embeddings[suffix] = self.next_token_bias[subset][suffix][template_idx]
                    selected_attn_mask[suffix] = self.next_token_attn_mask[subset][suffix][template_idx]
            else:
                selected_embeddings = None
                selected_attn_mask = None

            pkv_idx = []
            att_mask_idx = []
            for suffix in self.suffixes:
                pkv_idx.append(pkv[suffix][template_idx])
                att_mask_idx.append(attention_mask[suffix][template_idx])

            encoded_prompt = self.generate_text_features_from_prompt(
                pkv_idx,
                att_mask_idx,
                self.class_token,
                selected_embeddings,
                selected_attn_mask,
                subset=subset
            )
        else:
            encoded_prompts = []
            for template_idx in range(self.num_text_template):
                if self.token_bias:
                    selected_embeddings = {}
                    selected_attn_mask = {}
                    for suffix in self.suffixes:
                        selected_embeddings[suffix] = self.next_token_bias[subset][suffix][template_idx]
                        selected_attn_mask[suffix] = self.next_token_attn_mask[subset][suffix][template_idx]
                else:
                    selected_embeddings = None
                    selected_attn_mask = None

                pkv_idx = []
                att_mask_idx = []
                for suffix in self.suffixes:
                    pkv_idx.append(pkv[suffix][template_idx])
                    att_mask_idx.append(attention_mask[suffix][template_idx])

                encoded_prompt = self.generate_text_features_from_prompt(
                    pkv_idx,
                    att_mask_idx,
                    self.class_token,
                    selected_embeddings,
                    selected_attn_mask,
                    subset=subset
                )
                encoded_prompts.append(encoded_prompt)

            encoded_prompt = torch.stack(encoded_prompts, dim=0)
        if subset == 'base':
            dtset = 'source'
        else:
            dtset = 'target'
        outputs = ((encoded_prompt, self.text_embeddings[subset]),)
        return outputs

    def generate_text_features_from_prompt(self, pkv, attention_mask, class_token, selected_embeddings=None,
                                           selected_attn_mask=None, subset='base'):
        if subset == 'base':
            dtset = 'source'
        else:
            dtset = 'target'

        first_level_key = next(iter(self.text_embeddings))
        second_level_key = next(iter(self.text_embeddings[first_level_key][dtset]))
        num_classes = self.text_embeddings[subset][dtset][second_level_key].shape[0]
        tokens = {}
        for suffix in self.suffixes:
            suffix_token = self.class_proj(class_token[suffix][0])
            tokens[suffix] = suffix_token.unsqueeze(0).expand(num_classes, -1, -1)

        device = next(iter(tokens.values())).device

        updated_attention_masks = {}
        if selected_embeddings is not None:
            attention_mask = torch.cat((attention_mask.to(device), selected_attn_mask.to(device),
                                        torch.ones((attention_mask.shape[0], tokens.shape[1])).to(device)), dim=1)

            tokens = torch.cat([
                selected_embeddings,
                tokens,
            ], dim=1)
        else:
            # Add all-ones mask so the model attends to the newly added class tokens
            for idx, suffix in enumerate(self.suffixes):
                current_attention_mask = attention_mask[idx]
                current_tokens = tokens[suffix]
                current_attention_mask = current_attention_mask.to(current_tokens.device)
                ones_tensor = torch.ones((current_attention_mask.shape[0], current_tokens.shape[1]),
                                         device=current_tokens.device)
                updated_attention_mask = torch.cat((current_attention_mask, ones_tensor), dim=1)
                updated_attention_masks[suffix] = updated_attention_mask
            attention_mask = [updated_attention_masks[suffix] for suffix in self.suffixes]

        # Cumsum on last dim of attention_mask, clamped non-negative
        # Keep only last 16 dims (new tokens) to ensure uniform position encoding start across batch
        position_ids = []
        for suffix in self.suffixes:
            current_attention_mask = updated_attention_masks[suffix]
            current_tokens = tokens[suffix]
            # position_ids
            current_position_ids = torch.clamp(torch.cumsum(current_attention_mask, dim=-1).long() - 1, min=0)
            current_position_ids = current_position_ids[:, -current_tokens.shape[1]:]
            position_ids.append(current_position_ids)
        prepared_attention_masks = []
        for suffix in self.suffixes:
            current_attention_mask = attention_mask[self.suffixes.index(suffix)]
            current_tokens = tokens[suffix]
            current_pkv = pkv[self.suffixes.index(suffix)]
            prepared_attention_mask = _prepare_4d_causal_attention_mask(
                current_attention_mask,
                (num_classes, current_tokens.shape[1]), # (class_num, 16)
                current_tokens,
                current_pkv.shape[-2]
            )
            prepared_attention_masks.append(prepared_attention_mask)
        attention_mask = prepared_attention_masks

        hidden_states = [tokens[suffix] for suffix in self.suffixes]
        new_hidden_states = []
        normalized_hidden_states = []
        class_embed = []
        for suffix_idx, suffix in enumerate(self.suffixes):
            current_hidden_states = hidden_states[suffix_idx]
            current_attention_mask = attention_mask[suffix_idx]
            current_position_ids = position_ids[suffix_idx]
            current_pkv = pkv[suffix_idx].to(device=current_hidden_states.device,dtype=current_hidden_states.dtype)
            for idx, decoder_layer in enumerate(self.class_decoder[suffix]):
                layer_outputs = decoder_layer(
                    current_hidden_states,
                    attention_mask=current_attention_mask,
                    position_ids=current_position_ids,
                    past_key_value=current_pkv[idx],
                    use_cache=False,
                    output_attentions=True
                )
                current_hidden_states = layer_outputs[0]
            new_hidden_states.append(current_hidden_states)
            current_hidden_states = new_hidden_states[suffix_idx]
            normalized_current_hidden_states = self.class_norm[suffix](current_hidden_states)
            normalized_hidden_states.append(normalized_current_hidden_states)
            n_current_hidden_states = normalized_hidden_states[suffix_idx]
            current_class_embed = n_current_hidden_states[:, -self.num_special_tokens:,
                                  :]
            if self.args.prompt_dropout != 0:
                current_class_embed = self.dropout(current_class_embed) * self.class_embed_weight.exp()  # [50,16,4096]
            class_embed.append(current_class_embed)

        # hidden_states = normalized_hidden_states
        all_embeds = []
        for suffix_idx, suffix in enumerate(self.suffixes):
            current_class_embed = class_embed[suffix_idx]
            current_embeds = []
            for i in range(self.llm_prompt_depth):
                projected_embed = self.text_proj[i](current_class_embed)
                biased_embed = projected_embed + self.llm_prompt_bias[suffix][i]
                current_embeds.append(biased_embed)
            all_embeds.append(current_embeds)

        averaged_embeds = []
        for depth in range(self.llm_prompt_depth):
            depth_embeds = [all_embeds[i][depth] for i in range(self.suffixes_length)]
            normalized_weights = nn.functional.softmax(self.combine_weights, dim=0)
            combined_embed = sum(w * embed for w, embed in zip(normalized_weights, depth_embeds))
            # averaged_embed = torch.stack(depth_embeds, dim=0).mean(dim=0)
            averaged_embeds.append(combined_embed)

        stacked_embeds = torch.stack(averaged_embeds, dim=0)
        encoded_prompt = self.encode_LLM_prompt(stacked_embeds, subset=subset)
        return encoded_prompt

    def encode_LLM_prompt(self, prompts, subset):
        device = prompts.device
        input_ids = self.text_inputs[subset].input_ids.to(device)
        attention_mask = self.text_inputs[subset].attention_mask.to(device)
        position_ids = None
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        self.text_features_by_layer = []
        # self.text_token_importance_by_layer = []

        if self.prompt_type == 'suffix':
            # [50,30,512]
            hidden_states = self.clip_model.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
            initial_text_prompts = self.text_prompts[0]
            hidden_states = torch.cat([
                hidden_states[:, :1, :],
                initial_text_prompts.unsqueeze(0).expand(hidden_states.shape[0], -1, -1),
                hidden_states[:, 1 + self.num_text_ctx:-self.num_special_tokens - 1, :],
                prompts[0],
                hidden_states[self.eos_offset[subset]].unsqueeze(1)
            ], dim=1)

        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
        for idx, encoder_layer in enumerate(self.clip_model.text_model.encoder.layers):
            if idx > 0 and idx < self.text_prompt_depth:
                if self.prompt_type == 'suffix':
                    if idx < self.llm_prompt_depth:
                        current_text_prompts = self.text_prompts[idx]
                        hidden_states = torch.cat([
                            hidden_states[:, :1, :],
                            current_text_prompts.unsqueeze(0).expand(hidden_states.shape[0], -1, -1),
                            hidden_states[:, 1 + self.num_text_ctx:-self.num_special_tokens - 1, :],
                            prompts[idx],
                            hidden_states[:, -1:, :]
                        ], dim=1)

            layer_outputs = encoder_layer(  # clip.textencoder
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=self.training
            )
            hidden_states = layer_outputs[0]
            if self.training:
                self_attention = layer_outputs[1]

            if self.training and idx < self.llm_prompt_depth:
                # 1. Get self-attention maps
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    self_attn = self_attention  # Self-Attention Maps
                    # 2. Extract class attention map (EOS token attending to other tokens)
                    eos_attn = extract_cls_attention(self_attn, is_vision=False)
                    # 3. Compute MAG attention on current hidden_states
                    text_attn, text_features = self.mag_module.forward_text(hidden_states)
                    token_attn = text_attn  # token attentan map
                    # 4. Compute token importance scores
                    token_importance = calculate_token_importance_score(
                        eos_attn, self_attn, token_attn, is_vision=False
                    )
                    # 5. Save filtered features and importance scores
                    self.text_features_by_layer.append(text_features)

                    S_text = hidden_states.shape[1]
                    # Mask 1: LLM mapped tokens (excluding class names)
                    target_mask1_text = torch.zeros(S_text, dtype=torch.bool, device=hidden_states.device)
                    llm_token_start_idx = S_text - self.num_special_tokens - 1
                    llm_token_end_idx = S_text - 1
                    if llm_token_start_idx < llm_token_end_idx:
                        target_mask1_text[llm_token_start_idx: llm_token_end_idx] = True
                    # Exclude class name tokens from mask1
                    class_name_start_idx = 1 + self.num_text_ctx
                    class_name_end_idx = S_text - self.num_special_tokens - 1
                    if class_name_start_idx < class_name_end_idx:
                        target_mask1_text[class_name_start_idx: class_name_end_idx] = False

                    # Mask 2: Prompt tokens added at this layer
                    target_mask2_text = torch.zeros(S_text, dtype=torch.bool, device=hidden_states.device)
                    prompt_start_idx = 0
                    prompt_end_idx = 1 + self.num_text_ctx
                    if prompt_start_idx < prompt_end_idx:
                        target_mask2_text[prompt_start_idx: prompt_end_idx] = True

                    if self.token_modification_method == 'importance_based_dropout':
                        hidden_states = apply_importance_based_dropout(
                            hidden_states, token_importance, is_vision=False,
                            target_mask1=target_mask1_text,
                            target_mask2=target_mask2_text,
                            noise_sigma=self.noise_injection_sigma,
                            noise_max_prob=self.noise_max_prob,
                            noise_min_prob=self.noise_min_prob
                        )
        last_hidden_state = hidden_states
        last_hidden_state = self.clip_model.text_model.final_layer_norm(last_hidden_state)

        if self.prompt_type == 'prefix':
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[:, -1, :]
        text_features = self.clip_model.text_projection(pooled_output)  # [50,512]

        return text_features

    def forward(self, x, current_epoch=None, subset=None):
        if self.training:
            self.current_epoch = current_epoch
            loss, pred = self.run(x)
            return loss, pred
        else:
            scores = self.run(x, subset)
            return None, scores

    def compute_all_class_embeddings(self, subset):
        outputs = self.class_fn(subset=subset)
        class_embed = outputs[0]

        self.all_class_embed = class_embed

    def extract_image_features(self, img, ori, subset, labels=None, target="default", dropout=False):
        if self.visual_prompting:
            image_features = self.extract_prompt_image_features(img, ori, subset, labels, model=self.lora_model[target])
        else:
            image_features = self.lora_model[target](img)[1]
            if dropout:
                image_features = self.image_dropout(image_features)
            image_features = self.clip_model.visual_projection(image_features)
        return image_features

    def extract_prompt_image_features(self, img, ori, subset, labels, model, dropout=False):
        self.visual_features_by_layer =[]
        self.visual_token_importance_by_layer = []

        hidden_states = model.embeddings(img)  # [batch,197,768]
        hidden_states = torch.cat(
            [hidden_states, self.visual_prompts[0].unsqueeze(0).expand(hidden_states.shape[0], -1, -1)],
            dim=1)
        hidden_states = model.pre_layrnorm(hidden_states)

        for idx, encoder_layer in enumerate(model.encoder.layers):
            if idx > 0 and idx < self.visual_prompt_depth:
                current_visual_prompts = self.visual_prompts[idx]
                hidden_states = torch.cat(
                    [hidden_states[:, :-self.visual_prompts.shape[1]],
                     current_visual_prompts.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)],
                    dim=1
                )
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=self.training
            )
            hidden_states = layer_outputs[0]  # [batch,201,768]
            if self.training:
                self_attention = layer_outputs[1]

            if self.training and idx < self.visual_prompt_depth:
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    self_attn = self_attention
                    cls_attn = extract_cls_attention(self_attn, is_vision=True)
                    visual_attn, visual_features = self.mag_module.forward_visual(hidden_states)
                    token_importance = calculate_token_importance_score(
                        cls_attn, self_attn, visual_attn, is_vision=True
                    )
                    self.visual_features_by_layer.append(visual_features)
                    S_vis = hidden_states.shape[1]
                    # Mask 1: Patch tokens (excluding CLS)
                    target_mask1_vis = torch.zeros(S_vis, dtype=torch.bool, device=hidden_states.device)
                    patch_end_idx = S_vis - self.visual_prompts.shape[1]  # End of patch tokens (before prompts)
                    if 1 < patch_end_idx:  # Ensure indices are valid (start from 1 to exclude CLS)
                        target_mask1_vis[1:patch_end_idx] = True
                    # Mask 2: Prompt tokens added at this layer
                    target_mask2_vis = torch.zeros(S_vis, dtype=torch.bool, device=hidden_states.device)
                    target_mask_vis_initial = target_mask1_vis.clone()

                    prompt_start_idx = S_vis - self.visual_prompts.shape[1]  # Start of prompts
                    prompt_end_idx = S_vis
                    if prompt_start_idx < prompt_end_idx:
                        target_mask2_vis[prompt_start_idx: prompt_end_idx] = True

                    current_epoch = getattr(self, 'current_epoch', -1)

                    if self.token_modification_method == 'importance_based_dropout':
                        hidden_states = apply_importance_based_dropout(
                            hidden_states, token_importance, is_vision=True,
                            target_mask1=target_mask1_vis,
                            target_mask2=target_mask2_vis,
                            noise_sigma=self.noise_injection_sigma,
                            noise_max_prob=self.noise_max_prob,
                            noise_min_prob=self.noise_min_prob,
                            current_epoch = current_epoch,
                        )

        last_hidden_states = hidden_states
        pooled_output = last_hidden_states[:, 0, :]
        pooled_output = model.post_layernorm(pooled_output)

        visual_features = self.clip_model.visual_projection(pooled_output)
        visual_features = visual_features

        return visual_features

    def run(self, x, subset=None):
        if self.training:
            img, img2, labels, _ = x
        else:
            img = x[0]
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.605)

        normalize_fn = lambda x: F.normalize(x, dim=-1)
        logit_scale = self.logit_scale.exp()

        embeds = {}

        if self.training:
            embeds['learn'], embeds['clip'] = self.class_fn(subset='base')[0]
        else:
            embeds['learn'], embeds['clip'] = self.all_class_embed

        if self.training:
            embeds['clip'] = torch.stack(list(embeds['clip']['source'].values())).mean(dim=0)
        else:
            if subset == 'base':
                embeds['clip'] = torch.stack(list(embeds['clip']['source'].values())).mean(dim=0)
            else:
                embeds['clip'] = torch.stack(list(embeds['clip']['target'].values())).mean(dim=0)

        with torch.inference_mode():
            orig_image_features = self.clip_model.vision_model(img)[1]
            orig_image_features = self.clip_model.visual_projection(
                orig_image_features)

        learn_text_embeding = embeds['learn']
        raw_text_embeding = embeds['clip']
        x1 = self.adapter(learn_text_embeding)
        x2 = self.adapter(raw_text_embeding)

        # Swap the dimensions of embeds
        for k, v in embeds.items():
            if embeds[k].ndim == 3:
                embeds[k] = normalize_fn(v).permute(0, 2, 1)
            else:
                embeds[k] = normalize_fn(v).permute(1, 0)


        target_pred = {}
        generated_pred = {}
        if self.training:
            class_features = self.extract_image_features(img2, ori=orig_image_features, labels=labels,
                                                         subset='base')
            image_features = class_features
            for feature_name in ['image_features']:
                if locals()[feature_name].ndim != embeds['learn'].ndim:
                    locals()[feature_name] = locals()[feature_name].unsqueeze(0)
            target_pred['learn'] = normalize_fn(image_features) @ embeds['learn']
            target_pred['clip'] = normalize_fn(orig_image_features) @ embeds['clip']

            target_pred['learn'] = target_pred['learn'].float()

            target_pred['clip'] = target_pred['clip'].float()
            raw_text_embeding = raw_text_embeding.float()
            learn_text_embeding = learn_text_embeding.float()
            image_features = image_features.float()
            orig_image_features = orig_image_features.float()
        else:
            class_features = self.extract_image_features(img,ori=orig_image_features, subset=subset)
            image_features = class_features
            if image_features.ndim != embeds['learn'].ndim:
                image_features = image_features.unsqueeze(0)
            target_pred['learn'] = torch.matmul(normalize_fn(image_features), embeds['learn'])
            target_pred['learn'] = target_pred['learn'].float()
            target_pred['clip'] = normalize_fn(orig_image_features) @ embeds['clip']
            target_pred['clip'] = target_pred['clip'].float()

        if self.training:
            total_loss = self.base_loss(target_pred['learn'] * logit_scale, labels)
            total_loss += F.l1_loss(normalize_fn(learn_text_embeding),
                                    normalize_fn(raw_text_embeding)) * 25
            total_loss += F.l1_loss(normalize_fn(image_features),
                                    normalize_fn(orig_image_features)) * 10
            if self.distillation_type == 'soft':
                dist_loss = F.kl_div(F.log_softmax(target_pred['learn'] * logit_scale, dim=-1),
                                     F.log_softmax(target_pred['clip'] * logit_scale, dim=-1),
                                     reduction='sum', log_target=True) / target_pred['learn'].numel()
                total_loss += dist_loss * self.lambda_dist
            if self.args.fix_lam:
                lam = self.args.lam_res
            else:
                lam = self.args.lam_res - get_annealing_down_params(self.args.lam_res, self.current_epoch,
                                                                    self.args.max_epochs)
            zn_text = Demix(normalize_fn(learn_text_embeding), normalize_fn(raw_text_embeding), lam,
                            image_features=normalize_fn(orig_image_features), logit_scale=logit_scale)
            zn_vision = Demix(normalize_fn(image_features), normalize_fn(orig_image_features), lam,
                              text_features=normalize_fn(raw_text_embeding), logit_scale=logit_scale)
            total_loss += -self.args.weight_res * (zn_text + zn_vision)
            if len(self.visual_features_by_layer) > 0 and len(self.text_features_by_layer) > 0:
                mag_loss = 0.0
                num_layers = min(len(self.visual_features_by_layer), len(self.text_features_by_layer))
                for layer_idx in range(num_layers):
                    visual_feat = self.visual_features_by_layer[layer_idx]
                    text_feat = self.text_features_by_layer[layer_idx]
                    layer_loss = self.mag_module.compute_similarity_loss(
                        visual_feat, text_feat, labels
                    )
                    mag_loss += layer_loss
                mag_loss /= num_layers
                total_loss += mag_loss * 30
            return total_loss, target_pred['learn']
        else:
            if target_pred['learn'].ndim == 2:
                return target_pred['learn']
            else:
                return F.softmax(target_pred['learn'].float(), dim=-1).mean(dim=0)
