import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calculate_token_importance_score(cls_attn, self_attn, token_attn, is_vision=True):
    """
    Calculate token importance score for vision and text branches.

    Args:
        cls_attn: Class attention scores, shape [batch_size, token_num-1]
        self_attn: Self-attention map, shape [batch_size, num_heads, token_num, token_num]
        token_attn: Token attention scores from MAG module, shape [batch_size, num_learnable_tokens, token_num]
        is_vision: Whether this is the vision branch

    Returns:
        token_importance_score: Token importance scores, shape [batch_size, token_num-1]
    """
    batch_size, num_heads, seq_len, _ = self_attn.shape

    # Build content token mask (excluding special tokens)
    content_tokens_idx = torch.ones(seq_len, dtype=torch.bool, device=self_attn.device)
    if is_vision:
        content_tokens_idx[0] = False  # [CLS]
    else:
        content_tokens_idx[-1] = False  # [EOS]

    # Build attention mask: remove self-attention diagonal
    diag_mask = ~torch.eye(seq_len, dtype=torch.bool, device=self_attn.device)[None, None, :, :]
    self_attn_mask = diag_mask

    # Select content tokens (keep in both query and key dimensions)
    content_self_attn = self_attn[:, :, content_tokens_idx, :][:, :, :, content_tokens_idx]  # [B, H, Q', K']
    mask = self_attn_mask[:, :, content_tokens_idx, :][:, :, :, content_tokens_idx]

    # Apply mask and get max attention scores
    masked_self_attn = content_self_attn.masked_fill(~mask, float('-inf'))
    self_attn_max = torch.max(masked_self_attn, dim=3)[0]  # [B, H, Q']

    # Average across heads
    self_attn_w = self_attn_max.mean(dim=1)  # [B, Q']

    # Normalize
    self_attn_w = self_attn_w / (self_attn_w.sum(dim=1, keepdim=True) + 1e-8)

    # Process token attention scores - exclude special token positions
    if is_vision:
        token_attn_for_content = token_attn[:, :, 1:]  # Exclude [CLS]
    else:
        token_attn_for_content = token_attn[:, :, :-1]  # Exclude [EOS]
    token_attn_max = torch.max(token_attn_for_content, dim=1)[0]  # [batch_size, token_num-1]
    token_attn_w = token_attn_max / (token_attn_max.sum(dim=1, keepdim=True) + 1e-8)

    # Normalize cls_attn
    cls_attn = cls_attn / (cls_attn.sum(dim=1, keepdim=True) + 1e-8)

    # Validate dimensions
    assert self_attn_w.shape == token_attn_w.shape == cls_attn.shape, \
        f"dimension not match: self_attn_w={self_attn_w.shape}, token_attn_w={token_attn_w.shape}, cls_attn={cls_attn.shape}"

    # Compute final token importance score
    token_importance_score = (cls_attn + self_attn_w + token_attn_w) / 3.0

    return token_importance_score


def extract_cls_attention(attention_maps, is_vision=True):
    """
    Extract CLS/EOS token attention to other tokens.

    Args:
        attention_maps: Attention maps, shape [batch_size, num_heads, seq_len, seq_len]
        is_vision: If True, extract first token attention; otherwise extract last token attention

    Returns:
        cls_attention: Shape [batch_size, seq_len-1]
    """
    batch_size, num_heads, seq_len, _ = attention_maps.shape

    if is_vision:
        # Vision branch: use CLS token (index 0) attention to other tokens
        cls_attention = attention_maps[:, :, 0, 1:]  # [batch_size, num_heads, seq_len-1]
    else:
        # Text branch: use EOS token (last one) attention to other tokens
        cls_attention = attention_maps[:, :, -1, :-1]  # [batch_size, num_heads, seq_len-1]

    # Average across all heads
    cls_attention = cls_attention.mean(dim=1)  # [batch_size, seq_len-1]

    return cls_attention

def _normalize_importance_scores(scores):
    """
    Min-Max normalize importance scores to [0, 1] range for each sample.
    Input scores: [B, S']
    Output normalized_scores: [B, S']
    """
    min_vals = torch.min(scores, dim=1, keepdim=True)[0]
    max_vals = torch.max(scores, dim=1, keepdim=True)[0]
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized = (scores - min_vals) / range_vals
    return normalized

def _importance_to_probability(normalized_score, max_prob=0.5, min_prob=0.05):
    """
    Map normalized importance score to dropout probability.
    Higher score (more important) -> lower probability.
    Input normalized_score: Score in [0, 1] range
    Output probability: Probability in [min_prob, max_prob] range
    """
    return max_prob - normalized_score * (max_prob - min_prob)

def apply_importance_based_dropout(
    hidden_states, token_importance_score, is_vision,
    target_mask1, target_mask2, noise_sigma,
    noise_max_prob=0.5, noise_min_prob=0.05,
    current_epoch=-1,
):
    """
    Apply random Gaussian noise based on token importance with dynamic probability assignment.
    Returns a modified copy without altering the original hidden_states.

    Args:
        hidden_states: Input features [B, S, D]
        token_importance_score: Token importance scores [B, S'] (unnormalized)
        is_vision: Whether this is the vision branch
        target_mask1: Boolean mask [S] marking first group of target tokens
        target_mask2: Boolean mask [S] marking second group of target tokens
        noise_sigma: Gaussian noise standard deviation
        noise_max_prob: Max noise probability for least important tokens
        noise_min_prob: Min noise probability for most important tokens

    Returns:
        modified_hidden_states: Features after applying noise [B, S, D]
    """
    B, S, D = hidden_states.shape
    _, S_prime = token_importance_score.shape

    if noise_sigma <= 0 or noise_max_prob < noise_min_prob or (noise_max_prob == 0 and noise_min_prob == 0):
        return hidden_states

    modified_hidden_states = hidden_states.clone()

    # Prepare and normalize importance scores
    full_scores_normalized = torch.full((B, S), 0.5,
                                        device=hidden_states.device,
                                        dtype=token_importance_score.dtype)

    normalized_importance = _normalize_importance_scores(token_importance_score)  # [B, S']

    # Fill normalized scores to correct positions
    if is_vision:
        if S == S_prime + 1: full_scores_normalized[:, 1:] = normalized_importance
        elif S == S_prime: full_scores_normalized = normalized_importance
        else: raise ValueError("Vision dimension mismatch for importance scores")
    else:
        if S == S_prime + 1: full_scores_normalized[:, :-1] = normalized_importance
        elif S == S_prime: full_scores_normalized = normalized_importance
        else: raise ValueError("Text dimension mismatch for importance scores")

    # Determine target tokens
    target_mask1_dev = target_mask1.to(device=hidden_states.device)
    target_mask2_dev = target_mask2.to(device=hidden_states.device)
    if target_mask1_dev.dim() == 1:
        target_mask1_dev = target_mask1_dev.unsqueeze(0).expand(B, -1)
    if target_mask2_dev.dim() == 1:
        target_mask2_dev = target_mask2_dev.unsqueeze(0).expand(B, -1)

    combined_target_mask = target_mask1_dev | target_mask2_dev  # [B, S]

    # Generate Gaussian noise
    gaussian_noise = torch.randn_like(hidden_states) * noise_sigma + 1.0

    # Apply noise based on importance-derived probabilities
    for i in range(B):
        indices_target_sample = torch.where(combined_target_mask[i])[0]
        num_targets = indices_target_sample.numel()

        if num_targets > 0:
            scores_sample = full_scores_normalized[i, indices_target_sample]  # [num_targets]
            noise_probs_sample = _importance_to_probability(scores_sample, noise_max_prob, noise_min_prob)
            random_draws = torch.rand(num_targets, device=hidden_states.device)
            noise_decision_mask = random_draws < noise_probs_sample
            final_indices_to_noise = indices_target_sample[noise_decision_mask]

            if final_indices_to_noise.numel() > 0:
                modified_hidden_states[i, final_indices_to_noise, :] *= gaussian_noise[i, final_indices_to_noise, :]

    return modified_hidden_states