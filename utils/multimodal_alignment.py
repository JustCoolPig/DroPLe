import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultimodalAlignmentGuidance(nn.Module):
    def __init__(self, visual_dim=768, text_dim=512, shared_dim=512, num_tokens=64, temperature=1.0):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.shared_dim = shared_dim
        self.num_tokens = num_tokens
        self.temperature = temperature
        # Learnable tokens as modality bridge
        self.learnable_tokens = nn.Parameter(torch.empty(num_tokens, shared_dim))
        nn.init.kaiming_uniform_(self.learnable_tokens, a=math.sqrt(5))

        # Mapping layers
        self.visual_mapping = nn.Linear(visual_dim, shared_dim)
        self.text_mapping = nn.Linear(text_dim, shared_dim)

    def forward_visual(self, visual_tokens):
        mapped_visual_tokens = self.visual_mapping(visual_tokens)
        # Pass temperature to _compute_attention
        visual_attn = self._compute_attention(mapped_visual_tokens)
        visual_features = self._extract_features(mapped_visual_tokens, visual_attn)
        return visual_attn, visual_features

    def forward_text(self, text_tokens):
        mapped_text_tokens = self.text_mapping(text_tokens)
        # Pass temperature to _compute_attention
        text_attn = self._compute_attention(mapped_text_tokens)
        text_features = self._extract_features(mapped_text_tokens, text_attn)
        return text_attn, text_features

    def compute_similarity_loss(self, visual_features, text_features, labels):
        """Compute similarity loss"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        total_loss = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            # Get the label for the current sample
            label_idx = labels[i].item()

            # Get features
            v_feat = visual_features[i]  # [num_tokens, shared_dim]
            t_feat = text_features[label_idx]  # [num_tokens, shared_dim]

            # Normalize
            v_feat_norm = F.normalize(v_feat, dim=-1)
            t_feat_norm = F.normalize(t_feat, dim=-1)

            # Compute intermediate distribution
            m_feat = 0.5 * (v_feat_norm + t_feat_norm)

            kl_v_m = 1.0 - torch.sum(v_feat_norm * m_feat, dim=-1)
            kl_t_m = 1.0 - torch.sum(t_feat_norm * m_feat, dim=-1)
            js_div = 0.5 * (kl_v_m + kl_t_m)

            total_loss += js_div.mean()

        return total_loss / batch_size

    def _compute_attention(self, tokens):
        """Compute attention between learnable tokens and input tokens"""
        # Expand learnable tokens to match batch_size
        batch_size = tokens.size(0)
        learnable_tokens_expanded = self.learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        attention_scores = torch.matmul(
            learnable_tokens_expanded,
            tokens.transpose(1, 2)
        )

        scaled_scores = attention_scores * self.temperature
        # Apply softmax
        attention = F.softmax(scaled_scores, dim=-1)

        return attention

    def _extract_features(self, tokens, attention):
        """Extract features using attention weights"""
        features = torch.matmul(attention, tokens)
        return features