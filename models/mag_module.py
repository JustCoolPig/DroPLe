import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiModalityAlignmentGuidance(nn.Module):
    """Multi-modality Alignment Guidance Module

    This module implements alignment of visual and linguistic features
    based on learnable shared tokens.
    """

    def __init__(self,
                 visual_dim,  # Visual feature dimension
                 text_dim,  # Text feature dimension
                 shared_dim,  # Shared feature space dimension
                 num_tokens=4,  # Number of shared tokens
                 clip_model=None,
                 ):
        super().__init__()

        self.num_tokens = num_tokens
        self.shared_dim = shared_dim

        # Learnable shared tokens
        self.shared_tokens = nn.Parameter(
            torch.empty(num_tokens, shared_dim).normal_(0, 1)
        )
        nn.init.kaiming_uniform_(self.shared_tokens, a=math.sqrt(5))

        # Check if clip_model is provided
        if clip_model is None:
            raise ValueError("clip_model must be provided for vision projection")

        # Linear layers to map visual and text features to shared space
        self.vision_proj = lambda x: clip_model.visual_projection(x)
        self.text_proj = nn.Identity()
        # self.vision_proj = nn.Linear(visual_dim, shared_dim)
        # self.text_proj = nn.Linear(text_dim, shared_dim)

        # Scale factor for attention scores
        self.scale_factor = shared_dim ** -0.5

    def forward(self, visual_features=None, text_features=None):
        """
        Args:
            visual_features: Visual features [batch_size, num_visual_tokens, visual_dim] (optional)
            text_features: Text features [num_classes, num_text_tokens, text_dim] (optional)

        Returns:
            Token attention maps and extracted features for both visual and text modalities
        """
        result = {}

        # Process visual features
        if visual_features is not None:
            batch_size = visual_features.shape[0]
            v_proj = self.vision_proj(visual_features)  # [batch, num_visual, shared_dim]

            # Prepare shared tokens for visual branch
            v_shared_tokens = self.shared_tokens.unsqueeze(0).expand(batch_size, -1, -1)

            # Compute attention between visual features and shared tokens
            v_attn = torch.matmul(v_shared_tokens, v_proj.transpose(-1, -2)) * self.scale_factor
            v_attn = F.softmax(v_attn, dim=-1)

            # Extract aligned visual features
            v_aligned_features = torch.bmm(v_attn, v_proj)  # [batch, num_tokens, shared_dim]

            result['visual_attn'] = v_attn
            result['visual_aligned'] = v_aligned_features

        # Process text features
        if text_features is not None:
            num_classes = text_features.shape[0]
            t_proj = self.text_proj(text_features)  # [num_classes, num_text, shared_dim]

            # Prepare shared tokens for text branch
            t_shared_tokens = self.shared_tokens.unsqueeze(0).expand(num_classes, -1, -1)

            # Compute attention between text features and shared tokens
            t_attn = torch.matmul(t_shared_tokens, t_proj.transpose(-1, -2)) * self.scale_factor
            t_attn = F.softmax(t_attn, dim=-1)

            # Extract aligned text features
            t_aligned_features = torch.bmm(t_attn, t_proj)  # [num_classes, num_tokens, shared_dim]

            result['text_attn'] = t_attn
            result['text_aligned'] = t_aligned_features

        return result

    def compute_alignment_loss(self, visual_aligned=None, text_aligned=None):
        """Compute alignment loss - using similarity of global features

        Args:
            visual_aligned: Aligned visual features [batch, num_tokens, dim]
            text_aligned: Aligned text features [num_classes, num_tokens, dim]

        Returns:
            alignment_loss: Alignment loss value
        """
        # If either modality feature is missing, return zero loss
        if visual_aligned is None or text_aligned is None:
            return torch.tensor(0.0, device=self.shared_tokens.device)

        # Compute global features
        visual_global = visual_aligned.mean(dim=0)  # [num_tokens, dim]
        text_global = text_aligned.mean(dim=0)  # [num_tokens, dim]

        # Compute similarity loss for each token position
        loss = 0
        for i in range(self.num_tokens):
            v_feat = visual_global[i]  # [dim]
            t_feat = text_global[i]  # [dim]

            # Compute cosine similarity, goal is to maximize similarity, i.e., minimize 1-similarity
            sim = F.cosine_similarity(v_feat.unsqueeze(0), t_feat.unsqueeze(0), dim=1)  # [1]
            token_loss = (1 - sim.mean())
            loss += token_loss * 0.2

        return loss / self.num_tokens


class MAGManager:
    """MAG Manager for coordinating multi-modality alignment processing between visual and text branches"""

    def __init__(self, mag_modules):
        """
        Args:
            mag_modules: List of MAG modules
        """
        self.mag_modules = mag_modules
        self.visual_features_cache = {}  # Cache visual features
        self.text_features_cache = {}  # Cache text features
        self.layer_results = {}  # Store MAG processing results for each layer
        self.stage = 0  # Mark current processing stage: 0-collection stage, 1-usage stage

    def cache_visual_features(self, layer_idx, features):
        """Cache visual features

        Args:
            layer_idx: Layer index
            features: Features [batch, num_tokens, dim]
        """
        self.visual_features_cache[layer_idx] = features

    def cache_text_features(self, layer_idx, features):
        """Cache text features

        Args:
            layer_idx: Layer index
            features: Features [batch, num_tokens, dim]
        """
        self.text_features_cache[layer_idx] = features

    def process_all_alignments(self):
        """Process alignments for all layers, should be called after both modality features are cached"""
        for layer_idx in range(len(self.mag_modules)):
            if (layer_idx in self.visual_features_cache and
                    layer_idx in self.text_features_cache):
                visual_features = self.visual_features_cache[layer_idx]
                text_features = self.text_features_cache[layer_idx]

                mag_module = self.mag_modules[layer_idx]
                mag_outputs = mag_module(visual_features, text_features)

                alignment_loss = mag_module.compute_alignment_loss(
                    mag_outputs['visual_aligned'],
                    mag_outputs['text_aligned']
                )

                self.layer_results[layer_idx] = {
                    'visual_aligned': mag_outputs['visual_aligned'],
                    'text_aligned': mag_outputs['text_aligned'],
                    'visual_attn': mag_outputs['visual_attn'],
                    'text_attn': mag_outputs['text_attn'],
                    'alignment_loss': alignment_loss
                }

        # Set stage marker to 1, indicating alignment has been processed and aligned features can be used
        self.stage = 1

    def get_result(self, layer_idx):
        """Get MAG processing result for a specific layer

        Args:
            layer_idx: Layer index

        Returns:
            result: Processing result, returns None if not exists
        """
        return self.layer_results.get(layer_idx, None)

    def clear_cache(self):
        """Clear cache"""
        self.visual_features_cache.clear()
        self.text_features_cache.clear()
        self.layer_results.clear()
        self.stage = 0

    def compute_total_loss(self):
        """Compute total alignment loss across all layers

        Returns:
            total_loss: Total alignment loss
        """
        total_loss = 0
        count = 0

        for result in self.layer_results.values():
            if 'alignment_loss' in result:
                total_loss += result['alignment_loss']
                count += 1

        return total_loss / max(count, 1)  # Avoid division by zero