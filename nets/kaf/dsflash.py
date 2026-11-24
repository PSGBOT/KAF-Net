"""DSFlash pair-conditioned relations for the undirected PSR label space."""

from functools import partial
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nets.kaf.fpn import get_fpn
from nets.kaf.kaf_swint import SwinV2Features


def _pool_spatial_tokens(features, grid):
    # Torch 1.13 can accumulate in the input dtype for CUDA adaptive pooling.
    if features.dtype in (torch.float16, torch.bfloat16):
        features = features.float()
    return F.adaptive_avg_pool2d(features, grid)


class RelationOutput(NamedTuple):
    logits: torch.Tensor
    swapped_logits: torch.Tensor | None
    consistency: torch.Tensor


class DSFlashHead(nn.Module):
    """Condition a shared spatial feature map on each pair of independent masks."""

    def __init__(self, num_relations, feature_channels, width, depth, heads, patch_grid, pair_chunk=32):
        super().__init__()
        self.num_relations = num_relations
        self.patch_grid = patch_grid
        self.pair_chunk = pair_chunk
        self.project = nn.Linear(feature_channels, width)
        self.mask_role_embedding = nn.Parameter(torch.empty(3, width))
        self.class_token = nn.Parameter(torch.empty(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, patch_grid ** 2 + 1, width))
        self.blocks = nn.Sequential(*(
            nn.TransformerEncoderLayer(
                width, heads, dim_feedforward=4 * width, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ))
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, num_relations))
        for parameter in (self.mask_role_embedding, self.class_token, self.position):
            nn.init.trunc_normal_(parameter, std=0.02)

    def _encode_pairs(self, image_tokens, first, second, background):
        coverage = torch.stack((first, second, background), dim=-1).to(image_tokens.dtype)
        patch_tokens = image_tokens + coverage @ self.mask_role_embedding.to(image_tokens.dtype)
        class_tokens = self.class_token.to(image_tokens.dtype).expand(len(image_tokens), -1, -1)
        tokens = torch.cat((class_tokens, patch_tokens), dim=1) + self.position.to(image_tokens.dtype)
        representation = self.norm(self.blocks(tokens)[:, 0])
        gate = self.gate(representation).sigmoid()
        return gate * representation, (1.0 - gate) * representation

    def forward(self, features, masks, pairs):
        if len(pairs) == 0:
            # A rank with no valid pairs still participates in the DDP backward pass.
            zero = sum(p.reshape(-1)[0] * 0.0 for p in self.parameters()) + features.sum() * 0.0
            logits = features.new_zeros((0, self.num_relations)) + zero
            return RelationOutput(logits, logits if self.training else None, features.new_zeros(0))

        grid = (self.patch_grid, self.patch_grid)
        image_tokens = _pool_spatial_tokens(features, grid).flatten(2).transpose(1, 2)
        image_tokens = self.project(image_tokens)
        coverage = F.adaptive_avg_pool2d(masks.float(), grid).flatten(2)

        encode = self._encode_pairs
        if self.training and torch.is_grad_enabled() and len(pairs) > self.pair_chunk:
            # Recompute the attention activations in backward when several chunks are needed.
            encode = partial(checkpoint, encode, use_reentrant=False, preserve_rng_state=False)
        logits, swapped_logits, consistency = [], [], []
        for chunk in pairs.split(self.pair_chunk):
            image, first, second = chunk.unbind(dim=1)
            tokens = image_tokens[image]
            first_coverage, second_coverage = coverage[image, first], coverage[image, second]
            # Background is the complement of the union, including when the masks overlap.
            union = masks[image, first] | masks[image, second]
            background = 1.0 - F.adaptive_avg_pool2d(union.float(), grid).flatten(1)
            forward, backward = encode(tokens, first_coverage, second_coverage, background)
            scores = (self.classifier(forward) + self.classifier(backward)) * 0.5
            if self.training:
                swapped_forward, swapped_backward = encode(tokens, second_coverage, first_coverage, background)
                swapped_scores = (self.classifier(swapped_forward) + self.classifier(swapped_backward)) * 0.5
                swapped_logits.append(swapped_scores)
                mismatch = (forward.float() - swapped_backward.float()).square()
                mismatch += (backward.float() - swapped_forward.float()).square()
                consistency.append(mismatch.mean(dim=-1))
            logits.append(scores)
        logits = torch.cat(logits, dim=0)
        return RelationOutput(
            logits,
            torch.cat(swapped_logits) if self.training else None,
            torch.cat(consistency) if self.training else logits.new_zeros(len(pairs)),
        )


class DSFlashSwin(nn.Module):
    def __init__(self, num_relations, image_size, width, depth, heads, patch_grid, pair_chunk, pretrained=True):
        super().__init__()
        self.encoder = SwinV2Features(image_size=image_size, pretrained=pretrained)
        self.fpn = get_fpn(96)
        self.head = DSFlashHead(num_relations, 4 * 256, width, depth, heads, patch_grid, pair_chunk)

    def forward(self, images, masks, pairs):
        pyramid = self.fpn(self.encoder(images))
        grid = (self.head.patch_grid, self.head.patch_grid)
        features = torch.cat([_pool_spatial_tokens(level, grid) for level in pyramid], dim=1)
        return self.head(features, masks, pairs)
