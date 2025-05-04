import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel

# -----------------------------------------------------------------------------
# local imports – image + tabular encoders
# -----------------------------------------------------------------------------

sys.path.append("RETFound_MAE")  # <- adjust if RETFound repo lives elsewhere
from models_vit import RETFound_mae  # type: ignore

from gated_tabTransformer import TabTransformer  # gated variant supplied by user

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _freeze(module: nn.Module):
    """Disable gradients for *all* parameters in the given module."""
    for p in module.parameters():
        p.requires_grad = False

# -----------------------------------------------------------------------------
# Image‑encoders
# -----------------------------------------------------------------------------

class ResNet50Encoder(nn.Module):
    """Backbone = torchvision ResNet‑50 w/ final FC removed."""

    output_dim: int = 2048

    def __init__(self, *, pretrained: bool = True):
        super().__init__()
        self.net = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.net.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, 2048)
        return self.net(x)


class RETFoundEncoder(nn.Module):
    """RETFound MAE (ViT) encoder – loads *encoder‑only* weights."""

    output_dim: int = 1024

    def __init__(self, *, weights_path: str):
        super().__init__()
        self.net = RETFound_mae(img_size=224, num_classes=0)
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        state_dict = {k: v for k, v in checkpoint['model'].items() 
                     if not k.startswith("decoder") and "mask_token" not in k}
        self.net.load_state_dict(state_dict, strict=True)
        print(f"Loaded RETFound MAE pretrained weights from {weights_path}")
        print(f"[RETFound] loaded {len(state_dict):,} encoder weights from {weights_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, 1024)
        return self.net(x)

# -----------------------------------------------------------------------------
# Cross‑modal fusion
# -----------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Single Multi‑Head Cross‑Attention block + residual LN."""

    def __init__(self, *, hidden_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:  # (B, 1, D)
        out, _ = self.cross_attn(query=q, key=kv, value=kv)
        return self.ln(q + out)

# -----------------------------------------------------------------------------
# Multi‑modal backbone
# -----------------------------------------------------------------------------

class MultiModalModel(nn.Module):
    """Image‑encoder ✕ gated TabTransformer → cross‑attention → BERT decoder → MLP head."""

    def __init__(
        self,
        *,
        category_dims: List[int],
        num_continuous: int,
        num_classes: int,
        image_encoder_type: str = "retfound",
        retfound_weights: str = "RETFound_MAE/RETFound_mae_natureOCT.pth",
        tab_dim: int = 64,
        hidden_dim: int = 768,
        num_heads: int = 8,
        freeze_encoders: bool = True,
        tab_pretrained: Optional[str] = None,
    ):
        super().__init__()

        # ---------------- image encoder ----------------------------- #
        if image_encoder_type == "resnet50":
            self.img_enc = ResNet50Encoder(pretrained=True)
        elif image_encoder_type == "retfound":
            self.img_enc = RETFoundEncoder(weights_path=retfound_weights)
        else:
            raise ValueError(f"unknown image_encoder_type={image_encoder_type}")

        # project image features to hidden_dim
        self.img_proj = nn.Linear(self.img_enc.output_dim, hidden_dim)

        # ---------------- gated tab encoder ------------------------- #
        # Use the gated TabTransformer with dim=tab_dim
        self.tab_enc = TabTransformer(
            categories=category_dims,  # passed orig_category_dims from factory
            num_continuous=num_continuous,
            dim=64,                     # pretrained embedding dim
            depth=4,
            heads=4,                    # pretrained number of attention heads
            dim_out=64,                  # pretrained output dim (num classes)
        )

                # optionally load TabTransformer pre‑training
        if tab_pretrained and Path(tab_pretrained).is_file():
            # load checkpoint (could be dict or state_dict)
            ckpt = torch.load(tab_pretrained, map_location=torch.device('cpu'), weights_only=False)
            state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            # filter: drop classification head and any mismatched shapes
            model_state = self.tab_enc.state_dict()
            enc_state = {}
            for k, v in state.items():
                if k.startswith('mlp'):
                    continue
                if k in model_state and v.shape == model_state[k].shape:
                    enc_state[k] = v
            missing, unexpected = self.tab_enc.load_state_dict(enc_state, strict=False)
            print(f"[TabTransformer] loaded encoder weights ({len(enc_state)} keys) from {tab_pretrained} | missing={len(missing)}, unexpected={len(unexpected)}")

        # freeze encoders if requested if requested
        if freeze_encoders:
            _freeze(self.img_enc)
            _freeze(self.tab_enc)

        # project tab features to match hidden_dim
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)

        # ---------------- fusion + decoder -------------------------- #
        self.fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads)
        self.decoder = BertModel.from_pretrained("bert-base-uncased")
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )

    def forward(
        self,
        image: torch.Tensor,          # (B, C, 224, 224)
        x_categ: torch.Tensor,        # (B, N_cat)
        x_cont: torch.Tensor,         # (B, N_cont)
    ) -> torch.Tensor:               # → logits (B, num_classes)

        # encode image and project
        img_feat = self.img_proj(self.img_enc(image)).unsqueeze(1)   # (B, 1, hidden_dim)

        # encode tabular and project
        tab_raw = self.tab_enc(x_categ, x_cont)                     # (B, tab_dim)
        tab_feat = self.tab_proj(tab_raw).unsqueeze(1)               # (B, 1, hidden_dim)

        # cross‑attention fusion (tab queries image)
        fused = self.fusion(q=tab_feat, kv=img_feat)                 # (B, 1, hidden_dim)

        # prepare for BERT decoder
        seq = torch.cat([fused, tab_feat], dim=1)                    # (B, 2, hidden_dim)
        attn_mask = torch.ones(seq.shape[:2], dtype=torch.long, device=seq.device)
        bert_out = self.decoder(inputs_embeds=seq, attention_mask=attn_mask)
        cls_tok = bert_out.last_hidden_state[:, 0, :]                # (B, hidden_dim)

        # MLP head
        return self.cls(cls_tok)

# -----------------------------------------------------------------------------
# factory helpers (used by train.py)
# -----------------------------------------------------------------------------

def create_model(args, dataset):
    if args.model_type == "multimodal":
        return _create_multimodal(args, dataset)
    if args.model_type == "image_only":
        return _create_image_only(args, dataset)
    if args.model_type == "tabular_only":
        return _create_tab_only(args, dataset)
    raise ValueError(f"unknown model_type={args.model_type}")


def _create_multimodal(args, dataset):
    # Hard-coded to match your gated TabTransformer pretraining
    orig_category_dims = [29, 2, 2, 3, 4, 2, 2, 2, 2]
    model = MultiModalModel(
        category_dims=orig_category_dims,           # exactly the same as pretraining
        num_continuous=len(dataset.continuous_cols),
        num_classes=dataset.get_num_classes(),      # 6
        image_encoder_type=args.image_encoder_type,
        retfound_weights=args.retfound_weights,
        tab_dim=64,                                 # must match pretrained dim
        hidden_dim=args.hidden_dim,
        num_heads=4,                                # must match pretrained heads
        freeze_encoders=True,
        tab_pretrained="./tab_weights/gated_tabtransformer_weights_6classes.pth",
    )
    # summary
    tot = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params: {tot:,} | trainable: {train:,} ({train/tot:.2%})")
    return model


def _create_image_only(args, dataset):
    out_dim = 2048 if args.image_encoder_type == "resnet50" else 1024
    if args.image_encoder_type == "resnet50":
        enc = ResNet50Encoder(pretrained=True)
    elif args.image_encoder_type == "retfound":
        enc = RETFoundEncoder(weights_path=args.retfound_weights)
    else:
        raise ValueError
    head = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, dataset.get_num_classes()))
    model = nn.Sequential(enc, head)
    if args.freeze_encoders:
        _freeze(enc)
    return model


# def _create_tab_only(args, dataset):
#     tab_enc = TabTransformer(
#     categories=categories,
#     num_continuous=len(cont_cols),
#     dim=64, depth=4, heads=4,
#     dim_out=num_classes
# )
#     clf = nn.Sequential(nn.Linear(args.tab_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, dataset.get_num_classes()))
#     model = nn.Sequential(tab_enc, clf)
#     if args.freeze_encoders:
#         _freeze(tab_enc)
#     return model
